import torch
import cv2
import sys
import os
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import deepspeed
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2Config
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import List, Optional, Tuple, Union, Dict, Any
from inject_utils.pre_map_head_mlp import DenseMapNet, DenseMapNet_1
from inject_utils.dino3_utils import Dinodecoer_MHCA_projector, Dinodecoer
from inject_utils.sam3_utils import init_sam_model, run_sam_inference, GenerateSamMask
from inject_utils.da3_utils import DA3genDenseMap
from transformers import AutoImageProcessor, AutoModel
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

_global_da3_model = None
def _get_da3_model():
    global _global_da3_model 
    with torch.no_grad():
        if _global_da3_model is not None:
            return _global_da3_model
        da3_model = DepthAnything3.from_pretrained("depth_anything_3/pretrain")
        for param in da3_model.parameters():
            param.requires_grad = False
        da3_model.eval()
        _global_da3_model = da3_model
    
    return da3_model

@dataclass
class Qwen2_5_VLCausalLMOutputWithPast_(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    depth_map_loss: Optional[torch.FloatTensor] = None
    dino_loss_l2: Optional[torch.FloatTensor] = None
    sam_loss_CE: Optional[torch.FloatTensor] = None

class CustomQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.resized_W = kwargs.get("resized_W", 896)
        self.resized_H = kwargs.get("resized_H", 448)
        self.num_view = kwargs.get("num_views", 6)
        self.output_dir = kwargs.get("output_dir", './')
        self.hidden_size = getattr(config, "hidden_size", 3584)
        self.max_window_layers = getattr(config, "max_window_layers", 28)

        self.use_dino_head = kwargs.get("use_dinov3", False)
        self.use_sam_head = kwargs.get("use_sam3", False)
        self.use_depth_head = kwargs.get("use_depth", False)
        self.check_depth = False
        self.check_pre_mask = True
        self.w_dinov3 = float(kwargs.get("w_dinov3", 0.5))
        self.w_sam3 = float(kwargs.get("w_sam3", 0.3))
        self.w_da3 = float(kwargs.get("w_da3", 1.0))
        
        self.dino_model = None 
        self.dino_pretrain_path = kwargs.get("dino_pretrain_path", 'dinov3/pretrain/dinov3-vith16plus-pretrain-lvd1689m')
        if self.training and self.use_dino_head:
            if "dino_token_id" not in kwargs:
                raise KeyError("dino_token_id is required but not provided.")
            self.dino_token_id = kwargs["dino_token_id"]
            self.DinoDecoer = Dinodecoer(
                n_imgs=self.num_view,
                patch_w=4,
                patch_h=2,
                embed_dims=self.hidden_size * 4,
                patch_size=16,
                mid_dims=224, #
                out_dims=1280, # 1280
                factor_1=8,
                factor_2=2,
                w_dino=self.w_dinov3,
                mse_loss=True
            )
            self._init_dino_model()

        self.use_sam_head = kwargs.get("use_sam3", False)
        if self.training and self.use_sam_head:
            self.sam_pretrain_path = kwargs.get("sam_pretrain_path", 'sam3/pretrain/sam3/sam3.pt')
            if "sam_token_id" not in kwargs:
                raise KeyError("sam_token_id is required but not provided.")
            self.sam_token_id = kwargs["sam_token_id"]
            self.SemanticMapNet = GenerateSamMask(
                n_imgs=self.num_view,
                sem_classes=7,
                patch_w=4,
                patch_h=2,
                embed_dims=self.hidden_size * 4,
                factor_1=8,
                factor_2=4,
                factor_3=4,
                out_dims=224,
                w_sam=self.w_sam3
            )

        if self.training and self.use_depth_head:
            if "depth_token_id" not in kwargs:
                raise KeyError("depth_token_id is required but not provided.")
            self.depth_token_id = kwargs["depth_token_id"]
            self.DenseMapNet = DA3genDenseMap(
                n_imgs=self.num_view,
                patch_w=4,
                patch_h=2,
                embed_dims=self.hidden_size * 4,
                factor_1=8,
                factor_2=4,
                factor_3=4,
                out_dims=224, # 224
                w_depth=self.w_da3
            )

    def _init_dino_model(self):
        """初始化DINO模型, 避免与Deepspeed Zero-3冲突"""
        self.dino_model = AutoModel.from_pretrained(
            self.dino_pretrain_path,
            dtype=torch.float32,
        )
        for param in self.dino_model.parameters():
            param.requires_grad = False
        self.dino_model.eval()
        
        if hasattr(self.dino_model, 'module'):
            for param in self.dino_model.module.parameters():
                param._no_deepspeed = True
                if hasattr(param, 'ds_id'):
                    param.ds_status = ZeroParamStatus.NOT_AVAILABLE
        else:
            for param in self.dino_model.parameters():
                param._no_deepspeed = True
                if hasattr(param, 'ds_id'):
                    param.ds_status = ZeroParamStatus.NOT_AVAILABLE
        self.dino_model._no_deepspeed = True

    def to(self, device=None, dtype=None, non_blocking=False):
        result = super().to(device=device, dtype=dtype, non_blocking=non_blocking)
        if self.dino_model is not None:
            self.dino_model = self.dino_model.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return result

    def generate(self, *args, **kwargs):
        self._img_list = kwargs.pop("img_list", None)
        self._sam_path = kwargs.pop("sam_path", None)
        self._da3_input = kwargs.pop("da3_input", None)
        self._intrinsic = kwargs.pop("intrinsic", None)
        self._extrinsic = kwargs.pop("extrinsic", None)
        self._depth_maps = kwargs.pop("depth_maps", None)
        return super().generate(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        img_list=None,
        sam_path=None,
        da3_input=None,
        intrinsic=None,
        extrinsic=None,
        depth_maps=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        pixel_values_videos=None,
        logits_to_keep=None,** kwargs,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True, # output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )
        last_hidden_states = outputs[0] # [2, 2219, 3584]
        logits = self.lm_head(last_hidden_states) # [2, 2219, 152064]
        loss = None
        depth_map_loss_l1 = None
        sam_sem_loss_CE = None
        dino_feat_loss_l2 = None

        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous() # [2, 2198, 152064]
            shift_labels = labels[..., 1:].contiguous() # [2, 2199] --> [2, 2198]
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size) # [b*2198, 152064]
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if self.training:
            multi_layer_hidden_states = [outputs[1][4], outputs[1][10], outputs[1][18], outputs[1][28]]
            multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim=-1)
            #########  DINO v3 ############
            dino_token_mask = None
            if self.use_dino_head and input_ids is not None:
                dino_token_mask = input_ids == self.dino_token_id # [B, N]
            if self.use_dino_head and dino_token_mask is not None:
                dino_hidden_states = self.get_expert_hs(multi_layer_hidden_states, dino_token_mask)

                if img_list is None:
                    img_list = getattr(self, "_img_list", None)
                dino_visual = self._run_dino_inference(
                    img_list,
                    device=last_hidden_states.device
                ) # B*S, 800, 1280
                dino_feat_loss_l2 = self.DinoDecoer(dino_hidden_states, dino_visual.to(dino_hidden_states.dtype).to(dino_hidden_states.device))
                
                loss = loss + dino_feat_loss_l2
            else:
                dino_hidden_states = None

            #########  SAM v3 ############
            sam_token_mask = None
            if self.use_sam_head and input_ids is not None:
                sam_token_mask = input_ids == self.sam_token_id # [B, N]

            if self.use_sam_head and sam_token_mask is not None:
                sam_hidden_states = self.get_expert_hs(multi_layer_hidden_states, sam_token_mask)

                if sam_path is None:
                    sam_path = getattr(self, "_sam_path", None)

                sam_pseudo_label = torch.tensor(sam_path, dtype=torch.bfloat16, device=sam_hidden_states.device) # B, 3, 560, 1120
                sam_sem_loss_CE = self.SemanticMapNet(sam_hidden_states, sam_pseudo_label, check_pre_mask=self.check_pre_mask, out_dir=self.output_dir)
                loss = loss + sam_sem_loss_CE
            else:
                sam_hidden_states = None

            #########  DA v3 ############
            depth_token_mask = None
            if self.use_depth_head and input_ids is not None:
                depth_token_mask = input_ids == self.depth_token_id # [B, N]

            if self.use_depth_head and depth_token_mask is not None:
                depth_hidden_states = self.get_expert_hs(multi_layer_hidden_states, depth_token_mask)

                if da3_input is None:
                    da3_input = getattr(self, "_da3_input", None)

                if da3_input is None:
                    if depth_maps is None:
                        depth_maps = getattr(self, "_depth_maps", None)
                    depth_maps = torch.tensor(depth_maps, dtype=torch.bfloat16, device=depth_hidden_states.device) # B, 3, 448, 896
                    depth_map_loss_l1 = self.DenseMapNet(depth_hidden_states, depth_maps.to(depth_hidden_states.dtype).to(depth_hidden_states.device), check_depth=self.check_depth, out_dir=self.output_dir, lidar_depth=True)
                else:
                    if intrinsic is None:
                        intrinsic = getattr(self, "_intrinsic", None)
                    intrinsic = np.array(intrinsic)
                    if extrinsic is None:
                        extrinsic = getattr(self, "_extrinsic", None)
                    extrinsic = np.array(extrinsic)

                    da3_pseudo_depth = self._run_da3_inference(da3_input, intrinsic, extrinsic, device=depth_hidden_states.device, check_depth=self.check_depth, out_dir=self.output_dir) # [B, 3, H, W]
                    da3_pseudo_depth = da3_pseudo_depth.clone().detach()
                    depth_map_loss_l1 = self.DenseMapNet(depth_hidden_states, da3_pseudo_depth.to(depth_hidden_states.dtype).to(depth_hidden_states.device), check_depth=self.check_depth, out_dir=self.output_dir)

                loss = loss + depth_map_loss_l1

        return Qwen2_5_VLCausalLMOutputWithPast_(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            depth_map_loss=depth_map_loss_l1,
            dino_loss_l2=dino_feat_loss_l2,
            sam_loss_CE=sam_sem_loss_CE,
        )

    def get_expert_hs(self, hidden_states, expert_token_mask):
        expert_hidden_states = []
        for b in range(hidden_states.size(0)):
            hidden_b = hidden_states[b]
            mask_b = expert_token_mask[b]
            tokens_b = hidden_b[mask_b]
            expert_hidden_states.append(tokens_b)
        if expert_hidden_states:
            expert_hidden_states = torch.stack(expert_hidden_states, dim=0) # [B, 3*8, 3584]

        return expert_hidden_states

    def _run_dino_inference(self, input_list, device='cuda'):
        if not hasattr(self, 'dino_model') or self.dino_model is None:
            self._init_dino_model()

        if str(self.dino_model.device) != str(device):
            self.dino_model = self.dino_model.to(device)
        
        dino_inputs = []
        for i in range(len(input_list)):
            inp = input_list[i]["pixel_values"]
            dino_inputs.append(inp)

        input_tensor = torch.stack(dino_inputs, dim=0)
        input_reshaped = input_tensor.view(-1, *input_tensor.shape[2:])
        input_reshaped = input_reshaped.to(device=device, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = self.dino_model(pixel_values=input_reshaped)
        
        return outputs.last_hidden_state[:, 5:, :]

    def _run_da3_inference(self, input_list, intrinsic, extrinsic, device='cuda', check_depth=True, out_dir='./'):
        da3_model = _get_da3_model()
        if str(da3_model.device) != str(device):
            da3_model = da3_model.to(device)
        da3_depth_out = []
        for i in range(len(input_list)):
            with torch.no_grad():
                prediction = da3_model.inference(
                    input_list[i],
                    extrinsics=extrinsic[i],
                    intrinsics=intrinsic[i],
                    process_res=self.resized_W,
                    process_res_method="upper_bound_resize",
                )
                da3_depth_out.append(prediction[0].depth) # 3 H, W
            if check_depth:
                ############# visualization ############
                save_path = os.path.join(out_dir, 'da3_pre_check')
                os.makedirs(save_path, exist_ok=True)
                n_images = prediction[0].depth.shape[0]
                fig, axes = plt.subplots(2, n_images, figsize=(12, 6))

                if n_images == 1:
                    axes = axes.reshape(2, 1)

                for j in range(n_images):
                    image_path = Path(input_list[i][j])
                    # shutil.copy(image_path, dst_dir)

                    image_name = image_path.name
                    if prediction[0].processed_images is not None:
                        axes[0, j].imshow(prediction[0].processed_images[j])
                    axes[0, j].set_title(f"Input {j+1}")
                    axes[0, j].axis('off')
                    
                    # Show depth map
                    depth_vis = visualize_depth(prediction[0].depth[j], cmap="Spectral")
                    axes[1, j].imshow(depth_vis)
                    axes[1, j].set_title(f"Depth {j+1}")
                    axes[1, j].axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f"da3_depth_{i*3+j}.png"))

        da3_depth_out = [torch.from_numpy(ndarray) for ndarray in da3_depth_out]
        da3_depth = torch.stack(da3_depth_out, dim=0) # B, 3, H, W

        return da3_depth
