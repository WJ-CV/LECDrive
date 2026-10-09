import torch
from vggt.models.vggt import VGGT
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F
from einops import rearrange
from torch.cuda.amp import autocast
from PIL import Image
from torchvision import transforms as TF

def load_and_preprocess_images(image_path_list, resized_H, resized_W, mode="crop"):
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    new_height = resized_H
    new_width = resized_W

    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        img = img.convert("RGB")

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)
    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


def fetch_img_list_navsim(example):
    process_img_list = []
    for exp in example["messages"]:
        if exp.get("role") == "user":
            content_list = exp.get("content", [])
    for content in content_list:
        if 'image' not in content:
            continue
        image_path = content['image']
        resized_height, resized_width = content["resized_height"], content["resized_width"]
        img_path = image_path.replace("file://", "")
        process_img_list.append(img_path)

    process_imgs = load_and_preprocess_images(process_img_list, resized_height, resized_width)

    return process_imgs # [3, 3, 294, 518]

class VGGTWrapper:
    _instance = None
    
    @classmethod
    def get_instance(cls, device='cpu'):
        if cls._instance is None:
            cls._instance = VGGT()
            vggt_state_dict = torch.load('vggt/model.pt', map_location='cpu')
            cls._instance.load_state_dict(vggt_state_dict)
            cls._instance.eval()
            for param in cls._instance.parameters():
                param.requires_grad = False
            cls._instance._is_vggt_inference = True
        return cls._instance.to(device)

def run_vggt_inference(images, latent_depth=None, device='cuda'): # [B, 3, 3200, 2048]
    """执行 VGGT 推理并返回不追踪梯度的结果"""
    vggt_model = VGGTWrapper.get_instance(device)
    pre_depth = None
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        aggregated_tokens_list, ps_idx = vggt_model.aggregator(images)
        depth, depth_conf = vggt_model.depth_head(
            aggregated_tokens_list, images=images, patch_start_idx=ps_idx
        )
        final_token = aggregated_tokens_list[-1].detach()
        final_token = final_token.to(device).clone()

        if latent_depth is not None:
            latent_depth_in = torch.cat((final_token[:, :, :5, :].clone().detach(), latent_depth), 2)
            clone_vggt_list = [latent_depth_in for i in range(len(aggregated_tokens_list))]
            # clone_vggt_list = [tensor.clone().detach() for tensor in aggregated_tokens_list]
            # clone_vggt_list[-1] = latent_depth_in

            pre_depth, pre_depth_conf = vggt_model.depth_head(
            clone_vggt_list, images=images, patch_start_idx=ps_idx
            )

    return final_token, pre_depth, depth

class UpsampleWithPixelUnshuffle(nn.Module):
    def __init__(self, in_channels=3584, factor_1=2, factor_2=5, mid_channel=96, out_channel=2048):
        super().__init__()
        self.factor_1 = factor_1
        self.factor_2 = factor_2
        self.in_channels = in_channels
        self.mid_channel = mid_channel
        self.out_channel = out_channel

        self.linear_1 = nn.Linear(self.in_channels // (self.factor_1 ** 2), self.mid_channel * (self.factor_2 ** 2))
        self.out_mlp = nn.Sequential(
                        nn.Linear(self.mid_channel, self.mid_channel * 4),
                        nn.GELU(),
                        nn.Linear(self.mid_channel * 4, self.out_channel),
                        )
    def forward(self, x):  # x: [B, 3584, 4, 8]
        B = x.shape[0]
        x_1 = F.pixel_shuffle(x, upscale_factor=self.factor_1)  # -> [B, 896, 8, 16]
        x_trans_1 = x_1.flatten(2, 3).permute(0, 2, 1)

        x_2 = self.linear_1(x_trans_1).permute(0, 2, 1) 
        x_2_trans = x_2.view(B, -1, x_1.shape[2], x_1.shape[3]) # [B, 96*25, 8, 16]

        x_3 = F.pixel_shuffle(x_2_trans, upscale_factor=self.factor_2)  # -> [B, 96, 40, 80]
        x_trans_3 = x_3.flatten(2, 3).permute(0, 2, 1) 
        feat_out = self.out_mlp(x_trans_3) # # -> [B, 3200, 2048]

        return feat_out

class VGGTfeatprojection(nn.Module):
    def __init__(
        self,
        n_imgs=3,
        patch_w=8,
        patch_h=4,
        embed_dims=3584,
        patch_size=10,
        factor_1=2,
        factor_2=5,
        mid_channel=96,
    ):
        super().__init__()
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.n_imgs = n_imgs
        self.patch_size = patch_size
        self.embed_dims = embed_dims
        self.mid_channel = mid_channel

        self.feat_unsample_vggt = UpsampleWithPixelUnshuffle(in_channels=self.embed_dims, factor_1=factor_1, factor_2=factor_2, mid_channel=self.mid_channel)

    def forward(self, depth_hidden_states): # [B, 24, 3584]  [B, 1, 448, 896]
        B, n_token, dim = depth_hidden_states.shape

        depth_feat = depth_hidden_states.view(B, self.n_imgs, self.patch_h, self.patch_w, -1)
        depth_feat = depth_feat.permute(0, 1, 4, 2, 3).flatten(0,1)

        depth_feat_ = self.feat_unsample_vggt(depth_feat)
        depth_feat_out = depth_feat_.view(B, self.n_imgs, depth_feat_.shape[1], depth_feat_.shape[2])

        return depth_feat_out # -> [B, 3, 3200, 2048]


class UpsampleWithPixelUnshuffle_1(nn.Module):
    def __init__(self, in_channels=3584, factor_1=8, factor_2=7, factor_3=5, out_channels=32):
        super().__init__()
        self.factor_1 = factor_1
        self.factor_2 = factor_2
        self.factor_3 = factor_3
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear_1 = nn.Linear(self.in_channels // (self.factor_1 ** 2), self.out_channels * (self.factor_2 ** 2))
        self.linear_2 = nn.Linear(self.out_channels, self.out_channels * (self.factor_3 ** 2))
        self.linear_out = nn.Linear(self.out_channels, 1)

    def forward(self, x):  # x: [B, 3584, 2, 4]
        B = x.shape[0]
        x_1 = F.pixel_shuffle(x, upscale_factor=self.factor_1)  # -> [B, 56, 16, 32]
        x_trans_1 = x_1.flatten(2, 3).permute(0, 2, 1)

        x_2 = self.linear_1(x_trans_1).permute(0, 2, 1) 
        x_2_trans = x_2.view(B, -1, x_1.shape[2], x_1.shape[3]) # [B, 32*49, 16, 32]

        x_3 = F.pixel_shuffle(x_2_trans, upscale_factor=self.factor_2)  # -> [B, 32, 112, 224]
        x_trans_3 = x_3.flatten(2, 3).permute(0, 2, 1) 

        x_4 = self.linear_2(x_trans_3).permute(0, 2, 1)
        x_4_trans = x_4.view(B, -1, x_3.shape[2], x_3.shape[3]) # [B, 32*49, 112, 224]

        x_5 = F.pixel_shuffle(x_4_trans, upscale_factor=self.factor_3)  # -> [B, 32, 448, 896]
        x_trans_5 = x_5.flatten(2, 3).permute(0, 2, 1)

        x_out = self.linear_out(x_trans_5).permute(0, 2, 1)
        x_out = x_out.view(B, -1, x_5.shape[2], x_5.shape[3])

        return x_out

class DenseMapNet_1(nn.Module):
    def __init__(
        self,
        n_imgs=3,
        patch_w=4,
        patch_h=2,
        embed_dims=3584,
        patch_size=280,
        factor_1=8,
        factor_2=7,
        factor_3=5,
        out_dims=32,
        w_depth=0.5
    ):
        super().__init__()
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.n_imgs = n_imgs
        self.patch_size = patch_size
        self.embed_dims = embed_dims
        self.out_dims = out_dims
        self.H = self.patch_h * self.patch_size
        self.W = self.patch_w * self.patch_size
        self.w_depth = w_depth

        self.feat_unsample_depth = UpsampleWithPixelUnshuffle_1(in_channels=self.embed_dims, factor_1=factor_1, factor_2=factor_2, factor_3=factor_3, out_channels=self.out_dims)

    def forward(self, depth_hidden_states, gt_depths_map): # [B, 24, 3584]  [B, 1, 448, 896]
        B, n_token, dim = depth_hidden_states.shape

        depth_feat = depth_hidden_states.view(B, self.n_imgs, self.patch_h, self.patch_w, -1)
        depth_feat = depth_feat.permute(0, 1, 4, 2, 3)

        depth_f = depth_feat.clone().view(B * self.n_imgs, dim, self.patch_h, self.patch_w)

        depth_maps = self.feat_unsample_depth(depth_f)

        depth_map_loss = self.w_depth * self.compute_depth_map_loss(depth_maps, gt_depths_map)

        return depth_map_loss

    def compute_depth_map_loss(self, depth_preds, gt_depths):
        loss = 0.0
        pred = depth_preds.reshape(-1)
        gt = gt_depths.reshape(-1)
        fg_mask = torch.logical_and(
            gt > 0.0, torch.logical_not(torch.isnan(pred))
        )
        gt = gt[fg_mask]
        pred = pred[fg_mask]
        with autocast(enabled=False):
            error = torch.abs(pred - gt).sum()
            _loss = (
                error
                / max(1.0, len(gt) * len(depth_preds))
            )
        loss = loss + _loss
        return loss

    def traj_loss(self, traj_heatmap_preds, gt_traj_heatmap): # [B, 1, H, W]
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(traj_heatmap_preds, gt_traj_heatmap)
        return loss

    def compute_traj_heatmap_loss(self, traj_heatmap_pred, gt_traj_heatmap, use_focal=True):
        """
        traj_heatmap_pred: Tensor of shape [B, 1, H, W], logits (not passed through sigmoid)
        gt_traj_heatmap: Tensor of shape [B, 1, H, W], uint8 or float, values in [0, 255]
        use_focal: whether to use focal loss instead of BCE
        """
        # 1. Ensure float32 and normalize GT to [0,1]
        pred_probs = torch.sigmoid(traj_heatmap_pred)
        gt_traj_heatmap = gt_traj_heatmap.float() / 255.0
        gt_traj_heatmap = torch.clamp(gt_traj_heatmap, 0.0, 1.0)  # 防止极小负数或溢出

        # 2. Compute loss
        if use_focal:
            # Optional focal loss
            bce_loss = nn.BCEWithLogitsLoss(reduction='none')
            bce = bce_loss(pred_probs, gt_traj_heatmap)
            pt = torch.exp(-bce)
            focal_loss = (1 - pt) ** 2 * bce
            loss = focal_loss.mean()
        else:
            # Standard BCE loss
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(pred_probs, gt_traj_heatmap)

        return loss

