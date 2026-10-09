import logging
import os
import sys
import importlib
from collections import defaultdict
from pathlib import Path
from functools import partial
from copy import deepcopy
# from torch.utils.data import Subset
import datasets
import torch
import numpy as np
import random
import transformers
import torch.distributed as dist
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
from typing import Optional
from open_r1.configs import SFTConfig, NuScenesDataConfig, preprocess_data_args, index_data_args
from open_r1.utils.callbacks import get_callbacks
import open_r1.dataloader as DATASETS
from open_r1.dataloader.utils import save_json
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from qwen_vl_utils import process_vision_info
logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from torch.utils.tensorboard import SummaryWriter
# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
from typing import Tuple
# from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from inject_utils.Qwen2_5_vggt_fusion_inject_cam import CustomQwen2_5_VLForConditionalGeneration
from inject_utils.utils import fetch_info, fetch_depth, lidar2img, fetch_map, fetch_img_list, fetch_img_list_navsim

class CustomTrainer(SFTTrainer):
    def __init__(self, *args, log_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = SummaryWriter(log_dir=log_dir)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        dist_loss = outputs.dist_loss.item() if outputs.dist_loss is not None else 0.0

        global_step = self.state.global_step if hasattr(self, "state") else 0
        self.writer.add_scalar("loss/train_loss", (loss - dist_loss).item(), global_step)
        self.writer.add_scalar("loss/dist_loss", dist_loss, global_step)

        self.log({
            "loss": loss.item() - dist_loss,
            "dist_loss": dist_loss,
        })

        return (loss, outputs) if return_outputs else loss

    def teardown(self):
        super().teardown()
        self.writer.close()

def img_resize(data, W=896, H=448):
    for item in data:
        messages = item.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for content_item in content:
                        if content_item.get("type") == "image":
                            content_item["resized_width"] = W
                            content_item["resized_height"] = H
    return data

def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward

processor = None

def collate_fn(examples, prompt_mask, resized_W, resized_H, img_W, img_H):
    examples = img_resize(examples, W=resized_W, H=resized_H)
    texts = [
        processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
        for example in examples
    ]
    image_inputs = []
    vggt_img_list = []
    cam2lidar_inputs = []
    for example in examples:
        imgs, vids = process_vision_info(example["messages"])
        image_inputs.append(imgs)

        cam2lidar = lidar2img(example['gold'], resized_H, resized_W, img_H, img_W)
        cam2lidar_inputs.append(cam2lidar)

        process_img_list = fetch_img_list_navsim(example)
        vggt_img_list.append(process_img_list)

    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )

    ignore_token_ids = [
        processor.tokenizer.pad_token_id,  # 填充token  151643
        processor.tokenizer.convert_tokens_to_ids(processor.image_token),  # 图像token  151655
    ]
    labels = batch["input_ids"].clone()
    for token_id in ignore_token_ids:
        labels[labels == token_id] = -100

    batch["labels"] = labels
    batch["img_list"] = vggt_img_list
    batch["cam2lidar"] = cam2lidar_inputs
    if not prompt_mask:
        return batch

    # 在label中不带Prompt
    question_texts = [
        processor.apply_chat_template([example["messages"][0]], tokenize=False, add_generation_prompt=True)
        for example in examples
    ]

    for idx, i_question_text in enumerate(question_texts):
        cur_question_batch = processor(
            text=[i_question_text],
            images=image_inputs,
            return_tensors="pt",
            padding=False,
        )
        batch["labels"][idx, :cur_question_batch['input_ids'].shape[1]] = -100

    return batch

class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, action_type, samples_per_type, dataset_list, ratio_shuffle, mixed_shuffle):
        self.datasets = dataset_list
        self.action_type = action_type
        self.ratio_shuffle = ratio_shuffle
        self.mixed_shuffle = mixed_shuffle
        self._samples = []
        for dataset in self.datasets:
            self._samples.extend(dataset._samples)

        self.samples_per_type = samples_per_type
        self.update_action_indices()

        if torch.cuda.current_device() == 0:
            print("\n=== Base Data Distribution Statistics ===")
            print(f"Total samples: {len(self._samples)}")
            print(f"Without actions: {len(self.wo_action_indices)}")
            print(f"Number of action classes: {self.num_classes}")
            print("Samples per class:")
            for cur_action in sorted(self.actions):
                print(f"  {cur_action}: {len(self.action_indices[cur_action])}")
            print("="*40 + "\n")

        if self.ratio_shuffle:
            self.local_shuffle()
        else:
            self.local_fixed()

        if self.mixed_shuffle:
            random.shuffle(self.shuffled_samples)
        self.print_shuffled_samples()

    def print_shuffled_samples(self):
        print(f"Current Data Length: {len(self.shuffled_samples)}")
        lat_action_list = [i_sample[self.action_type] for i_sample in self.shuffled_samples if self.action_type in i_sample.keys()]
        print(f"Samples per class [{self.action_type}]:")
        for cur_action in set(lat_action_list):
            print(f"  {cur_action}: {lat_action_list.count(cur_action)}")
        print('Sample 0 & 1 & 2')
        print([self.shuffled_samples[0]])
        print("="*40 + "\n")

    def update_action_indices(self):
        self.action_indices, self.wo_action_indices = defaultdict(list), []
        for idx in range(len(self._samples)):
            if self.action_type in self._samples[idx].keys():
                cur_action = self._samples[idx][self.action_type]
                self.action_indices[cur_action].append(idx)
            else:
                self.wo_action_indices.append(idx)

        self.actions = list(self.action_indices.keys())
        self.num_classes = len(self.actions)

    def local_shuffle(self):
        print('-' * 80)
        print('In local_shuffle ..')
        assert 0
        self.update_action_indices()
        sampled_indices = []
        for cur_action in self.actions:
            indices = self.action_indices[cur_action]
            if len(indices) >= self.samples_per_type[cur_action]:
                selected = np.random.choice(indices, self.samples_per_type[cur_action], replace=False)
            else:
                selected = np.random.choice(indices, self.samples_per_type[cur_action], replace=True)
            sampled_indices.extend(selected.tolist())

        np.random.shuffle(self.wo_action_indices)
        np.random.shuffle(sampled_indices)
        self.shuffled_samples = [self._samples[i] for i in self.wo_action_indices + sampled_indices]

    def local_fixed(self):
        print('-' * 80)
        print('In local_fixed ..')
        self.shuffled_samples = self._samples

    def __len__(self):
        return len(self.shuffled_samples)

    def __getitem__(self, idx):
        return self.shuffled_samples[idx]

def main(script_args, data_args, sft_args, model_args):
    # Set seed for reproducibility
    set_seed(sft_args.seed)
    print('sft_args.output_dir rrrrrrrrrrrrrrrrrrrrrrrr', sft_args.output_dir)

    data_args = preprocess_data_args(data_args)
    saved_args = {
        'model_name_or_path': model_args.model_name_or_path,
        'notes': sft_args.notes,
        'learning_rate': sft_args.learning_rate,
        'weight_decay': sft_args.weight_decay,
        'dataset_name': script_args.dataset_name,
        'dataset_classname': data_args.dataset_classname,
        'per_device_train_batch_size': sft_args.per_device_train_batch_size,
        'warmup_ratio': sft_args.warmup_ratio,
        'ratio_shuffle': data_args.ratio_shuffle,
        'mixed_shuffle': data_args.mixed_shuffle,
        'required_cams': data_args.required_cams,
        'image_description': data_args.image_description,
        'balanced_action': data_args.balanced_action,
        'query_prompt_type': data_args.query_prompt_type,
        'time_str': os.path.split(sft_args.output_dir)[-1],
        'sample_bins': data_args.sample_bins,
        'sample_ratios': data_args.sample_ratios,
        'sample_size': data_args.sample_size,
        'cache_file': data_args.cache_file
    }
    print(saved_args)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = sft_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {sft_args.local_rank}, device: {sft_args.device}, n_gpu: {sft_args.n_gpu}"
        + f" distributed training: {bool(sft_args.local_rank != -1)}, 16-bits training: {sft_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"NuScenes parameters {data_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Sft parameters {sft_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(sft_args.output_dir):
        last_checkpoint = get_last_checkpoint(sft_args.output_dir)
    if last_checkpoint is not None and sft_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################
    dataset_class = getattr(DATASETS, data_args.dataset_classname)

    cur_data_args = index_data_args(deepcopy(data_args), 0)
    train_dataset_list = [dataset_class(data_path=script_args.dataset_name, data_args=cur_data_args)]
    if len(data_args.append_datasets) > 0:
        for i_dataset, cur_append_dataset in enumerate(data_args.append_datasets.split(',')):
            cur_data_args = index_data_args(deepcopy(data_args), i_dataset+1)
            train_dataset_list.append(
                getattr(DATASETS, cur_append_dataset)(data_path=script_args.dataset_name, data_args=cur_data_args)
            )
    train_dataset = CombinedDataset(
        action_type=data_args.balanced_action,
        samples_per_type={
            'Go Straight': 8192,
            'Turn Left': 2048,
            'Turn Right': 2048,
            'Deviate Left': 2048,
            'Deviate Right': 2048,
            'U Turn': 512
        },
        dataset_list=train_dataset_list,
        ratio_shuffle=data_args.ratio_shuffle,
        mixed_shuffle=data_args.mixed_shuffle
    )

    ################
    # Load tokenizer
    ################
    global processor
    if "vl" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        logger.info("Using AutoProcessor for vision-language model.")
    else:
        processor = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
        )
        logger.info("Using AutoTokenizer for text-only model.")

    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if sft_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        resized_W=data_args.re_weight,
        resized_H=data_args.re_height,
        num_views=data_args.num_views,
    )
    # sft_args.model_init_kwargs = model_kwargs
    
    if "Qwen2-VL" in model_args.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    elif "Qwen2.5-VL" in model_args.model_name_or_path:
        # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     model_args.model_name_or_path, **model_kwargs
        # )
        model = CustomQwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    else:
        raise ValueError(f"Unsupported model: {model_args.model_name_or_path}")

# ###################
#     for name, param in model.named_parameters():
#         if 'prompt_tuning' not in name and 'lm_head' not in name:
#             param.requires_grad = False
#         else:
#             param.requires_grad = True

#     if dist.is_initialized():
#         dist.barrier()

#     if not dist.is_initialized() or dist.get_rank() == 0:
#         device = next(model.parameters()).device
#         print(f"Model device: {device}")
        
#         print("All parameter names and shapes:")
#         for name, param in model.named_parameters():
#             if param.requires_grad == True:
#                 print(f"  {name}: {param.shape} (requires_grad: {param.requires_grad})")
        
#         trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         total_params = sum(p.numel() for p in model.parameters())

#         print(f"Trainable parameters: {trainable_params:,}")
#         print(f"Total parameters: {total_params:,}")
#         if total_params > 0:
#             print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
#         else:
#             print("Warning: total_params is 0! Check model loading.")
            
#     # with open("model_structure.txt", "w", encoding="utf-8") as f:
#     #     f.write(str(model))
#     # with open("model_parameters.txt", "w", encoding="utf-8") as f:
#     #     for name, param in model.named_parameters():
#     #         f.write(f"{name} : {param.shape}\n")
#     # print("已将模型结构和参数名保存到 model_structure.txt 和 model_parameters.txt")
# ###################

    ############################
    # Initialize the SFT Trainer
    ############################

    sft_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    sft_args.remove_unused_columns = False
    
    trainer = CustomTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=processor.tokenizer,
        data_collator=partial(collate_fn, prompt_mask=data_args.use_prompt_mask, resized_W=data_args.re_weight, resized_H=data_args.re_height, img_W=data_args.img_weight, img_H=data_args.img_height),
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(train_dataset, data_args, sft_args, model_args),
        log_dir=sft_args.output_dir + "/tensorboard_logs",
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if sft_args.resume_from_checkpoint is not None:
        checkpoint = sft_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    print("*** Save model ***")
    trainer.save_model(sft_args.output_dir)
    print(f"Model saved to {sft_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        # "finetuned_from": model_args.model_name_or_path,
        # "dataset": list(script_args.dataset_name),
        # "dataset_tags": list(script_args.dataset_name),
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(sft_args.output_dir)
        processor.save_pretrained(sft_args.output_dir)

    ##########
    # Evaluate
    ##########
    if sft_args.do_eval:
        data_args.split = 'val'
        eval_dataset = dataset_class(data_path=script_args.dataset_name, data_args=data_args)
        # eval_dataset = Subset(eval_dataset, range(10))
        logger.info("*** Evaluate ***")
        trainer.tokenizer.padding_side = 'left'
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    Path('logs').mkdir(parents=True, exist_ok=True)

    time_str = os.path.split(sft_args.output_dir)[-1]
    save_json(f'logs/{time_str}.json', saved_args)
    with open(f'logs/success.txt', 'a') as f:
        f.write(time_str + '\n')

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, NuScenesDataConfig, SFTConfig, ModelConfig))
    script_args, data_args, sft_args, model_args = parser.parse_args_and_config()
    main(script_args, data_args, sft_args, model_args)
