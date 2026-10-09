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
from typing import Tuple
# from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from inject_utils.Qwen2_5_CoVT_dino_depth_CA_projection import CustomQwen2_5_VLForConditionalGeneration
from inject_utils.utils import fetch_img_list, fetch_img_list_navsim, fetch_img_list_depth_navsim, fetch_depth_navsim
from inject_utils.dino3_utils import preprocess_dino_images

class CustomTrainer(SFTTrainer):
    def __init__(self, *args, log_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = SummaryWriter(log_dir=log_dir)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        depth_map_loss = outputs.depth_map_loss.item() if outputs.depth_map_loss is not None else 0.0
        dino_loss_l2 = outputs.dino_loss_l2.item() if outputs.dino_loss_l2 is not None else 0.0

        global_step = self.state.global_step if hasattr(self, "state") else 0
        self.writer.add_scalar("loss/train_loss", (loss - depth_map_loss - dino_loss_l2).item(), global_step)
        self.writer.add_scalar("loss/depth_map_loss", depth_map_loss, global_step)
        self.writer.add_scalar("loss/dino_loss_l2", dino_loss_l2, global_step)

        self.log({
            "loss": loss.item() - depth_map_loss - dino_loss_l2,
            "depth_map_loss": depth_map_loss,
            "dino_loss_l2": dino_loss_l2,
        })

        return (loss, outputs) if return_outputs else loss

    def teardown(self):
        super().teardown()
        self.writer.close()

def token_img_resize(data, W=896, H=448, dinov3=False, depth=False):
    if dinov3:
        dino_token="<DINO_TOKEN>"
        dino_num_tokens=4
        dino_content = {
                    "type": "text",
                    "text": dino_token * dino_num_tokens
                }
    if depth:
        depth_token="<DEPTH_TOKEN>"
        depth_num_tokens=8
        depth_content = {
                    "type": "text",
                    "text": depth_token * depth_num_tokens
                }
    for item in data:
        messages = item.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", [])
                if isinstance(content, list):
                    new_content = []
                    for content_item in content:
                        if content_item.get("type") == "image":
                            content_item["resized_width"] = W
                            content_item["resized_height"] = H
                            new_content.append(content_item)
                            if dinov3:
                                new_content.append(dino_content)
                            if depth:
                                new_content.append(depth_content)

                        else:
                            new_content.append(content_item)
                    msg["content"] = new_content
    return data

processor = None

def collate_fn(examples, prompt_mask, resized_W, resized_H, use_dinov3=False, use_depth=False):
    examples = token_img_resize(examples, W=resized_W, H=resized_H, dinov3=use_dinov3, depth=use_depth)
    texts = [
        processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
        for example in examples
    ]
    image_inputs = []
    dino_input_list = []
    depth_maps =[]
    for example in examples:
        imgs, vids = process_vision_info(example["messages"])
        image_inputs.append(imgs)

        if use_dinov3:
            dino_input = preprocess_dino_images(example, resize=(resized_H, resized_W))
            dino_input_list.append(dino_input)
        if use_depth:
            depth_map = fetch_depth_navsim(example)
            depth_maps.append(depth_map)

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
    if use_dinov3:
        DINO_TOKEN = "<DINO_TOKEN>"
        dino_token_id = processor.tokenizer.convert_tokens_to_ids(DINO_TOKEN) # 151665
        ignore_token_ids.append(processor.tokenizer.convert_tokens_to_ids(DINO_TOKEN))
    if use_depth:
        DEPTH_TOKEN = "<DEPTH_TOKEN>"
        depth_token_id = processor.tokenizer.convert_tokens_to_ids(DEPTH_TOKEN) # 151666
        ignore_token_ids.append(processor.tokenizer.convert_tokens_to_ids(DEPTH_TOKEN))

    labels = batch["input_ids"].clone()
    for token_id in ignore_token_ids:
        labels[labels == token_id] = -100

    batch["labels"] = labels
    batch["img_list"] = dino_input_list
    batch["depth_maps"] = depth_maps
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

    # # 在label中不带Prompt
    # question_texts = [
    #     processor.apply_chat_template([example["messages"][-1]], tokenize=False, add_generation_prompt=True)
    #     for example in examples
    # ]

    # for idx, i_question_text in enumerate(question_texts):
    #     cur_question_batch = processor(
    #         text=[i_question_text],
    #         images=image_inputs,
    #         return_tensors="pt",
    #         padding=False,
    #     )
    #     label_tokens_num = cur_question_batch['input_ids'].shape[1]
    #     batch["labels"][idx, :-label_tokens_num] = -100

    return batch

class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, action_type, dataset_list, mixed_shuffle):
        self.datasets = dataset_list
        self.action_type = action_type
        self.mixed_shuffle = mixed_shuffle
        self._samples = []
        for dataset in self.datasets:
            self._samples.extend(dataset._samples)

        if torch.cuda.current_device() == 0:
            print(f"Total samples: {len(self._samples)}")
            print("="*40 + "\n")

        self.shuffled_samples = self._samples
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
    train_dataset = CombinedDataset(
        action_type=data_args.balanced_action,
        dataset_list=train_dataset_list,
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

    if data_args.use_dinov3:
        DINO_TOKEN = "<DINO_TOKEN>"
        if DINO_TOKEN not in processor.tokenizer.additional_special_tokens:
            processor.tokenizer.add_special_tokens({
                "additional_special_tokens": [DINO_TOKEN]
            })
            print(f"Added special token: {DINO_TOKEN}")
        dino_token_id = processor.tokenizer.convert_tokens_to_ids(DINO_TOKEN) # 151665
    else:
        dino_token_id = None

    if data_args.use_depth:
        DEPTH_TOKEN = "<DEPTH_TOKEN>"
        if DEPTH_TOKEN not in processor.tokenizer.additional_special_tokens:
            processor.tokenizer.add_special_tokens({
                "additional_special_tokens": [DEPTH_TOKEN]
            })
            print(f"Added special token: {DEPTH_TOKEN}")
        depth_token_id = processor.tokenizer.convert_tokens_to_ids(DEPTH_TOKEN) # 151666
    else:
        depth_token_id = None
    
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
        use_dinov3=data_args.use_dinov3,
        use_depth=data_args.use_depth,
        dino_token_id=dino_token_id,
        depth_token_id=depth_token_id,
        dino_pretrain_path=data_args.dino_pretrain_path,
    )
    
    if "Qwen2-VL" in model_args.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    elif "Qwen2.5-VL" in model_args.model_name_or_path:
        model = CustomQwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    else:
        raise ValueError(f"Unsupported model: {model_args.model_name_or_path}")

    # 调整模型嵌入层大小以匹配新词汇表
    if data_args.use_dinov3 and DINO_TOKEN not in processor.tokenizer.added_tokens_encoder:
        model.resize_token_embeddings(original_vocab_size + 1)
        logger.info(f"Resized model embeddings to {original_vocab_size + 1} tokens")
    if data_args.use_depth and DEPTH_TOKEN not in processor.tokenizer.added_tokens_encoder:
        model.resize_token_embeddings(original_vocab_size + 1)
        logger.info(f"Resized model embeddings to {original_vocab_size + 1} tokens")

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
        data_collator=partial(collate_fn, prompt_mask=data_args.use_prompt_mask, resized_W=data_args.re_weight, resized_H=data_args.re_height, use_dinov3=data_args.use_dinov3, use_depth=data_args.use_depth),
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

    # # Save everything else on main process
    # kwargs = {
    #     # "finetuned_from": model_args.model_name_or_path,
    #     # "dataset": list(script_args.dataset_name),
    #     # "dataset_tags": list(script_args.dataset_name),
    #     "tags": ["open-r1"],
    # }
    if trainer.accelerator.is_main_process:
        # trainer.create_model_card(**kwargs)
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
