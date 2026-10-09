#!/usr/bin/env python

from dataclasses import dataclass, field
from functools import partial
from typing import Optional
from tqdm import tqdm
import sys
import json
import os
import re
import itertools
import pickle
import torch
import torch.distributed as dist
import numpy as np
from qwen_vl_utils import process_vision_info
from torch.utils.data import Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModel, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from trl import (TrlParser, get_kbit_device_map, get_peft_config,
                 get_quantization_config)

from open_r1.eval.json_parser import JsonParserFilter
from open_r1.eval.angle_parser import AngleParserFilter
from open_r1.eval.nuscenes import estimate, calculate
import open_r1.dataloader as DATASETS
from open_r1.dataloader.utils import preprocess_meta_action_str
from open_r1.configs import NuScenesDataConfig
from inject_utils.Qwen2_5_CoVT_dinov3_sam3_da3_lidardepth_CA import CustomQwen2_5_VLForConditionalGeneration
from inject_utils.open_loop import merge_data_ADE_4s
from inject_utils.utils import fetch_lidardepth_navsim
from inject_utils.dino3_utils import preprocess_dino_images
from inject_utils.sam3_utils import fetch_sam_images
from inject_utils.da3_utils import extract_camera_parameters

def load_rank_outputs(output_dir, num_ranks):
    all_data = []

    for rank in range(num_ranks):
        path = os.path.join(output_dir, f"results_rank{rank}.pkl")
        if not os.path.exists(path):
            print(f"[Warning] File not found: {path}")
            continue

        outputs=[]
        with open (path, 'rb') as f:
            while True:
                try:
                    item = pickle.load(f)
                    outputs.append(item[1])
                except EOFError:
                    break
        all_data.extend(outputs)

    return clean_sort_dedup(all_data)

def clean_sort_dedup(data):
    dedup_dict = {}

    for item in data:
        item_id = item[0]["id"]
        # 提取数字部分：例如 'navsim_4s_test_00000042' -> 42
        match = re.search(r"(\d+)$", item_id)
        if match:
            num_id = int(match.group(1))
            # 保留最后一个或第一个（视逻辑修改）
            dedup_dict[num_id] = item
        else:
            print(f"[Warning] Could not extract ID number from: {item_id}")

    # 根据数字 ID 排序
    sorted_items = [dedup_dict[k] for k in sorted(dedup_dict.keys())]
    return sorted_items

def merge_and_save(all_data, output_dir, max_len=None):
    flat_data = [item for sublist in all_data for item in sublist]

    final_path = os.path.join(output_dir, "results.pkl")
    with open(final_path, "wb") as f:
        pickle.dump({"predictions": flat_data}, f)

    print(f"[✓] Merged {len(flat_data)} items to {final_path}")

    return final_path
    
@dataclass
class InferConfig:
    """Arguments for model inference"""

    adapter_path: str = field(
        default=None, metadata={"help": "The benchmarks to run after training."}
    )
    model_name_or_path: str = field(
        default="data/pretrained/Qwen/Qwen2.5-VL-3B-Instruct", metadata={"help": "The pretrained model path."}
    )
    dataset_name: str = field(
        default="data/datasets/nuScenes",
        metadata={"help": "The path of dataset."}
    )
    output_dir: str = field(default="output/eval", metadata={"help": "The save directory of output."})
    batch_size: int = field(default=1, metadata={"help": "The batch size on each gpu."})
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    attn_implementation: str = field(
        default="flash_attention_2", metadata={"help": "The attention type use in model."}
    )
    max_new_tokens: int = field(default=4096, metadata={"help": "The max length of generation."})
    generation_num_beams: int = field(default=1, metadata={"help": "The beam size in generation."})
    num_workers: int = field(default=8, metadata={"help": "number of dataloader workers"})

def token_img_resize(data, W=896, H=448, dinov3=False, sam3=False, depth=False):
    if dinov3:
        dino_token="<DINO_TOKEN>"
        dino_num_tokens=8
        dino_content = {
                    "type": "text",
                    "text": dino_token * dino_num_tokens
                }
    if sam3:
        sam_token = "<SAM_TOKEN>"
        sam_num_tokens = 8
        # sam_content = {
        #     "type": "text",
        #     "text": sam_token * sam_num_tokens
        # }
        sam_content = {
            "type": "text",
            "text": f"In this view, the semantic segmentation masks for Vehicles, Pedestrians, Traffic signs, Traffic Lights, and Road Markings is {sam_token * sam_num_tokens}"
        }
    if depth:
        depth_token="<DEPTH_TOKEN>"
        depth_num_tokens=32
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
                            if sam3:
                                new_content.append(sam_content)
                            if depth:
                                new_content.append(depth_content)

                        else:
                            new_content.append(content_item)
                    msg["content"] = new_content
    return data

def collate_fn(examples, prompt_mask, processor, resized_W, resized_H, use_dinov3=False, use_sam3=False, use_depth=False):
    examples = token_img_resize(examples, W=resized_W, H=resized_H, dinov3=use_dinov3, sam3=use_sam3, depth=use_depth)
    case_id = examples[0]['id']

    messages = [example['messages'][:-1] for example in examples]

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]

    image_inputs, video_inputs = process_vision_info(messages)
    # dino_input_list = []
    # sam_img_path = []
    # da3_img_list = []
    # cam_intrinsics = []
    # cam_extrinsics = []
    # depth_maps =[]
    # for example in examples:
    #     if use_dinov3:
    #         dino_input = preprocess_dino_images(example, resize=(resized_H, resized_W))
    #         dino_input_list.append(dino_input)
    #     if use_sam3:
    #         img_list = fetch_sam_images(example)
    #         sam_img_path.append(img_list)
    #     if use_depth:
    #         img_list = fetch_sam_images(example)
    #         da3_img_list.append(img_list)

    #         intrinsics, extrinsics = extract_camera_parameters(example['gold'], resized_H=resized_H, resized_W=resized_W, img_H=1080, img_W=1920)
    #         cam_intrinsics.append(intrinsics)
    #         cam_extrinsics.append(extrinsics)

    #         depth_map = fetch_lidardepth_navsim(example)
    #         depth_maps.append(depth_map)

    batch = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    ignore_token_ids = [
        processor.tokenizer.pad_token_id,  # 填充token  151643
        processor.tokenizer.convert_tokens_to_ids(processor.image_token),  # 图像token  151655
    ]

    # batch["img_list"] = dino_input_list
    # batch["sam_path"] = sam_img_path
    # batch["da3_input"] = da3_img_list
    # batch["intrinsic"] = cam_intrinsics
    # batch["extrinsic"] = cam_extrinsics
    # batch["depth_maps"] = depth_maps
    batch['id']=case_id
    return batch

def main(data_args, infer_args):
    os.makedirs(infer_args.output_dir, exist_ok=True)

    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank % torch.cuda.device_count())

    processor = AutoProcessor.from_pretrained(
        infer_args.model_name_or_path,
        trust_remote_code=True,
        padding_side='left',
    )

    model_kwargs = dict(
        resized_W=data_args.re_weight,
        resized_H=data_args.re_height,
        num_views=data_args.num_views,
        dino_token_id=None,
        sam_token_id=None,
        depth_token_id=None,
        use_dinov3=False,
        use_sam3=False,
        use_depth=False,
    )

    model = CustomQwen2_5_VLForConditionalGeneration.from_pretrained(
        infer_args.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(torch.device("cuda"))

    if infer_args.adapter_path:
        model = PeftModel.from_pretrained(model, infer_args.adapter_path).to(torch.device("cuda"))
    model.eval()

    dataset_class = getattr(DATASETS, data_args.dataset_classname)
    eval_dataset = dataset_class(data_path=infer_args.dataset_name, data_args=data_args)

    sampler = DistributedSampler(
        eval_dataset, num_replicas=world_size, rank=local_rank, shuffle=False
    )

    dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=infer_args.batch_size,
        sampler=sampler,
        collate_fn=partial(collate_fn, prompt_mask=True, processor=processor, resized_W=data_args.re_weight, resized_H=data_args.re_height, use_dinov3=data_args.use_dinov3, use_sam3=data_args.use_sam3, use_depth=data_args.use_depth),
        num_workers=infer_args.num_workers,
        pin_memory=True,
    )
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    rank_name = 'rank_results'
    rank_path = os.path.join(infer_args.output_dir, rank_name)
    os.makedirs(rank_path, exist_ok=True)
    rank_output_path = os.path.join(rank_path, f"results_rank{local_rank}.pkl")

    # ============ Inference Loop ============
    all_preds = []
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Inference", disable=(local_rank != 0))
        for idx, inputs in enumerate(iterator):
            inputs.to("cuda")
            sample_id = inputs.pop('id', None)
            global_index = idx * world_size + local_rank
            # Inference: Generation of the output
            past_key_values = None
            generated_ids = model.generate(**inputs, max_new_tokens=infer_args.max_new_tokens, use_cache=True, past_key_values=past_key_values) # 
            # generated_ids = model.generate(**inputs, max_new_tokens=infer_args.max_new_tokens) #
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )
            structured_outputs = []
            for text in output_texts:
                try:
                    traj = json.loads(text)
                except Exception:
                    traj = [] 
                ######## 去除 heading ########
                # traj = [tr[:-1] for tr in traj]
                #########################
                structured_outputs.append({
                    "id": sample_id,
                    "pre_traj": traj
                })
            # all_preds.append((global_index, structured_outputs))

            with open(rank_output_path, "ab") as f:
                pickle.dump((global_index, structured_outputs), f)

    dist.barrier()
    # rank0负责合并所有rank结果
    if dist.get_rank() == 0:
        all_data = load_rank_outputs(rank_path, world_size)
        result_pkl_path = merge_and_save(all_data, infer_args.output_dir, len(eval_dataset))
        filename = os.path.splitext(os.path.basename(result_pkl_path))[0] + '.json'
        merge_data_ADE_4s(result_pkl_path, filename)
    dist.barrier()


if __name__ == "__main__":
    parser = TrlParser((NuScenesDataConfig, InferConfig))
    data_args, infer_args = parser.parse_args_and_config()
    main(data_args, infer_args)
