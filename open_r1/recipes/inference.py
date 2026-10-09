#!/usr/bin/env python

from dataclasses import dataclass, field
from functools import partial
from typing import Optional
from tqdm import tqdm
import sys
import json
import os
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
from depth_utils.Qwen2_5_3D_pre import CustomQwen2_5_VLForConditionalGeneration
from depth_utils.utils import fetch_info, fetch_depth, lidar2img, merge_data_ADE_3s, fetch_map

def insert_camera_tokens(messages, camera_token="<CAMERA_TOKEN>", num_tokens=5):
    for message in messages:
        new_content = []
        for msg in message["messages"][0]["content"]:
            new_content.append(msg)
            if msg["type"] == "image":
                camera_text = camera_token * num_tokens
                new_content.append({
                    "type": "text",
                    "text": camera_text
                })
        message["messages"][0]["content"] = new_content
    return messages

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


def collate_fn(examples, prompt_mask, processor):
    examples = insert_camera_tokens(examples, num_tokens=1)
    # remove assistant answer from the examples
    case_id = examples[0]['id']
    messages = [
        example['messages'][:-1]
        for example in examples
        if 'messages' in example and isinstance(example['messages'], list)
    ]
    for example in examples:
        if 'messages' not in example:
            print("Missing 'messages' in example:", example)
    # messages = [example['messages'][:-1] for example in examples]
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)

    traj_maps = []
    depth_maps =[]
    cam2lidar_inputs = []
    for example in examples:
        depth_map, traj_map = fetch_map(example)
        traj_maps.append(traj_map)
        depth_maps.append(depth_map)

        cam2lidar = lidar2img(example['gold'])
        cam2lidar_inputs.append(cam2lidar)

    batch = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    batch["traj_maps"] = traj_maps
    batch["depth_maps"] = depth_maps
    batch["cam2lidar"] = cam2lidar_inputs
    batch['id']=case_id
    return batch

def main(data_args, infer_args):
    # setup ddp
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank % torch.cuda.device_count())

    # load processor / model
    processor = AutoProcessor.from_pretrained(
        infer_args.model_name_or_path,
        trust_remote_code=True,
        # https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/discussions/19
        padding_side='left',
    )

    CAMERA_TOKEN = "<CAMERA_TOKEN>"
    if CAMERA_TOKEN not in processor.tokenizer.additional_special_tokens:
        # 添加为额外特殊token
        processor.tokenizer.add_special_tokens({
            "additional_special_tokens": [CAMERA_TOKEN]
        })
        logger.info(f"Added special token: {CAMERA_TOKEN}")

    # model = AutoModel.from_pretrained(    Qwen2_5_VLForConditionalGeneration
    model = CustomQwen2_5_VLForConditionalGeneration.from_pretrained(
        infer_args.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(torch.device("cuda"))

    if infer_args.adapter_path:
        model = PeftModel.from_pretrained(model, infer_args.adapter_path).to(torch.device("cuda"))
    model.eval()

    # 调整模型嵌入层大小以匹配新词汇表
    original_vocab_size = len(processor.tokenizer)
    if CAMERA_TOKEN not in processor.tokenizer.added_tokens_encoder:
        model.resize_token_embeddings(original_vocab_size + 1)
        logger.info(f"Resized model embeddings to {original_vocab_size + 1} tokens")

    dataset_class = getattr(DATASETS, data_args.dataset_classname)
    eval_dataset = dataset_class(data_path=infer_args.dataset_name, data_args=data_args)
    # eval_dataset = Subset(eval_dataset, range(4))

    sampler = DistributedSampler(
        eval_dataset, num_replicas=world_size, rank=local_rank, shuffle=False
    )

    dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=infer_args.batch_size,
        sampler=sampler,
        collate_fn=partial(collate_fn, prompt_mask=True, processor=processor),
        num_workers=infer_args.num_workers,
        pin_memory=True,
    )
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # inference
    all_preds = []
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Inference", disable=(local_rank != 0))
        for idx, inputs in enumerate(iterator):
            inputs.to("cuda")
            sample_id = inputs.pop('id', None)
            global_index = idx * world_size + local_rank

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=infer_args.max_new_tokens)
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
                    traj = []  # fallback: 空列表
                structured_outputs.append({
                    "id": sample_id,
                    "pre_traj": traj
                })
            all_preds.append((global_index, structured_outputs))# global_index

    dist.barrier(device_ids=[local_rank])  # distributed barrier

    # gather results
    merged_outputs = [None for _ in range(world_size)]
    dist.all_gather_object(merged_outputs, json.dumps(all_preds))

    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

    # dump to disk
    if dist.get_rank() == 0:
        os.makedirs(infer_args.output_dir, exist_ok=True)

        # sort by indices & truncated to the length of eval dataset
        merged_outputs = sorted(merged_outputs, key=lambda x: x[0])
        merged_outputs = [item for _, sublist in merged_outputs for item in sublist][:len(eval_dataset)]
        pred_output_path = os.path.join(infer_args.output_dir, f"results.pkl")
        with open(pred_output_path, "wb") as f:
            pickle.dump({"predictions": merged_outputs}, f)

        print(f"Results saved to {pred_output_path}")

        # parse predictions
        parsed_outputs = []
        if "json" in data_args.query_prompt_type:
            filter = JsonParserFilter()
            for response in merged_outputs:
                ret = filter.apply(response)
                parsed_outputs.append(ret)
        elif "angle" in data_args.query_prompt_type:
            filter = AngleParserFilter()
            for response in merged_outputs:
                ret = filter.apply(response)
                parsed_outputs.append(ret)
        else:
            raise NotImplementedError(f"Unsupported {data_args.query_prompt_type}.")

        # dump parsed results
        parsed_output_path = os.path.join(infer_args.output_dir, f"results_parsed.pkl")
        with open(parsed_output_path, "wb") as f:
            pickle.dump(parsed_outputs, f)

        # # calculate metrics
        # if len(parsed_outputs) > 0:
        #     scores, golds = [], []
        #     for idx, pred in enumerate(parsed_outputs):
        #         gt = eval_dataset.dataset.get_info(idx) if \
        #                 isinstance(eval_dataset, Subset) else eval_dataset.get_info(idx)
        #         example = {
        #             'gold': {
        #                 "trajectory": gt["fut_trajectory"],
        #                 # "lateral_control": gt["lat_action"],
        #                 # "longitudinal_control": gt["lon_action"],
        #                 "lateral_control": preprocess_meta_action_str(gt["lat_action"]),
        #                 "longitudinal_control": preprocess_meta_action_str(gt["lon_action"]),
        #             },
        #         }
        #         scores.append(calculate(data=example, **pred))
        #         golds.append(example['gold'])
        #     metrics = estimate(scores=scores, golds=golds)
        #     print(f"metrics:\n{json.dumps(metrics, indent=2)}")

    # cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = TrlParser((NuScenesDataConfig, InferConfig))
    data_args, infer_args = parser.parse_args_and_config()
    main(data_args, infer_args)
