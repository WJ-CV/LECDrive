import os
import torch
import sys
import json
import requests
import numpy as np
from PIL import Image
import sam3
from tqdm import tqdm
import matplotlib.cm as cm
from typing import List
from io import BytesIO
from multiprocessing import Pool, Manager
from sam3.visualization_utils import plot_results
from sam3 import build_sam3_image_model
from sam3.train.transforms.basic_for_api import (
    ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
)
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3.eval.postprocessors import PostProcessImage
from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
torch.inference_mode().__enter__()


def overlay_and_save_instance_single_masks_on_black(
    image, 
    results, 
    img_id: int, 
    save_dir: str = "./vis",  # 保存目录
    img_name: str = "",
    visualization: bool = False,
):
    width, height = image.size
    semantic_label = np.zeros((height, width), dtype=np.uint8)

    for semantic_id in range(len(results)):
        masks = results[semantic_id]["masks"]  # [N, 1, H, W]
        pseudo_value = semantic_id + 1 # 255 - semantic_id * 50
    
        if masks.numel() == 0:
            continue

        masks_np = masks.cpu().numpy().astype(np.bool_)
        n_masks = masks_np.shape[0] 

        for i in range(n_masks):
            mask = masks_np[i]  # [H, W]
            if mask.ndim == 3:
                mask = mask[0]  # 取 [H, W] 的掩码

            semantic_label[mask] = pseudo_value 

    if visualization:
        name = os.path.basename(img_name)
        img_dir = os.path.dirname(img_name)
        save_dir = save_dir + img_dir
        os.makedirs(save_dir, exist_ok=True)

        vis_label = semantic_label * 42  # 0→0, 1→51, ...,5→255
        vis_label_pil = Image.fromarray(vis_label).convert("RGB")  # 转为RGB，和原图格式一致
        # combined_width = width * 2
        # combined_height = height
        # combined_img = Image.new("RGB", (combined_width, combined_height))
        # combined_img.paste(image, (0, 0))
        # combined_img.paste(vis_label_pil, (width, 0))

        save_path = os.path.join(save_dir, name)
        vis_label_pil.save(save_path)
        # print(f"Combined image saved to {save_path}")

    return semantic_label


def build_datapoint_from_image(
    image_path: str,
    prompts: List[str],
    transform=None,
):
    img = Image.open(image_path).convert("RGB")

    datapoint = create_empty_datapoint()
    set_image(datapoint, img)

    query_ids = []
    for text in prompts:
        qid = add_text_prompt(datapoint, text)
        query_ids.append(qid)

    if transform is not None:
        datapoint = transform(datapoint)

    return datapoint, img, query_ids

GLOBAL_COUNTER = 1
def create_empty_datapoint():
    """ A datapoint is a single image on which we can apply several queries at once. """
    return Datapoint(find_queries=[], images=[])

def set_image(datapoint, pil_image):
    """ Add the image to be processed to the datapoint """
    w,h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h,w])]

def add_text_prompt(datapoint, text_query):
    """ Add a text query to the datapoint """

    global GLOBAL_COUNTER
    # in this function, we require that the image is already set.
    # that's because we'll get its size to figure out what dimension to resize masks and boxes
    # In practice you're free to set any size you want, just edit the rest of the function
    assert len(datapoint.images) == 1, "please set the image first"

    w, h = datapoint.images[0].size
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[], # unused for inference
            is_exhaustive=True, # unused for inference
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=GLOBAL_COUNTER,
                original_image_id=GLOBAL_COUNTER,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            )
        )
    )
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER - 1

DEFAULT_PROMPTS = [
    "Vehicles",
    "Pedestrians",
    "Traffic signs", 
    "Traffic Lights",
    "Road Markings",
    "Road edge",
]

model = build_sam3_image_model(checkpoint_path="pretrain/sam3/sam3.pt")

transform = ComposeAPI(
    transforms=[
        RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
postprocessor = PostProcessImage(
    max_dets_per_img=-1,       # if this number is positive, the processor will return topk. For this demo we instead limit by confidence, see below
    iou_type="segm",           # we want masks
    use_original_sizes_box=True,   # our boxes should be resized to the image size
    use_original_sizes_mask=True,   # our masks should be resized to the image size
    convert_mask_to_rle=False, # the postprocessor supports efficient conversion to RLE format. In this demo we prefer the binary format for easy plotting
    detection_threshold=0.5,   # Only return confident detections
    to_cpu=False,
)


json_path = '/e2e-data/evad-tech-vla/wangjie68/experiments-VLA/2026-02-09_0349_repo_COVT_dinov3_sam3_da3_lidardepth-motiontoken_nus/cache_files/nuScenes_cache/nuscenes_val_planning_view_adjust.json'
with open(json_path, "r") as js:
    js_data = json.load(js)

for data in tqdm(js_data, desc="Processing js_data"):
    image_paths = []  
    contents = data["messages"][0]["content"]
    for content in contents:
        if content["type"] == "image":
            img_path= content["image"].replace("file://", "")
            image_paths.append(img_path)

    img_path_list = [image_paths[:3], image_paths[3:]]
    for img_paths in img_path_list:
        datapoints = []
        images = []
        all_query_ids = []

        for path in img_paths:
            dp, img, qids = build_datapoint_from_image(
                path,
                prompts=DEFAULT_PROMPTS,
                transform=transform,
            )
            datapoints.append(dp)
            images.append(img)
            all_query_ids.append(qids)

        batch = collate(datapoints, dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
        output = model(batch)
        processed_results = postprocessor.process_results(output, batch.find_metadatas)

        for img_idx, qids in enumerate(all_query_ids):
            img_name = img_paths[img_idx].split("samples/")[1]
            overlay_and_save_instance_single_masks_on_black(
                images[img_idx], 
                [processed_results[qid] for qid in qids],
                img_id=img_idx,
                save_dir="/e2e-data/evad-tech-vla/wangjie68/experiments-VLA/1_CoVT_train_files/sam3/sam3_semantic_mask_nus/",
                img_name=img_name,
                visualization=True,
            )
