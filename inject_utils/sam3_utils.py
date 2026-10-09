import os
import torch
import sys
import cv2
import requests
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
import sam3
import matplotlib.cm as cm
from typing import List
from io import BytesIO
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

def fetch_sam_images(example):
    img_path_list = []
    for exp in example["messages"]:
        if exp.get("role") == "user":
            content_list = exp.get("content", [])

    for content in content_list:
        if 'image' not in content:
            continue
        image_path = content['image']
        resized_height, resized_width = content["resized_height"], content["resized_width"]
        img_path = image_path.replace("file://", "")
        img_path_list.append(img_path)

    return img_path_list

def fetch_sam3_mask_navsim(example):
    sam_mask_list = []
    for exp in example["messages"]:
        if exp.get("role") == "user":
            content_list = exp.get("content", [])
    for content in content_list:
        if 'image' not in content:
            continue
        image_path = content['image']
        resized_height, resized_width = content["resized_height"], content["resized_width"]
        path = image_path.split('sensor_blobs/')[1]

        pwd = os.getcwd()
        sam_mask_path = pwd + '/sam3/sam3_semantic_mask/' + path
        pil_mask = Image.open(sam_mask_path).convert('L')
        mask_array = np.asarray(pil_mask).astype(np.float32)
        mask_array /= 42
        mask_array = np.round(mask_array).astype(np.uint8)
        mask_array = np.clip(mask_array, 0, 6)
        mask = cv2.resize(mask_array, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
        sam_mask_list.append(mask)

    return sam_mask_list

# def overlay_and_save_instance_masks(
#     image,
#     results,
#     img_id: int,
#     save_dir: str = "./vis",
#     alpha_scale: float = 0.5,
# ):
#     """
#     将 sam3 的 instance masks 覆盖到原图上并保存为 img_id.png

#     Args:
#         image: PIL.Image (RGB)
#         results: sam3 postprocessor 输出的单个 query 结果
#         img_id: 保存文件名使用的 id
#         save_dir: 保存目录
#         alpha_scale: mask 透明度系数 (0~1)
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     image = image.convert("RGBA")

#     masks = results["masks"]  # [N, 1, H, W]
#     if masks.numel() == 0:
#         # 没有实例，直接保存原图
#         save_path = os.path.join(save_dir, f"{img_id}.png")
#         image.convert("RGB").save(save_path)
#         return save_path

#     masks = 255 * masks.cpu().numpy().astype(np.uint8)
#     n_masks = masks.shape[0]

#     cmap = cm.get_cmap("rainbow", n_masks)
#     colors = [
#         tuple(int(c * 255) for c in cmap(i)[:3])
#         for i in range(n_masks)
#     ]

#     for mask, color in zip(masks, colors):
#         if mask.ndim == 3:
#             mask = mask[0]  # [H, W]

#         mask_img = Image.fromarray(mask, mode="L")

#         overlay = Image.new("RGBA", image.size, color + (0,))
#         alpha = mask_img.point(lambda v: int(v * alpha_scale))
#         overlay.putalpha(alpha)

#         image = Image.alpha_composite(image, overlay)

#     save_path = os.path.join(save_dir, f"{img_id}.png")
#     image.convert("RGB").save(save_path)

#     return save_path

# def overlay_and_save_instance_masks_on_black(
#     image,  # 输入图像（用于获取尺寸）
#     results,  # sam3 postprocessor 输出的单个 query 结果
#     img_id: int,  # 用于保存文件名的 img_id
#     save_dir: str = "./vis",  # 保存目录
# ):
#     os.makedirs(save_dir, exist_ok=True)

#     width, height = image.size

#     masks = results["masks"]  # [N, 1, H, W]
#     if masks.numel() == 0:
#         # save_path = os.path.join(save_dir, f"{img_id}.png")
#         black_image = Image.new("L", (width, height), 0)  # 创建黑色背景
#         # black_image.save(save_path)
#         return [black_image, 0]
#     else:
#         masks = 255 * masks.cpu().numpy().astype(np.uint8)  # 转换为 uint8，取值范围 [0, 255]
#         n_masks = masks.shape[0] 

#         black_image = Image.new("L", (width, height), 0)

#         for i in range(n_masks):
#             mask = masks[i]  # [H, W]
            
#             if mask.ndim == 3:
#                 mask = mask[0]  # 取 [H, W]

#             mask_img = Image.fromarray(mask, mode="L")

#             black_image.paste(255, (0, 0, width, height), mask_img)  # 把 mask 图像区域变为白色

#         # save_path = os.path.join(save_dir, f"{img_id}.png")
#         # black_image.save(save_path)  # 保存图像

#         return [black_image, 1]

# def overlay_and_save_instance_single_masks_on_black(
#     image, 
#     results, 
#     img_id: int, 
#     save_dir: str = "./vis",  # 保存目录
# ):
#     os.makedirs(save_dir, exist_ok=True)

#     width, height = image.size

#     segmentation_image = np.zeros((height, width), dtype=np.uint8)

#     for semantic_id in range(len(results)):
#         masks = results[semantic_id]["masks"]  # [N, 1, H, W]
#         pseudo_value = 255 - semantic_id * 50 
    
#         if masks.numel() == 0:
#             continue

#         masks = 255 * masks.cpu().numpy().astype(np.uint8)  # 转换为 uint8，取值范围 [0, 255]
#         n_masks = masks.shape[0]  # 实例数量

#         for i in range(n_masks):
#             mask = masks[i]  # [H, W]
            
#             if mask.ndim == 3:
#                 mask = mask[0]  # 取 [H, W]

#             segmentation_image[mask == 255] = pseudo_value

#     # segmentation_image_pil = Image.fromarray(segmentation_image)
#     # save_path = os.path.join(save_dir, f"{img_id}.png")
#     # segmentation_image_pil.save(save_path)  # 保存图像

#     return segmentation_image


def generate_semantic_labels_from_masks(
    image, 
    results, 
    img_id: int, 
    save_dir: str = "./semantic_pseudo_labels",  # 标签保存目录
    num_classes: int = 5,
    visualization: bool = False,
):
    width, height = image.size
    semantic_label = np.zeros((height, width), dtype=np.uint8)

    for semantic_id in range(len(results)):
        masks = results[semantic_id]["masks"]  # [N, 1, H, W]
        label_value  = semantic_id + 1

        if masks.numel() == 0:
            continue

        masks_np = masks.cpu().numpy().astype(np.bool_)
        n_masks = masks_np.shape[0] 

        for i in range(n_masks):
            mask = masks_np[i]  # [H, W]
            if mask.ndim == 3:
                mask = mask[0]  # 取 [H, W] 的掩码

            semantic_label[mask] = label_value 


    if visualization:
        os.makedirs(save_dir, exist_ok=True)

        vis_label = semantic_label * 51  # 0→0, 1→51, ...,5→255
        vis_label_pil = Image.fromarray(vis_label).convert("RGB")  # 转为RGB，和原图格式一致
        combined_width = width * 2
        combined_height = height
        combined_img = Image.new("RGB", (combined_width, combined_height))
        combined_img.paste(image, (0, 0))
        combined_img.paste(vis_label_pil, (width, 0))

        save_path = os.path.join(save_dir, f"combined_{img_id}.png")
        combined_img.save(save_path)
        print(f"Combined image saved to {save_path}")

    return semantic_label

def build_datapoint_from_image(
    image_path: str,
    prompts: List[str],
    transform=None,
    target_size=(1120, 560),
):
    img = Image.open(image_path).convert("RGB")
    # img_resized = img.resize(target_size, Image.BILINEAR)

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

transform = ComposeAPI(
    transforms=[
        RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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


def init_sam_model(pretrain, device):
    sam_model = build_sam3_image_model(
        checkpoint_path="sam3/pretrain/sam3/sam3.pt",
        device=device,
        )
    for param in sam_model.parameters():
        param.requires_grad = False
    sam_model.eval()

    if hasattr(sam_model, 'module'):
        for param in sam_model.module.parameters():
            if hasattr(param, 'ds_id'):
                param.ds_status = ZeroParamStatus.NOT_AVAILABLE
    else:
        for param in sam_model.parameters():
            if hasattr(param, 'ds_id'):
                param.ds_status = ZeroParamStatus.NOT_AVAILABLE
    sam_model._no_deepspeed = True
    return sam_model


def run_sam_inference(model, image_paths, pretrain_path=None, device=None, target_size=None, visualization=False):
    # model = init_sam_model(pretrain_path, device)
    model = model.to(device)
    # print(model.dot_prod_scoring.prompt_proj.weight[0, :20])
    total_img_list = sum(image_paths, [])
    datapoints = []
    images = []
    all_query_ids = []

    for path in total_img_list:
        dp, img, qids = build_datapoint_from_image(
            path,
            prompts=DEFAULT_PROMPTS,
            transform=transform,
            target_size=target_size,
        )
        datapoints.append(dp)
        images.append(img)
        all_query_ids.append(qids)

    batch = collate(datapoints, dict_key="dummy")["dummy"]
    batch = copy_data_to_device(batch, device, non_blocking=True)
    # model.eval()
    # with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
    output = model(batch)
    processed_results = postprocessor.process_results(output, batch.find_metadatas)
    # plot_results(images[0], processed_results[all_query_ids[0][0]])
    sam_semantic_labels = []
    for img_idx, qids in enumerate(all_query_ids):
        sam_pseudo_label = generate_semantic_labels_from_masks(
            images[img_idx], 
            [processed_results[qid] for qid in qids],
            img_id=img_idx,
            save_dir="./sam_semantic_train_labels",
            num_classes=5,
            visualization=visualization,
        )
        sam_pseudo_label = torch.from_numpy(sam_pseudo_label).long()
        sam_semantic_labels.append(sam_pseudo_label)
    
    sam_pseudo_mask = torch.stack(sam_semantic_labels, dim=0)
    return sam_pseudo_mask
            
class UpsampleWithPixelUnshuffle(nn.Module):
    def __init__(self, in_channels=3584, factor_1=8, factor_2=7, factor_3=5, out_channels=32):
        super().__init__()
        self.factor_1 = factor_1
        self.factor_2 = factor_2
        self.factor_3 = factor_3
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear_1 = nn.Linear(self.in_channels // (self.factor_1 ** 2), self.out_channels * (self.factor_2 ** 2))
        self.linear_2 = nn.Linear(self.out_channels, self.out_channels * (self.factor_3 ** 2))
        # self.linear_out = nn.Linear(self.out_channels, 6)

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
        # x_trans_5 = x_5.flatten(2, 3).permute(0, 2, 1)

        # x_out = self.linear_out(x_trans_5).permute(0, 2, 1)
        # x_out = x_out.view(B, -1, x_5.shape[2], x_5.shape[3])

        return x_5

class GenerateSamMask(nn.Module):
    def __init__(
        self,
        n_imgs=3,
        sem_classes=6,
        patch_w=4,
        patch_h=2,
        embed_dims=3584,
        factor_1=4,
        factor_2=4,
        factor_3=4,
        out_dims=96,
        w_sam=0.5
    ):
        super().__init__()
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.n_imgs = n_imgs
        self.sem_classes = sem_classes
        self.embed_dims = embed_dims
        self.out_dims = out_dims
        self.w_sam = w_sam

        self.feat_unsample_sam = UpsampleWithPixelUnshuffle(in_channels=self.embed_dims, factor_1=factor_1, factor_2=factor_2, factor_3=factor_3, out_channels=self.out_dims)
        # self.foreground_pre = nn.Conv2d(self.out_dims, 1, 1, 1, 0)
        self.semantic_pre = nn.Sequential(
            nn.Conv2d(self.out_dims, self.out_dims, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_dims, self.sem_classes, kernel_size=1, stride=1, padding=0)
        )

        class_weights = torch.tensor([0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        # self.criterion_bce = nn.BCEWithLogitsLoss()

    def forward(self, sam_hidden_states, sam_pseudo_label, check_pre_mask=False, out_dir='./'): # [B, 24, 3584]torch.bfloat16    [B, 3, 560, 1120]torch.int64
        B, n_token, dim = sam_hidden_states.shape

        sam_embed = sam_hidden_states.view(B, self.n_imgs, self.patch_h, self.patch_w, -1)
        sam_embed = sam_embed.permute(0, 1, 4, 2, 3)

        sam_embed_ = sam_embed.clone().view(B * self.n_imgs, dim, self.patch_h, self.patch_w)

        gen_semantic_mask = self.feat_unsample_sam(sam_embed_) # [B*3, 64, 128, 256]  torch.bfloat16

        sam_pseudo_label = sam_pseudo_label.flatten(0,1)
        if sam_pseudo_label.shape[2:] != gen_semantic_mask.shape[1:]:
            gen_semantic_mask = F.interpolate(gen_semantic_mask, size=sam_pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        # foreground_binary_label = (sam_pseudo_label > 0).float()
        # foreground_binary_label = foreground_binary_label.unsqueeze(1).to(gen_semantic_mask.dtype).to(gen_semantic_mask.device)
        # gen_foreground_mask = self.foreground_pre(gen_semantic_mask)
        # foreground_bce_loss = 0.01 * self.criterion_bce(gen_foreground_mask, foreground_binary_label)

        gen_semantic_mask = self.semantic_pre(gen_semantic_mask)
        semantic_ce_loss = self.criterion(gen_semantic_mask, sam_pseudo_label.to(torch.int64))

        if check_pre_mask:
            save_path = os.path.join(out_dir, 'sam3_pre_check')
            os.makedirs(save_path, exist_ok=True)
            gen_semantic_mask_prob = F.softmax(gen_semantic_mask, dim=1)
            gen_mask = torch.argmax(gen_semantic_mask_prob, dim=1)
            # gen_foreground = torch.sigmoid(gen_foreground_mask.squeeze(1))
            for i in range(gen_mask.shape[0]):
                # foreground_binary = (gen_foreground[i] > 0.7).float()
                # foreground_np = foreground_binary.cpu().numpy().astype(np.uint8) * 255
                # froe_mask = Image.fromarray(foreground_np)
                # froe_mask.save(f"sam3_pre_check/pre_foreground_mask_{i}.png")

                image_to_save = gen_mask[i].cpu().numpy()
                image = Image.fromarray(image_to_save.astype(np.uint8) * 42)
                image.save(os.path.join(save_path, f"pre_semantic_mask_{i}.png"))

                pseudo_label = sam_pseudo_label[i].to(torch.float32).cpu().numpy()
                label = Image.fromarray(pseudo_label.astype(np.uint8) * 42)
                label.save(os.path.join(save_path, f"label_semantic_mask_{i}.png"))

        return self.w_sam * semantic_ce_loss  # 