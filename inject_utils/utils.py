import cv2
from PIL import Image
import numpy as np
import os
import pickle
import json
import ast
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from vggt.utils.load_fn import load_and_preprocess_images
import torch.nn.functional as F

def fetch_img_list(example):
    process_img_list = []
    depth_map_list = []
    content_list = example["messages"][0]["content"]
    for content in content_list:
        if 'image' not in content:
            continue
        image_path = content['image']
        resized_height, resized_width = content["resized_height"], content["resized_width"]
        img_path = image_path.replace("file://", "")
        process_img_list.append(img_path)

        pwd = os.getcwd()
        path = image_path.split('samples/')[1]

        depth_map_path = pwd + '/nuscenes_dataset/samples_sparse_depth/' + path
        depth_map_path = depth_map_path.replace('.jpg', '.png') 
        pil_prior = Image.open(depth_map_path)
        np_prior = np.asarray(pil_prior).astype(np.float32)
        np_prior /= 500
        prior = cv2.resize(np_prior, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
        depth_map_list.append(prior)

    process_imgs = load_and_preprocess_images(process_img_list)

    return process_imgs, depth_map_list # [3, 3, 294, 518]

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

    process_imgs = load_and_preprocess_images(process_img_list)

    return process_imgs # [3, 3, 294, 518]

def fetch_img_list_depth_navsim(example):
    process_img_list = []
    depth_map_list = []
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

        pwd = os.getcwd()
        path = image_path.split('sensor_blobs/')[1]

        depth_map_path = pwd + '/navsim_dataset/prior_depth_map/' + path
        depth_map_path = depth_map_path.replace('.jpg', '.png') 
        pil_prior = Image.open(depth_map_path)
        np_prior = np.asarray(pil_prior).astype(np.float32)
        np_prior /= 500
        prior = cv2.resize(np_prior, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
        depth_map_list.append(prior)

    process_imgs = load_and_preprocess_images(process_img_list)

    return process_imgs, depth_map_list 


def fetch_lidardepth_navsim(example):
    depth_map_list = []
    for exp in example["messages"]:
        if exp.get("role") == "user":
            content_list = exp.get("content", [])
    for content in content_list:
        if 'image' not in content:
            continue
        image_path = content['image']
        resized_height, resized_width = content["resized_height"], content["resized_width"]

        pwd = os.getcwd()
        path = image_path.split('sensor_blobs/')[1]

        depth_map_path = pwd + '/navsim_dataset/prior_depth_map/' + path
        depth_map_path = depth_map_path.replace('.jpg', '.png') 
        pil_prior = Image.open(depth_map_path)
        np_prior = np.asarray(pil_prior).astype(np.float32)
        np_prior /= 500
        prior = cv2.resize(np_prior, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
        depth_map_list.append(prior)

    return depth_map_list

def fetch_map(example):
    depth_map_list = []
    traj_map_list = []

    content_list = example["messages"][0]["content"]
    for content in content_list:
        if 'image' not in content:
            continue
        image_path = content['image']
        resized_height, resized_width = content["resized_height"], content["resized_width"]

        # result_path = visualization(image_path)
        # if result_path:
        #     print(f"叠加图像已保存到: {result_path}")

        pwd = os.getcwd()
        path = image_path.split('sensor_blobs/')[1]

        depth_map_path = pwd + '/navsim_dataset/prior_depth_map/' + path
        depth_map_path = depth_map_path.replace('.jpg', '.png') 

        traj_heatmap_path = pwd + '/navsim_dataset/traj_heatmap/' + path

        pil_prior = Image.open(depth_map_path)
        np_prior = np.asarray(pil_prior).astype(np.float32)
        np_prior /= 500
        prior = cv2.resize(np_prior, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
        depth_map_list.append(prior)

        if os.path.exists(traj_heatmap_path):
            traj_heatmap = Image.open(traj_heatmap_path)
            np_traj_heatmap = np.asarray(traj_heatmap).astype(np.float32)
            traj_map = cv2.resize(np_traj_heatmap, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
        else:
            traj_map = np.zeros_like(prior, dtype=np.float32)
        traj_map_list.append(traj_map)

    return depth_map_list, traj_map_list


def fetch_info(example):
    infos = example["gold"]
    intrinsics = infos["intrinsics"]
    intrinsics = np.asarray(intrinsics).astype(np.float32)   # (4, 3, 3)

    s2l_rotation = infos["sensor2lidar_rotation"]
    s2l_R = np.asarray(s2l_rotation).astype(np.float32)  # (4, 3, 3)
    s2l_translation = infos["sensor2lidar_translation"]
    s2l_T = np.asarray(s2l_translation).astype(np.float32)   # (4, 3)
    extrinsics = np.concatenate([s2l_R, s2l_T[:, :, np.newaxis]], axis=2)    # (4, 3, 4)

    content_list = example["messages"][0]['content']
    images_list = []
    for content in content_list:
        if 'image' not in content:
            continue
        image_path = content['image']
        image_path = image_path.replace('file://', '')
        resized_height, resized_width = content["resized_height"], content["resized_width"]
        try:
            img = Image.open(image_path)
            np_img = np.asarray(img).astype(np.float32)
            img_ = cv2.resize(np_img, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
            images_list.append(img_)
        except Exception as e:
            print(f"[ERROR] Failed to load depth image: {image_path}")
        
    return images_list, extrinsics, intrinsics

def fetch_depth(example):
    content_list = example[0]['content']
    depth_list = []
    for content in content_list:
        if 'image' not in content:
            continue
        image_path = content['image']
        resized_height, resized_width = content["resized_height"], content["resized_width"]

        pwd = os.getcwd()
        path = image_path.split('navsim_v1.1_all/')[1]
        dep_path = os.path.join(pwd, path)
        if 'test' in dep_path:
            dep_path = dep_path.replace('test', 'test_depth').replace('.jpg', '.png')   # test_depth_prior
        elif 'trainval' in dep_path:
            dep_path = dep_path.replace('trainval', 'trainval_depth').replace('.jpg', '.png') # trainval_depth_prior

        if not os.path.exists(dep_path):
            raise FileNotFoundError(f"Depth file not found at path: {dep_path}")

        try:
            pil_prior = Image.open(dep_path)
            np_prior = np.asarray(pil_prior).astype(np.float32)
            np_prior /= 500
            prior = cv2.resize(np_prior, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
            depth_list.append(prior)
        except Exception as e:
            print(f"[ERROR] Failed to load depth image: {dep_path}")

    return depth_list

def lidar2img(example, resized_H, resized_W, img_H, img_W): # 588, 1036   1080, 1920
    img2lidar = []
    intrinsic_ = np.array(example['intrinsics'])
    sensor2lidar_rotation_ = np.array(example['sensor2lidar_rotation'])
    sensor2lidar_translation_ = np.array(example['sensor2lidar_translation'])

    # 计算缩放因子
    scale_x = resized_W / img_W
    scale_y = resized_H / img_H

    for i in range(len(intrinsic_)):    # 缩放内参
        intrinsic = intrinsic_[i]
        # 缩放 fx, fy, cx, cy
        intrinsic[0][0] *= scale_x   # fx
        intrinsic[0][2] *= scale_x   # cx
        intrinsic[1][1] *= scale_y   # fy
        intrinsic[1][2] *= scale_y   # cy
        intrinsic_[i] = intrinsic  

    for i in range(len(intrinsic_)):
        intrinsic = intrinsic_[i]
        sensor2lidar_rotation = sensor2lidar_rotation_[i]
        sensor2lidar_translation = sensor2lidar_translation_[i]

        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

        lidar2cam_r_ = np.linalg.inv(sensor2lidar_rotation)
        lidar2cam_t_ = sensor2lidar_translation @ lidar2cam_r_.T
        lidar2cam_rt_ = np.eye(4)
        lidar2cam_rt_[:3, :3] = lidar2cam_r_.T
        lidar2cam_rt_[3, :3] = -lidar2cam_t_
        lidar2cam_rt = (viewpad @ lidar2cam_rt_.T)
        img2lidar.append(np.linalg.inv(lidar2cam_rt))

    return img2lidar

def extract_camera_parameters(example, resized_H, resized_W, img_H, img_W):
    intrinsic_ = np.array(example['intrinsics'])
    sensor2lidar_rotation_ = np.array(example['sensor2lidar_rotation'])
    sensor2lidar_translation_ = np.array(example['sensor2lidar_translation'])

    camera_params = []  

    # scale_x = resized_W / img_W
    # scale_y = resized_H / img_H
    scale_x = 1 / img_W
    scale_y = 1 / img_H

    for i in range(len(intrinsic_)):
        intrinsic = intrinsic_[i]
        sensor2lidar_rotation = sensor2lidar_rotation_[i]
        sensor2lidar_translation = sensor2lidar_translation_[i]

        intrinsic[0][0] *= scale_x  # fx
        intrinsic[0][2] *= scale_x  # cx
        intrinsic[1][1] *= scale_y  # fy
        intrinsic[1][2] *= scale_y  # cy

        intrinsic = np.round(intrinsic, 2)
        sensor2lidar_rotation = np.round(sensor2lidar_rotation, 2)
        sensor2lidar_translation = np.round(sensor2lidar_translation, 2)

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = sensor2lidar_rotation
        extrinsic[:3, 3] = sensor2lidar_translation

        params = {
            'intrinsics': intrinsic.flatten(), 
            'extrinsics': extrinsic.flatten()
        }
        
        camera_params.append(params)
    
    return camera_params


def compute_ade(trajs, gt_trajs):
    trajs = torch.tensor(trajs, dtype=torch.float32)
    gt_trajs = torch.tensor(gt_trajs, dtype=torch.float32)
    ade = torch.norm(trajs - gt_trajs, dim=1).mean().item()
    return ade

def compute_L2(pkl_data):
    ades_1s = []
    ades_2s = []
    ades_3s = []
    for item in pkl_data['predictions']:
        try:
            pred = np.array(item['pre_traj'])
            gt = np.array(ast.literal_eval(item['messages'][1]['content']))
            # pred = np.cumsum(pred, axis=0)   
            # gt = np.cumsum(gt[:, :2], axis=0) 

            ade_1s = compute_ade(pred[:2], gt[:2])
            ades_1s.append(ade_1s)
            ade_2s = compute_ade(pred[:4], gt[:4])
            ades_2s.append(ade_2s)
            ade_3s = compute_ade(pred, gt)
            ades_3s.append(ade_3s)
        except Exception as e:
            print(f"Skipping ID {item['id']} due to error: {e}")

    avg_ade_1s = np.mean(ades_1s)
    avg_ade_2s = np.mean(ades_2s)
    avg_ade_3s = np.mean(ades_3s)

    return len(ades_1s), avg_ade_1s, avg_ade_2s, avg_ade_3s

def merge_data_ADE_3s(output_path):
    save_path = output_path.replace('results.pkl', '3DPE_results_path.pkl')
    if not os.path.exists(save_path):
        test_data_json = './cache/navsim_test.json'
        with open(output_path, 'rb') as f:
            data = pickle.load(f)
        with open(test_data_json, 'r') as f:
            json_data = json.load(f)

        id_to_messages = {item['id']: item['messages'] for item in json_data}

        for item in data['predictions']:
            item_id = item['id']
            if item_id in id_to_messages:
                item['messages'] = id_to_messages[item_id]
            else:
                print(f"[Warning] ID '{item_id}' not found in JSON file. Skipping.")

        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[Done] Merged data saved to: {save_path}")

    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    num, avg_ade_1s, avg_ade_2s, avg_ade_3s = compute_L2(data)
    avg_ade = (avg_ade_1s+avg_ade_2s+avg_ade_3s) / 3
    print(f"samples: {num}, ADE: 1s-{avg_ade_1s:.4f} meters, 2s-{avg_ade_2s:.4f} meters, 3s-{avg_ade_3s:.4f} meters, avg-{avg_ade:.4f}")

    # all_data_path = './cache/navsim_test-navsim+nuscenes.json'
    # with open(all_data_path, "r") as f:
    #     data = json.load(f)  # data 是一个 list
    # filtered_data = [item for item in data if "id" in item and "navsim_test_" in item["id"]]
    # with open("./cache/navsim_test.json", "w") as f:
    #     json.dump(filtered_data, f, indent=2)


def visualization(image_path):
    """
    读取图像和对应的轨迹热图，将热图以彩色方式叠加到图像上，并保存结果
    
    参数:
        image_path (str): 原始图像的路径
    """
    # 确保输出目录存在
    output_dir = os.path.join(os.getcwd(),'overlay_images')
    os.makedirs(output_dir, exist_ok=True)
    
    # 修正文件路径协议
    image_path = image_path.replace('file://', '')
    
    # 获取当前工作目录
    pwd = os.getcwd()
    
    # 提取相对路径
    try:
        relative_path = image_path.split('sensor_blobs/')[1]
    except IndexError:
        print(f"错误: 路径 {image_path} 不包含 'sensor_blobs/' 部分")
        return None
    
    # 构建轨迹热图路径
    traj_heatmap_path = os.path.join(pwd, 'navsim_dataset', 'traj_heatmap', relative_path)
    
    # 读取原始图像
    try:
        original_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"无法读取原始图像: {e}")
        return None
    
    # 读取轨迹热图
    try:
        traj_heatmap = Image.open(traj_heatmap_path).convert('L')  # 转换为单通道灰度图
    except Exception as e:
        print(f"无法读取轨迹热图: {e}")
        return None
    
    # 确保两个图像尺寸一致
    if original_image.size != traj_heatmap.size:
        traj_heatmap = traj_heatmap.resize(original_image.size, Image.LANCZOS)
    
    # 将PIL图像转换为numpy数组
    img_np = np.array(original_image)
    heatmap_np = np.array(traj_heatmap)
    
    # 将单通道热图转换为彩色热图
    # 归一化热图值到[0,1]范围
    normalized_heatmap = heatmap_np / 255.0
    
    # 使用jet颜色映射创建彩色热图
    jet_colormap = cm.get_cmap('jet')
    colored_heatmap = jet_colormap(normalized_heatmap)
    
    # 将RGBA转换为RGB（去掉Alpha通道）
    colored_heatmap = np.delete(colored_heatmap, 3, 2)
    
    # 将彩色热图转换为0-255范围的整数
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)
    
    # 叠加热图到原始图像上
    # 计算透明度（可以调整alpha值来控制热图的透明度）
    alpha = 0.6
    overlay = np.array(original_image).astype(np.float32)
    overlay = overlay * (1 - alpha) + colored_heatmap * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # 创建叠加后的PIL图像
    overlay_image = Image.fromarray(overlay)
    
    # 构建输出路径，保留原始文件名
    filename = os.path.basename(relative_path)
    output_path = os.path.join(output_dir, f"overlay_{filename}")
    
    # 保存叠加后的图像
    try:
        overlay_image.save(output_path)
        print(f"成功保存叠加图像到: {output_path}")
        return output_path
    except Exception as e:
        print(f"无法保存图像: {e}")
        return None



class CrossEntropyLossWithDynamicWeight(CrossEntropyLoss):
    def __init__(self, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
        super().__init__(weight=weight, ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing)

    def forward(self, input: torch.Tensor, target: torch.Tensor, weight_CE) -> torch.Tensor:
        batch_size, seq_len, num_classes = input.size()

        mask = (target != self.ignore_index).float()  # mask = 1 for valid tokens, 0 for -100 tokens
        
        dynamic_weights = torch.ones(batch_size, seq_len, device=input.device)

        for b in range(batch_size):
            valid_token_indices = (target[b] != self.ignore_index).nonzero().squeeze(-1)
            valid_token_count = valid_token_indices.size(0)

            if valid_token_count > 0:

                weight_increment = weight_CE / valid_token_count  
                for i, idx in enumerate(valid_token_indices):
                    dynamic_weights[b, idx] = 1 + i * weight_increment 

        loss = F.cross_entropy(input.view(-1, num_classes), target.view(-1), reduction='none', 
                               ignore_index=self.ignore_index, label_smoothing=self.label_smoothing)

        loss = loss.view(batch_size, seq_len)

        weighted_loss = loss * mask * dynamic_weights

        if self.reduction == 'mean':
            return weighted_loss.sum() / mask.sum()  
        elif self.reduction == 'sum':
            return weighted_loss.sum()  
        else:
            return weighted_loss  

class DenseMapNet(nn.Module):
    def __init__(
        self,
        n_imgs=1,
        patch_w=32,
        patch_h=16,
        embed_dims=3584,
        patch_size=14,
        out_dims=96
    ):
        super().__init__()
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.n_imgs = n_imgs
        self.patch_size = patch_size
        self.embed_dims = embed_dims
        self.out_dims = out_dims
        self.H = self.patch_h * self.patch_size * 2
        self.W = self.patch_w * self.patch_size * 2

        self.feat_unsample_traj = UpsampleWithPixelUnshuffle(in_channels=self.embed_dims, out_channels=self.out_dims)
        self.feat_unsample_depth = UpsampleWithPixelUnshuffle(in_channels=self.embed_dims, out_channels=self.out_dims)

    def forward(self, vision_hidden_states, camera_hidden_states, gt_traj_heatmap, gt_depths_map): # [B, 512, 3584] [B, 1, 3584]  [B, 1, 448, 896]
        B, n_token, dim = vision_hidden_states.shape

        vision_feat = rearrange(vision_hidden_states, 'b (c h1 h2) w -> b w c h1 h2', c=1, h1=self.patch_h, h2=self.patch_w)
        depth_feat = rearrange((vision_hidden_states + camera_hidden_states), 'b (c h1 h2) w -> b w c h1 h2', c=self.n_imgs, h1=self.patch_h, h2=self.patch_w)

        vision_feat = (
            vision_feat.transpose(1, 2)
            .clone()
            .view(
                B * self.n_imgs, dim, self.patch_h, self.patch_w
            )
        )
        depth_feat = (
            depth_feat.transpose(1, 2)
            .clone()
            .view(
                B * self.n_imgs, dim, self.patch_h, self.patch_w
            )
        )

        traj_heatmap = self.feat_unsample_traj(vision_feat)
        depth_maps = self.feat_unsample_depth(depth_feat)

        depth_map_loss = 0.1 * self.compute_depth_map_loss(depth_maps, gt_depths_map)
        traj_heatmap_loss = self.compute_traj_heatmap_loss(traj_heatmap, gt_traj_heatmap)

        return traj_heatmap_loss, depth_map_loss

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