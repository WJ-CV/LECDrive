import torch
import os
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torchvision import transforms as TF
from torch.cuda.amp.autocast_mode import autocast
from depth_anything_3.utils.visualize import visualize_depth
import matplotlib.pyplot as plt

def extract_camera_parameters(example, resized_W=504, img_H=1080, img_W=1920):
    intrinsic_ = np.array(example['intrinsics'])
    sensor2lidar_rotation_ = np.array(example['sensor2lidar_rotation'])
    sensor2lidar_translation_ = np.array(example['sensor2lidar_translation'])

    intrinsics_list = []
    extrinsics_list = [] 

    resized_H = resized_W * (img_H / img_W)
    resized_H = round(resized_H / 14) * 14

    scale_x = resized_W / img_W
    scale_y = resized_H / img_H

    for i in range(len(intrinsic_)):
        intrinsic = intrinsic_[i]
        sensor2lidar_rotation = sensor2lidar_rotation_[i]
        sensor2lidar_translation = sensor2lidar_translation_[i]

        intrinsic[0][0] *= scale_x  # fx
        intrinsic[0][2] *= scale_x  # cx
        intrinsic[1][1] *= scale_y  # fy
        intrinsic[1][2] *= scale_y  # cy

        intrinsic = np.round(intrinsic, 3)
        sensor2lidar_rotation = np.round(sensor2lidar_rotation, 3)
        sensor2lidar_translation = np.round(sensor2lidar_translation, 3)

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = sensor2lidar_rotation
        extrinsic[:3, 3] = sensor2lidar_translation

        intrinsics_list.append(intrinsic)
        extrinsics_list.append(extrinsic)

    return intrinsics_list, extrinsics_list


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
        # self.linear_out = nn.Linear(self.out_channels, 1)

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

class DA3genDenseMap(nn.Module):
    def __init__(
        self,
        n_imgs=3,
        patch_w=8,
        patch_h=4,
        embed_dims=3584,
        factor_1=4,
        factor_2=4,
        factor_3=4,
        out_dims=96,
        w_depth=0.5
    ):
        super().__init__()
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.n_imgs = n_imgs
        self.embed_dims = embed_dims
        self.out_dims = out_dims
        self.w_depth = w_depth

        self.feat_unsample_depth = UpsampleWithPixelUnshuffle_1(in_channels=self.embed_dims, factor_1=factor_1, factor_2=factor_2, factor_3=factor_3, out_channels=self.out_dims)
        self.densedepth_pre = nn.Sequential(
            nn.Conv2d(self.out_dims, self.out_dims, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_dims, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, depth_hidden_states, gt_depths_map, check_depth=True, out_dir='./', lidar_depth=True): # [B, 24, 3584]  [B, 3, 1078, 1918]
        B, n_token, dim = depth_hidden_states.shape

        depth_feat = depth_hidden_states.view(B, self.n_imgs, self.patch_h, self.patch_w, -1)
        depth_feat = depth_feat.permute(0, 1, 4, 2, 3)

        depth_f = depth_feat.clone().view(B * self.n_imgs, dim, self.patch_h, self.patch_w)

        depth_maps = self.feat_unsample_depth(depth_f) # B*S, 1, 560, 1120

        if gt_depths_map.shape[0]==B:
            gt_depths_map = gt_depths_map.flatten(0,1).unsqueeze(1)
        if gt_depths_map.shape[2:] != depth_maps.shape[2:]:
            depth_maps = F.interpolate(depth_maps, size=gt_depths_map.shape[2:], mode='bilinear', align_corners=False)
        
        depth_maps = self.densedepth_pre(depth_maps)
        if check_depth:
            save_path = os.path.join(out_dir, 'da3_pre_check')
            os.makedirs(save_path, exist_ok=True)
            for i in range(depth_maps.shape[0]):
                depth_map_numpy = depth_maps[i][0].float().detach().cpu().numpy()
                depth_vis = visualize_depth(depth_map_numpy, cmap="Spectral")
                plt.imsave(
                    os.path.join(save_path, f"pre_depth_{i}.png"),
                    depth_vis
                )

                gt_map_numpy = gt_depths_map[i][0].float().detach().cpu().numpy()
                gt_vis = visualize_depth(gt_map_numpy, cmap="Spectral")

                plt.imsave(
                    os.path.join(save_path, f"gt_depth_{i}.png"),
                    gt_vis
                )

                # depth_map_numpy = depth_maps[i][0].to(torch.float32).detach().cpu().numpy()
                # depth_vis = visualize_depth(depth_map_numpy, cmap="Spectral")
                # depth_pil_image = Image.fromarray((depth_vis * 255).astype(np.uint8))
                # depth_pil_image.save(os.path.join(save_path, f"pre_depth_{i}.png"))

        if lidar_depth:
            depth_map_loss = self.w_depth * self.compute_depth_map_loss(depth_maps, gt_depths_map)
        else:
            depth_map_loss = F.smooth_l1_loss(depth_maps, gt_depths_map, reduction='none') # F.l1_loss(depth_maps, gt_depths_map, reduction='none')
            mask = (gt_depths_map <= 60.0) & (gt_depths_map > 0)    # 60.0
            depth_map_loss = (depth_map_loss * mask).sum() / (mask.sum() + 1e-6)
            depth_map_loss = self.w_depth * depth_map_loss

        return  depth_map_loss

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