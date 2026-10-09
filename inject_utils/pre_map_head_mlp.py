import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F
from einops import rearrange
import numpy as np

class UpsampleWithPixelUnshuffle(nn.Module):
    def __init__(self, in_channels=3584, factor_1=4, factor_2=7, out_channels=96):
        super().__init__()
        self.factor_1 = factor_1
        self.factor_2 = factor_2
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear = nn.Linear(self.in_channels // (self.factor_1 ** 2), self.out_channels * (self.factor_2 ** 2))
        self.linear_out = nn.Linear(self.out_channels, 1)

    def forward(self, x):  # x: [B, 3584, 16, 32]
        x_1 = F.pixel_shuffle(x, upscale_factor=self.factor_1)  # -> [B, 224, 64, 128]
        B, C, H, W = x_1.shape

        x_trans = x_1.flatten(2, 3).permute(0, 2, 1) # 

        x_2 = self.linear(x_trans).permute(0, 2, 1)
        x_2_trans = x_2.view(B, -1, H, W)

        x_3 = F.pixel_shuffle(x_2_trans, upscale_factor=self.factor_2)  # -> [B, 96, 448, 896]
        x_3 = x_3.flatten(2, 3).permute(0, 2, 1) # 

        x_out = self.linear_out(x_3).permute(0, 2, 1)
        x_out = x_out.view(B, 1, H * self.factor_2, W * self.factor_2)

        return x_out


class DenseMapNet(nn.Module):
    def __init__(
        self,
        n_imgs=1,
        patch_w=32,
        patch_h=16,
        embed_dims=3584,
        patch_size=14,
        out_dims=96,
        w_depth=0.1
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

        # self.feat_unsample_traj = UpsampleWithPixelUnshuffle(in_channels=self.embed_dims, out_channels=self.out_dims)
        self.feat_unsample_depth = UpsampleWithPixelUnshuffle(in_channels=self.embed_dims, out_channels=self.out_dims)

    def forward(self, vision_hidden_states, camera_hidden_states, gt_traj_heatmap=None, gt_depths_map=None): # [B, 512, 3584] [B, 1, 3584]  [B, 1, 448, 896]
        B, n_token, dim = vision_hidden_states.shape

        vision_feat = vision_hidden_states.view(B, self.n_imgs, self.patch_h, self.patch_w, -1)
        if camera_hidden_states != None:
            camera_feat = camera_hidden_states.unsqueeze(2).unsqueeze(2)
            depth_feat = (vision_feat + camera_feat).permute(0, 4, 1, 2, 3)
        else:
            depth_feat = vision_feat.permute(0, 4, 1, 2, 3)

        if gt_depths_map is not None:
            depth_feat = (
                depth_feat.transpose(1, 2)
                .clone()
                .view(
                    B * self.n_imgs, dim, self.patch_h, self.patch_w
                )
            )

            depth_maps = self.feat_unsample_depth(depth_feat)
            depth_map_loss = self.w_depth * self.compute_depth_map_loss(depth_maps, gt_depths_map)
        else:
            depth_map_loss = 0.0
        
        if gt_traj_heatmap is not None:
            vision_feat = (
                vision_feat.permute(0, 4, 1, 2, 3).transpose(1, 2)
                .clone()
                .view(
                    B * self.n_imgs, dim, self.patch_h, self.patch_w
                )
            )
            traj_heatmap = self.feat_unsample_traj(vision_feat)
            traj_heatmap_loss = self.compute_traj_heatmap_loss(traj_heatmap, gt_traj_heatmap)

            return traj_heatmap_loss, depth_map_loss
        else:
            traj_heatmap_loss = 0
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

    def forward(self, depth_hidden_states, gt_depths_map): # [B, 24, 3584]  [B, 3, 1078, 1918]
        B, n_token, dim = depth_hidden_states.shape

        depth_feat = depth_hidden_states.view(B, self.n_imgs, self.patch_h, self.patch_w, -1)
        depth_feat = depth_feat.permute(0, 1, 4, 2, 3)

        depth_f = depth_feat.clone().view(B * self.n_imgs, dim, self.patch_h, self.patch_w)

        depth_maps = self.feat_unsample_depth(depth_f) # B*S, 1, 560, 1120

        depth_map_loss = self.w_depth * self.compute_depth_map_loss(depth_maps, gt_depths_map)
        return depth_map_loss

    def compute_depth_map_loss(self, depth_preds, gt_depths, Th=0.0):
        loss = 0.0
        pred = depth_preds.reshape(-1)
        gt = gt_depths.reshape(-1)
        fg_mask = torch.logical_and(
            gt > Th, torch.logical_not(torch.isnan(pred))
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


class CamparamNet(nn.Module):
    def __init__(
        self,
        n_imgs=3,
        embed_dims=3584,
        w_cam=0.1
    ):
        super().__init__()
        self.num_view = n_imgs
        self.hidden_size = embed_dims
        self.w_cam = w_cam

        self.intrinsics_decoder = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 4, 9),
            )
        self.extrinsics_decoder = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 4, 16),
            )

    def forward(self, camera_hidden_states, cam2lidar): # [B, 512, 3584]  [B, 1, 448, 896]
        intrinsics = torch.zeros(
            (camera_hidden_states.size(0), self.num_view, 9), 
            dtype=torch.bfloat16, 
            device=camera_hidden_states.device
        )

        extrinsics = torch.zeros(
            (camera_hidden_states.size(0), self.num_view, 16), 
            dtype=torch.bfloat16, 
            device=camera_hidden_states.device
        )
        for b in range(camera_hidden_states.size(0)):
            for v in range(self.num_view):
                params = cam2lidar[b][v]
                intri = torch.tensor(np.array(params['intrinsics']), dtype=torch.bfloat16, device=camera_hidden_states.device) 
                intrinsics[b, v] = intri
                extri = torch.tensor(np.array(params['extrinsics']), dtype=torch.bfloat16, device=camera_hidden_states.device)
                extrinsics[b, v] = extri

        pre_intrinsic = self.intrinsics_decoder(camera_hidden_states)
        pre_extrinsic = self.extrinsics_decoder(camera_hidden_states)
        cam_param_loss = torch.abs(pre_intrinsic - intrinsics).mean() + torch.abs(pre_extrinsic - extrinsics).mean()

        return self.w_cam * cam_param_loss