import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import trimesh
from einops import rearrange
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth


from depth_anything_3.utils.export.glb import export_to_glb
# from depth_anything_3.utils.export.gs import export_to_gs_video


from depth_anything_3.specs import Prediction
from depth_anything_3.utils.logger import logger

from depth_anything_3.utils.export.depth_vis import export_to_depth_vis
from depth_anything_3.utils.export.glb import (_filter_and_downsample,
                                               _estimate_scene_scale,
                                               _add_cameras_to_scene,
                                               _camera_frustum_lines,
                                               _index_color_rgb,
                                               set_sky_depth,
                                               get_conf_thresh,
                                               _depths_to_world_points_with_colors,
                                               _compute_alignment_transform_first_cam_glTF_center_by_points,
                                               
                                               )


def export_to_glb_now(
    prediction: Prediction,
    export_dir: str,
    num_max_points: int = 1_000_000,
    conf_thresh: float = 1.05,
    filter_black_bg: bool = False,
    filter_white_bg: bool = False,
    conf_thresh_percentile: float = 40.0,
    ensure_thresh_percentile: float = 90.0,
    sky_depth_def: float = 98.0,
    show_cameras: bool = True,
    camera_size: float = 0.03,
    # ===== 新增：Open3D 友好导出 =====
    export_o3d_ply: bool = True,         # 导出彩色点云 PLY（Open3D 可直接读）
    export_cam_ply: bool = True,         # 导出相机线框 PLY（Open3D LineSet）
) -> str:
    
    # 1) Use prediction.processed_images, which is already processed image data
    assert (
        prediction.processed_images is not None
    ), "Export to GLB: prediction.processed_images is required but not available"
    assert (
        prediction.depth is not None
    ), "Export to GLB: prediction.depth is required but not available"
    assert (
        prediction.intrinsics is not None
    ), "Export to GLB: prediction.intrinsics is required but not available"
    assert (
        prediction.extrinsics is not None
    ), "Export to GLB: prediction.extrinsics is required but not available"
    assert (
        prediction.conf is not None
    ), "Export to GLB: prediction.conf is required but not available"
    logger.info(f"conf_thresh_percentile: {conf_thresh_percentile}")
    logger.info(f"num max points: {num_max_points}")
    logger.info(f"Exporting to GLB with num_max_points: {num_max_points}")
    if prediction.processed_images is None:
        raise ValueError("prediction.processed_images is required but not available")

    
    images_u8 = prediction.processed_images  # (N,H,W,3) uint8

    # 2) Sky processing (if sky_mask is provided)
    if getattr(prediction, "sky_mask", None) is not None:
        set_sky_depth(prediction, prediction.sky_mask, sky_depth_def)

    # 3) Confidence threshold (if no conf, then no filtering)
    if filter_black_bg:
        prediction.conf[(prediction.processed_images < 16).all(axis=-1)] = 1.0
    if filter_white_bg:
        prediction.conf[(prediction.processed_images >= 240).all(axis=-1)] = 1.0
    conf_thr = get_conf_thresh(
        prediction,
        getattr(prediction, "sky_mask", None),
        conf_thresh,
        conf_thresh_percentile,
        ensure_thresh_percentile,
    )

    # 4) Back-project to world coordinates and get colors (world frame)
    points, colors = _depths_to_world_points_with_colors(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,  # w2c
        images_u8,
        prediction.conf,
        conf_thr,
    )
    
    # 5) Based on first camera orientation + glTF axis system, center by point cloud,
    # construct alignment transform, and apply to point cloud
    A = _compute_alignment_transform_first_cam_glTF_center_by_points(
        prediction.extrinsics[0], points
    )  # (4,4)

    if points.shape[0] > 0:
        points = trimesh.transform_points(points, A)

    # 6) Clean + downsample
    points, colors = _filter_and_downsample(points, colors, num_max_points)

    # 7) Assemble scene (add point cloud first)
    scene = trimesh.Scene()
    if scene.metadata is None:
        scene.metadata = {}
    scene.metadata["hf_alignment"] = A  # For camera wireframes and external reuse

    if points.shape[0] > 0:
        pc = trimesh.points.PointCloud(vertices=points, colors=colors)
        scene.add_geometry(pc)

    # # 8) Draw cameras (wireframe pyramids), using the same transform A
    # if show_cameras and prediction.intrinsics is not None and prediction.extrinsics is not None:
    #     scene_scale = _estimate_scene_scale(points, fallback=1.0)
    #     H, W = prediction.depth.shape[1:]
    #     _add_cameras_to_scene(
    #         scene=scene,
    #         K=prediction.intrinsics,
    #         ext_w2c=prediction.extrinsics,
    #         image_sizes=[(H, W)] * prediction.depth.shape[0],
    #         scale=scene_scale * camera_size,
    #     )

    # 9) Export GLB
    os.makedirs(export_dir, exist_ok=True)

    # ============================================================
    # 9.1) 新增：导出 Open3D 可视化的彩色点云 PLY（解决 glb 丢颜色问题）
    # ============================================================
    if export_o3d_ply and points.shape[0] > 0:
        try:
            import open3d as o3d
            import numpy as np

            pts = np.asarray(points)

            cols = np.asarray(colors)  # 可能是 uint8 [0,255] 或 float
            if cols.ndim == 2 and cols.shape[1] == 4:  # RGBA -> RGB
                cols = cols[:, :3]

            # 统一成 float32 [0,1]
            if cols.dtype != np.float32:
                cols = cols.astype(np.float32)
            if cols.max() > 1.0:
                cols = cols / 255.0

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float32))
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float32))

            ply_path = os.path.join(export_dir, "scene_points_rgb.ply")
            o3d.io.write_point_cloud(ply_path, pcd)
            logger.info(f"Exported Open3D point cloud to: {ply_path}")
        except Exception as e:
            logger.warning(f"Open3D PLY export failed: {e}")

    # ============================================================
    # 9.2) 新增：导出相机 frustum 为 Open3D LineSet PLY（可与点云叠加）
    # ============================================================
    if export_cam_ply and show_cameras and prediction.intrinsics is not None and prediction.extrinsics is not None:
        try:
            import open3d as o3d
            import numpy as np

            def _export_cameras_to_lineset_ply(
                ply_path: str,
                K: np.ndarray,
                ext_w2c: np.ndarray,
                image_sizes: list[tuple[int, int]],
                scale: float,
                A_align: np.ndarray,
            ):
                N = K.shape[0]
                if N == 0:
                    return

                all_points = []
                all_lines = []
                all_colors = []
                v_off = 0

                for i in range(N):
                    H, W = image_sizes[i]

                    # (8,2,3) world-frame segments
                    segs = _camera_frustum_lines(K[i], ext_w2c[i], W, H, scale)

                    # apply alignment transform A
                    segs_flat = segs.reshape(-1, 3)
                    segs_flat = trimesh.transform_points(segs_flat, A_align)

                    num_segs = segs.shape[0]
                    local_lines = np.stack(
                        [np.arange(0, 2 * num_segs, 2), np.arange(1, 2 * num_segs, 2)],
                        axis=1,
                    ) + v_off

                    rgb = _index_color_rgb(i, N)  # 0-255
                    color = (np.asarray(rgb, dtype=np.float32) / 255.0).reshape(1, 3)
                    local_colors = np.repeat(color, num_segs, axis=0)

                    all_points.append(segs_flat)
                    all_lines.append(local_lines)
                    all_colors.append(local_colors)
                    v_off += segs_flat.shape[0]

                pts = np.concatenate(all_points, axis=0)
                lines = np.concatenate(all_lines, axis=0)
                cols = np.concatenate(all_colors, axis=0)

                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(pts.astype(np.float32))
                ls.lines = o3d.utility.Vector2iVector(lines.astype(np.int32))
                ls.colors = o3d.utility.Vector3dVector(cols.astype(np.float32))
                o3d.io.write_line_set(ply_path, ls)

            scene_scale = _estimate_scene_scale(points, fallback=1.0)
            H, W = prediction.depth.shape[1:]
            cam_ply_path = os.path.join(export_dir, "cameras_frustums.ply")

            _export_cameras_to_lineset_ply(
                ply_path=cam_ply_path,
                K=prediction.intrinsics,
                ext_w2c=prediction.extrinsics,
                image_sizes=[(H, W)] * prediction.depth.shape[0],
                scale=scene_scale * camera_size,
                A_align=scene.metadata.get("hf_alignment", A),
            )
            logger.info(f"Exported Open3D camera frustums to: {cam_ply_path}")
        except Exception as e:
            logger.warning(f"Open3D camera LineSet export failed: {e}")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "depth_anything_3/pretrain"

# model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = DepthAnything3.from_pretrained(model_path)

model = model.to(device)
model.eval()
print(f"Model loaded on {device}")

# Load sample images and run inference
# image_paths = [
#     "assets/examples/SOH/000.png",
#     "assets/examples/SOH/010.png"
# ]


image_paths = [
    "/e2e-data/evad-tech-vla/wangjie68/experiments-VLA/2026-01-09_1340_repo_COVT_dinov3_sam3_da3_lidardepth-motiontoken/nips_visual/Go_Straight/2e064d6ff6305960.jpg",
    "/e2e-data/evad-tech-vla/wangjie68/experiments-VLA/2026-01-09_1340_repo_COVT_dinov3_sam3_da3_lidardepth-motiontoken/nips_visual/Go_Straight/20f2faf7aa8753b4.jpg",
    "/e2e-data/evad-tech-vla/wangjie68/experiments-VLA/2026-01-09_1340_repo_COVT_dinov3_sam3_da3_lidardepth-motiontoken/nips_visual/Go_Straight/92934ec0093055dd.jpg"
]


'''
def inference(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        align_to_input_ext_scale: bool = True,
        infer_gs: bool = False,
        render_exts: np.ndarray | None = None,
        render_ixts: np.ndarray | None = None,
        render_hw: tuple[int, int] | None = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        export_dir: str | None = None,
        export_format: str = "mini_npz",
        export_feat_layers: Sequence[int] | None = None,
        # GLB export parameters
        conf_thresh_percentile: float = 40.0,
        num_max_points: int = 1_000_000,
        show_cameras: bool = True,
        # Feat_vis export parameters
        feat_vis_fps: int = 15,
        # Other export parameters, e.g., gs_ply, gs_video
        export_kwargs: Optional[dict] = {},
    ) -> Prediction:
'''

# Run inference
prediction = model.inference(
    image=image_paths,
    process_res=1120,
    process_res_method="upper_bound_resize",
    export_dir=None,
    infer_gs=True,
)
prediction = prediction[0]
print(f"Depth shape: {prediction.depth.shape}")
print(f"Extrinsics: {prediction.extrinsics.shape if prediction.extrinsics is not None else 'None'}")
print(f"Intrinsics: {prediction.intrinsics.shape if prediction.intrinsics is not None else 'None'}")

# import pdb;pdb.set_trace()


gaussians_means = rearrange(prediction.gaussians.means, "b (v n) c -> (b v n) c", v=2)
image = prediction.processed_images
image = rearrange(image, "v h w c -> (v h w) c")
all_ply_pred = trimesh.points.PointCloud(vertices=gaussians_means.detach().cpu().numpy(),
                                colors=(image.astype(np.uint8)))
all_ply_pred.export("demo_all_pred.ply")


# Visualize input images and depth maps
n_images = prediction.depth.shape[0]

fig, axes = plt.subplots(2, n_images, figsize=(12, 6))

if n_images == 1:
    axes = axes.reshape(2, 1)

for i in range(n_images):
    # Show original image
    if prediction.processed_images is not None:
        axes[0, i].imshow(prediction.processed_images[i])
    axes[0, i].set_title(f"Input {i+1}")
    axes[0, i].axis('off')
    
    # Show depth map
    depth_vis = visualize_depth(prediction.depth[i], cmap="Spectral")
    axes[1, i].imshow(depth_vis)
    axes[1, i].set_title(f"Depth {i+1}")
    axes[1, i].axis('off')

plt.tight_layout()
# plt.show()
plt.savefig("demo_vis.png")


# export_to_glb(
#     prediction,
#     filter_black_bg=False,
#     filter_white_bg=False,
#     export_dir="./demo_vis",
#     show_cameras=True,
#     conf_thresh_percentile=5.0,
#     num_max_points=int(1_000_000),
# )

# export_to_glb_now(
#     prediction,
#     filter_black_bg=False,
#     filter_white_bg=False,
#     export_dir="./demo_vis",
#     show_cameras=True,
#     conf_thresh_percentile=5.0,
#     num_max_points=int(1_000_000),
#     export_o3d_ply=True,
#     export_cam_ply=True,
# )

# export_to_gs_video(
#     prediction,
#     export_dir="./demo_vis",
#     chunk_size=1, # 2, # 8, # 4,
#     trj_mode="smooth",
#     enable_tqdm=True,
#     vis_depth="vcat", # "hcat",
#     video_quality="high",
# )