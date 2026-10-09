import numpy as np
import sys
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig, Scene

from navsim.agents.covla.utils import sensor2prompt, infer_result2poses
from navsim.agents.covla.covla_infer import infer

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from navsim.visualization.plots import plot_bev_frame
from navsim.visualization.bev import add_trajectory_to_bev_ax
from navsim.visualization.config import TRAJECTORY_CONFIG
import matplotlib.pyplot as plt

import cv2
from pathlib import Path
import uuid
import os
import json
import pickle


def visualize(token, image, gt_poses, pred_trajs, infer_result_path):
    """
    在相机图像上可视化预测轨迹和真实轨迹（含航向箭头）
    
    参数：
    image: 包含相机信息的对象，需有以下属性：
        - image_path: PosixPath 图像路径
        - intrinsics: 3x3 相机内参矩阵
        - sensor2lidar_rotation: 3x3 旋转矩阵
        - sensor2lidar_translation: 3x1 平移向量
    gt_poses: ndarray 真实轨迹 (N,3) [x, y, heading]
    pred_trajs: list[ndarray] 预测轨迹列表，每个元素为 (M,3) 的轨迹
    """
    
    # 读取图像
    img = cv2.imread(str(image.image_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{image.image_path}")
    
    img_name = os.path.splitext(os.path.basename(str(image.image_path)))[0]

    # 构建坐标系转换矩阵
    R = image.sensor2lidar_rotation
    T = image.sensor2lidar_translation
    
    # LiDAR -> Camera 齐次变换矩阵
    lidar2cam = np.eye(4)
    lidar2cam[:3, :3] = R.T
    lidar2cam[:3, 3] = -R.T @ T
    
    # 定义可视化参数
    vis_config = {
        "colors": {
            "gt": (0, 255, 0),    # 绿色-真实轨迹
            "pred": (0, 0, 255)   # 红色-预测轨迹
        },
        "arrow_length": 0.8       # 箭头长度（米）
    }
    
    # 处理真实轨迹
    gt_points = _project_trajectory(gt_poses, lidar2cam, image.intrinsics, img.shape, vis_config["arrow_length"])
    _draw_trajectory(img, gt_points, vis_config["colors"]["gt"])
    
    pred_points = _project_trajectory(pred_trajs, lidar2cam, image.intrinsics, img.shape, vis_config["arrow_length"])
    _draw_trajectory(img, pred_points, vis_config["colors"]["pred"])
    
    # 保存结果
    output_dir = Path(infer_result_path).parent / "visualization"
    # output_dir = Path.cwd() / 'visulization' / infer_result_path.split('/')[-1]
    
    # 方案2：在系统临时目录创建（跨平台兼容）
    # import tempfile
    # output_dir = Path(tempfile.gettempdir()) / "av_visualizations"

    output_dir.mkdir(parents=True, exist_ok=True)
    
    img_name_token = f"{img_name}_{token}.jpg"
    output_path = output_dir / img_name_token
    
    # 保存图像
    cv2.imwrite(str(output_path), img)
    print(f"可视化结果已保存至：{output_path}")
    return img_name_token
    

def _project_trajectory(traj, lidar2cam, intrinsics, img_shape, arrow_length):
    """将轨迹点和航向箭头投影到图像坐标系"""
    fx, fy = intrinsics[0,0], intrinsics[1,1]
    cx, cy = intrinsics[0,2], intrinsics[1,2]
    points = []
    
    for pose in traj:
        x, y, heading = pose
        
        # 转换轨迹点到相机坐标系
        point_lidar = np.array([x, y, 0, 1.0])
        point_cam = lidar2cam @ point_lidar
        X, Y, Z = point_cam[:3]
        
        if Z <= 1e-6:
            continue
            
        # 投影到图像坐标
        u = int((fx * X / Z) + cx)
        v = int((fy * Y / Z) + cy)
        # if not (0 <= u < img_shape[1] and 0 <= v < img_shape[0]):
        #     continue
        
        # 计算航向箭头
        dir_vec = np.array([np.cos(heading), np.sin(heading), 0])  # 自车坐标系方向向量
        dir_cam = lidar2cam[:3, :3] @ dir_vec  # 转换到相机坐标系
        
        # 计算箭头终点三维坐标
        arrow_end_cam = point_cam[:3] + dir_cam * arrow_length
        X_end, Y_end, Z_end = arrow_end_cam
        
        # 投影箭头终点
        if Z_end <= 1e-6:
            arrow_end = None
        else:
            u_end = int((fx * X_end / Z_end) + cx)
            v_end = int((fy * Y_end / Z_end) + cy)
            arrow_end = (u_end, v_end) if (0 <= u_end < img_shape[1] and 0 <= v_end < img_shape[0]) else None
        
        points.append((u, v, arrow_end))
    
    return points


def _draw_trajectory(img, points, color, thickness=3):
    """绘制轨迹线和航向箭头"""
    for i in range(len(points)):
        u, v, arrow_end = points[i]
        
        # 绘制轨迹点
        cv2.circle(img, (u, v), radius=6, color=color, thickness=-1)
        
        # 绘制航向箭头
        if arrow_end:
            cv2.arrowedLine(img, (u, v), arrow_end, color, thickness=2, tipLength=0.3)
        
        # 绘制轨迹连线
        if i > 0:
            prev_u, prev_v, _ = points[i-1]
            cv2.line(img, (prev_u, prev_v), (u, v), color, thickness)

# def find_response(infer_results, agent_input):
#     images = []
#     for camera in agent_input.cameras:
#         frame = str(camera.cam_f0.image_path).split('navsim_4s/')[1]
#         images.append(frame)

#     for infer_result in infer_results["predictions"]:
#         # import pdb; pdb.set_trace()
#         f = 1
#         for idx in range(1): 
#             if images[3] not in infer_result['messages'][0]['content'][0]['image']:  ##################################
#                 f = 0
#                 break
#         if f == 1:
#             return infer_result['pre_traj'], infer_result['id']
#     return None, None

def find_response(infer_results, agent_input):
    images = []
    for camera in agent_input.cameras:
        frame = str(camera.cam_f0.image_path).split('navsim_4s/')[1]
        images.append(frame)

    for infer_result in infer_results["predictions"]:
        # import pdb; pdb.set_trace()
        f = 1
        for idx in range(4): 
            if images[idx] not in infer_result['messages'][0]['content'][idx]['image']:  ##################################
                f = 0
                break
        if f == 1:
            return infer_result['pre_traj'], infer_result['id']
    return None, None

def calculate_heading_from_trajectory(trajectory_points):
    if len(trajectory_points) < 2:
        raise ValueError("轨迹必须至少包含两个点。")

    headings = [0.0] * len(trajectory_points)

    # 遍历除最后一个点之外的所有点
    for i in range(len(trajectory_points) - 1):
        p1 = np.array(trajectory_points[i])
        p2 = np.array(trajectory_points[i+1])
        
        # 计算两点之间的x和y方向的差值
        delta_x = p2[0] - p1[0]
        delta_y = p2[1] - p1[1]
        
        # 使用 arctan2 计算从X轴正方向到该向量的角度（即航向角）
        # arctan2(y, x) 返回的是弧度
        headings[i] = np.arctan2(delta_y, delta_x)

    # 对于最后一个点，我们可以假设其航向角与前一个点相同
    if len(headings) > 1:
        headings[-1] = headings[-2]

    return headings

class CovlaAgent(AbstractAgent):
    """Constant velocity baseline agent."""
    
    def __init__(self):
        super().__init__()
        
    def name(self) -> str:
        """Inherited, see superclass."""

        return self.__class__.__name__

    def initialize(self, infer_result_path, visualization=False) -> None:
        """Inherited, see superclass."""
        self.requires_scene = True
        # self.model_path = '/e2e-data/users/lg/workspace/users/zhangyikai/outputs/vla_r1/Qwen2.5-VL-7B-Instruct/05_20_16_17_23'
        # # default: Load the model on the available device(s)
        # self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     self.model_path, torch_dtype="auto", device_map="auto"
        # )
        # # default processer
        # self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.infer_result_path = infer_result_path
        self.visualization = visualization
        self._trajectory_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
        
        if self.infer_result_path.endswith('.json'):
            with open(self.infer_result_path, 'r') as f:
                self.infer_results = json.load(f)
        elif self.infer_result_path.endswith('.pkl') or self.infer_result_path.endswith('.pickle'):
            with open(self.infer_result_path, 'rb') as f:
                self.infer_results = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {self.infer_result_path}")
        # self.infer_results = json.load(open(self.infer_result_path, "r"))
        # self.infer_results = pickle.load(open(self.infer_result_path, "rb"))

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors()

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene, token: str) -> Trajectory:  # 
        # """Inherited, see superclass."""

        # ego_velocity_2d = agent_input.ego_statuses[-1].ego_velocity
        # ego_speed = (ego_velocity_2d**2).sum(-1) ** 0.5

        # num_poses, dt = (
        #     self._trajectory_sampling.num_poses,
        #     self._trajectory_sampling.interval_length,
        # )
        # poses = np.array(
        #     [[(time_idx + 1) * dt * ego_speed, 0.0, 0.0] for time_idx in range(num_poses)],
        #     dtype=np.float32,
        # )
        # prompt = sensor2prompt(agent_input)
        # infer_result = infer(prompt, self.model, self.processor)
        # poses = infer_result2poses(infer_result)
        
        # if poses is None:
        #     return None

        response, cur_id = find_response(self.infer_results, agent_input)
        if response == None:
            return None
        headings = calculate_heading_from_trajectory(response)
        poses = np.zeros(shape=(len(response), 3), dtype=np.float32)
        for i in range(len(response)):
            poses[i][0] = response[i][0]
            poses[i][1] = response[i][1]
            poses[i][2] = headings[i]
            # poses[i][2] = response[i][2]

        future_trajs = scene.get_future_trajectory(num_trajectory_frames = 8)
        image = agent_input.cameras[-1].cam_f0
        if self.visualization:
            img_name_token = visualize(token, image, future_trajs.poses, poses, self.infer_result_path)

        cur_ret = Trajectory(poses, self._trajectory_sampling)

        if self.visualization:
            output_dir = Path(self.infer_result_path).parent / "visualization_bev"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / img_name_token

            fig,ax = plot_bev_frame(scene,1)
            #####  add_fut_trajectory_to_bev ######
            fut_trajs = Trajectory(future_trajs.poses, self._trajectory_sampling)
            ax = add_trajectory_to_bev_ax(ax,fut_trajs,TRAJECTORY_CONFIG['agent_gt'])
            #####   add_pre_trajectory_to_bev   #####
            ax = add_trajectory_to_bev_ax(ax,cur_ret,TRAJECTORY_CONFIG['agent'])
            #ax = add_trajectory_to_bev_ax(ax,Trajectory(trajectory_new),TRAJECTORY_CONFIG['human'])
            plt.savefig(str(output_path))
        return cur_ret, cur_id
