from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig
import re
import ast
import math
from typing import List, Tuple
import pandas as pd
import numpy as np

prompt_template = '''
- Steering: __steering_degree__ degrees
- Speed = __speed__ m/s
- Acceleration = __acceleration__ m/s²

What acceleration and trajectory comes next?

Answer:
'''

resize_kwargs = {
      "resized_height": 448,
      "resized_width": 896
   }

class VehicleDecision:
    def __init__(self):
        # 需用户定义的阈值参数（示例值，实际需用户填充）
        self.lateral_params = {
            'delta_thresholds': [0.1, 0.3, 0.5, 0.7],  # 转向角δ阈值(rad)
            'L': 2.7                                   # 轴距(m)
        }
        
        self.longitudinal_params = {
            'speed_stop_threshold': 0.1,               # 判定静止的速度阈值(m/s)
            'speed_thresholds': [5.0, 10.0, 15.0, 20.0],  # 速度分级阈值
            'accel_thresholds': [0.5, 1.0, -0.5, -1.0]    # 加速度分级阈值[m/s²]
        }

    def _compute_steering_angle(self, angular_rate_z, velocity_x):
        """根据运动学公式计算等效转向角δ"""
        if abs(velocity_x) < 1e-3:  # 避免除零
            return 0.0
        return math.atan(angular_rate_z * self.lateral_params['L'] / velocity_x)

    def _lateral_decision(self, delta):
        """5级横向决策：[-2, -1, 0, 1, 2]"""
        th = self.lateral_params['delta_thresholds']
        abs_delta = abs(delta)
        
        if delta > 0:  # 右转
            if abs_delta > th[3]: return "Turn Right"
            elif abs_delta > th[2]: return "Slight Right"
            else: return "Straight"
        else:           # 左转
            if abs_delta > th[3]: return "Turn Right"
            elif abs_delta > th[2]: return "Slight Left"
            else: return "Straight"

    def _longitudinal_decision(self, speed, accel):
        """6级纵向决策：0-4为运动状态，5为停止"""
        s_th = self.longitudinal_params['speed_thresholds']
        a_th = self.longitudinal_params['accel_thresholds']
        
        # 静止状态判断
        if abs(speed) < self.longitudinal_params['speed_stop_threshold']:
            return 5  # stop
        
        # 运动状态分级
        if accel > a_th[1]:   return "Hard Accelerate"  # 急加速
        elif accel > a_th[0]: return "Accelerate"  # 加速
        elif accel < a_th[3]: return "Hard Brake"  # 急减速
        elif accel < a_th[2]: return "Decelerate"  # 减速
        else:                 return "Maintain Speed"  # 匀速

    def compute_decision(self, canbus_info):
        """综合四帧数据生成单一决策"""
        # 提取所有帧的物理量
        velocities_x = [v[0] for v in canbus_info['canbus_velocity']]
        accelerations_x = [a[0] for a in canbus_info['canbus_acceleration']]
        angular_rates_z = [ar[2] for ar in canbus_info['canbus_angular_rate']]
        
        # 计算每帧的转向角δ
        deltas = []
        for v, ar in zip(velocities_x, angular_rates_z):
            if abs(v) < 1e-3:
                deltas.append(0.0)  # 静止时转向角无效，置零
            else:
                delta = math.atan(ar * self.lateral_params['L'] / v)
                deltas.append(delta)
        
        #--- 横向决策综合 ---
        # 方法：取平均转向角后分级
        avg_delta = sum(deltas) / len(deltas)
        lat_decision = self._lateral_decision(avg_delta)
        
        #--- 纵向决策综合 ---
        # 规则1：若所有帧均满足静止条件，则判定为stop
        all_stop = all(abs(v) < self.longitudinal_params['speed_stop_threshold'] 
                      for v in velocities_x)
        if all_stop:
            return lat_decision, "Stop"
        
        # 规则2：动态状态取均值后分级
        avg_speed = sum(velocities_x) / len(velocities_x)
        avg_accel = sum(accelerations_x) / len(accelerations_x)
        lon_decision = self._longitudinal_decision(avg_speed, avg_accel)
        
        return lat_decision, lon_decision

def seperate_canbus(canbus):
   canbus_pose = []
   canbus_quaternion = []
   canbus_acceleration = []
   canbus_velocity = []
   canbus_angular_rate = []
   for frame in canbus:
      canbus_pose.append(frame[:3])
      canbus_quaternion.append(frame[3:7])
      canbus_acceleration.append(frame[7:10])
      canbus_velocity.append(frame[10:13])
      canbus_angular_rate.append(frame[13:16])
   canbus_info = {
      "canbus_pose": canbus_pose,
      "canbus_quaternion": canbus_quaternion,
      "canbus_acceleration": canbus_acceleration,
      "canbus_velocity": canbus_velocity,
      "canbus_angular_rate": canbus_angular_rate,
   }
   return canbus_info


def format_frames(frames):
        return [{"type": "image", "image": f"file://{i_frame}", **resize_kwargs} for i_frame in frames]

def sensor2prompt(agent_input: AgentInput):

    frames = []
    for camera in agent_input.cameras:
        frames.append(camera.cam_f0.image_path)

    ego_velocity_2d = agent_input.ego_statuses[-1].ego_velocity
    ego_speed = (ego_velocity_2d**2).sum(-1) ** 0.5
    ego_acceleration_2d = agent_input.ego_statuses[-1].ego_acceleration
    ego_acceleration = (ego_acceleration_2d**2).sum(-1) ** 0.5
    his_trajs = []
    for item in agent_input.ego_statuses[1:]:
        his_trajs.append(item.ego_pose[:2])
    his_headings = calculate_steering_angle(his_trajs, 2.5)
    steering_degree = his_headings[1]
    canbus = []
    for item in agent_input.ego_statuses:
        canbus.append(item.canbus)
    canbus_info = seperate_canbus(canbus)
    decision_maker = VehicleDecision()
    lat_decision, lon_decision = decision_maker.compute_decision(canbus_info)

    prompt = prompt_template.replace("__speed__", str(round(ego_speed,2))).replace("__acceleration__", str(round(ego_acceleration,2)))
    prompt = prompt.replace("__steering_degree__", str(round(steering_degree,2)))
    #    prompt = prompt.replace("__steering__", lat_decision).replace("__throttle__", lon_decision)
   
    message = [
        {
            "role": "user",
            "content":
                format_frames(frames) + \
                [{"type": "text", "text": prompt}],
        }
    ]
    return message
    
def parsed_trajectory(infer_result):
   # 正则提取方案
   pattern = r'(\[\[.*?\]\])'
   infer_result = infer_result[0]
   match = re.search(pattern, infer_result)
   if match:
      trajectory_str = match.group(1)
      trajectory = ast.literal_eval(trajectory_str)
      return trajectory
   else:
      return None


def calculate_headings(trajectory: List[Tuple[float, float]], output_degrees: bool = True) -> List[float]:
    """
    计算轨迹点的航向角（每个点都生成一个角度值）
    
    :param trajectory: 轨迹点列表，格式 [[x0,y0], [x1,y1], ...]
    :param output_degrees: 强制为True以保证输出角度
    :return: 航向角列表，长度与输入轨迹一致
    """

    if len(trajectory) < 2:
        raise ValueError("轨迹至少需要包含2个点才能计算航向角")

    headings = []
    for i in range(len(trajectory)):
        if i < len(trajectory) - 1:
            # 正常计算当前点到下一个点的角度
            x_current, y_current = trajectory[i]
            x_next, y_next = trajectory[i+1]
            dx = x_next - x_current
            dy = y_next - y_current
        else:
            # 最后一个点：使用前一个点的位移
            dx, dy = 0, 0  # 触发复制逻辑

        # 处理零位移或最后一个点
        if dx == 0 and dy == 0:
            if headings:
                headings.append(headings[-1])
            else:
                headings.append(0.0)
            continue

        # 计算角度
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad) % 360
        headings.append(angle_deg)

    return headings[:len(trajectory)]

def calculate_steering_angle(trajectory_points, wheelbase):
    """
    Calculates the front wheel steering angle for each point on a trajectory
    using the bicycle model and Menger curvature.

    Args:
        trajectory_points (list of tuples): A list of (x, y) coordinates
                                           representing the trajectory.
        wheelbase (float): The wheelbase L of the vehicle.

    Returns:
        list of floats: A list of steering angles (in radians) for each
                        point where it can be calculated. The first and
                        last points will have None as steering angle.
    """
    if len(trajectory_points) < 3:
        raise ValueError("Trajectory must contain at least 3 points.")

    steering_angles = [None] * len(trajectory_points) # Initialize with None

    for i in range(1, len(trajectory_points) - 1):
        p_prev = np.array(trajectory_points[i-1])
        p_curr = np.array(trajectory_points[i])
        p_next = np.array(trajectory_points[i+1])

        # Side lengths of the triangle formed by the three points
        a = np.linalg.norm(p_curr - p_next)
        b = np.linalg.norm(p_prev - p_next)
        c = np.linalg.norm(p_prev - p_curr)

        # Avoid division by zero if points are coincident
        if a == 0 or b == 0 or c == 0:
            curvature = 0.0 # Or handle as an error/special case
        else:
            # Area of the triangle using Heron's formula or cross product
            # Using Shoelace formula / cross product for robustness with collinear points
            area = 0.5 * np.abs(p_prev[0]*(p_curr[1] - p_next[1]) + \
                                p_curr[0]*(p_next[1] - p_prev[1]) + \
                                p_next[0]*(p_prev[1] - p_curr[1]))

            # Menger curvature: kappa = 4 * Area / (a * b * c)
            # If area is very small (collinear points), curvature is ~0
            if area < 1e-9: # Tolerance for collinearity
                curvature = 0.0
            else:
                # Need to determine the sign of the curvature.
                # The Menger curvature formula gives unsigned curvature.
                # We can infer the sign from the change in yaw angle or
                # by looking at the cross product of (p_curr - p_prev) and (p_next - p_curr).
                # A positive cross product (z-component) indicates a left turn (positive curvature).
                # A negative cross product indicates a right turn (negative curvature).
                # (v1_x * v2_y) - (v1_y * v2_x)
                v1 = p_curr - p_prev
                v2 = p_next - p_curr

                # Ensure vectors are not zero-length
                if np.linalg.norm(v1) < 1e-9 or np.linalg.norm(v2) < 1e-9:
                    curvature_sign = 0.0
                else:
                    # Normalize vectors before cross product for consistent yaw check,
                    # or use the raw vectors if just checking turn direction.
                    # For curvature sign, the direction of turn matters.
                    cross_product_z = v1[0] * v2[1] - v1[1] * v2[0]
                    curvature_sign = np.sign(cross_product_z)

                # If points are collinear, area is 0, curvature_sign might be 0.
                if curvature_sign == 0 and area > 1e-9 : # Check if not truly collinear but cross_product was zero
                    # This can happen if points are almost collinear or due to precision.
                    # Fallback or more robust sign determination might be needed for noisy data.
                    # For now, if area is non-zero but cross_product is zero, means it's effectively straight.
                     curvature_val = (4 * area) / (a * b * c)
                     curvature = 0.0 # Treat as straight if cross product is numerically zero
                elif area == 0.0:
                    curvature = 0.0
                else:
                    curvature_val = (4 * area) / (a * b * c)
                    curvature = curvature_sign * curvature_val


        # Steering angle: delta = atan(L * kappa)
        steering_angle = np.arctan(wheelbase * curvature)
        steering_angles[i] = steering_angle

    return steering_angles

def infer_result2poses(infer_result):
    traj = parsed_trajectory(infer_result)
    poses = np.zeros(shape=(len(traj), 3), dtype=np.float32)
    
    if traj is None:
        return poses

    headings = calculate_steering_angle(traj, 2.5)
    headings[0] = 0
    headings[-1] = 0
    
    for i in range(len(traj)):
        poses[i][0] = traj[i][0]
        poses[i][1] = traj[i][1]
        poses[i][2] = headings[i]

    return poses