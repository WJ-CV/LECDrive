import math
import numpy as np
from .utils import (
    calculate_distance,
    format_det_answer,
    point_in_rotated_rect,
    closest_curve,
)
from .base_setting import *


def describe_objects2lane(
    objects_list,
    bboxes,
    velocity,
    attrs,
    lane_pts,
    crosswalks,
):
    lane_objects = {}
    lane_objects["others"] = []
    crosswalk_objects = {}

    combined_data = list(zip(objects_list, bboxes, velocity, attrs))

    # 只关心在自车周围50m以内的object
    filtered_data = [
        (name, bbox, vel, attr)
        for name, bbox, vel, attr in combined_data
        if abs(bbox[0]) <= 50 and abs(bbox[1]) <= 50
    ]
    # 按照从近到远排序
    sorted_data = sorted(filtered_data, key=lambda item: calculate_distance(item[1]))

    for name, bbox, vel, attr in sorted_data:
        desc_direction = True
        if attr == "":
            full_name = name
            desc_direction = False
        else:
            attr = attr.split(".")[1]
            full_name = name + f".{attr}"

        text = format_det_answer(full_name, bbox, vel, desc_direction)

        if "pedestrian" in name:
            for i, crosswalk in enumerate(crosswalks):
                if point_in_rotated_rect(bbox, crosswalk):
                    if i not in crosswalk_objects:
                        crosswalk_objects[i] = []
                    crosswalk_objects[i].append(text)
                # continue

        index, dist, _, _ = closest_curve(
            np.concatenate([bbox[:2], bbox[6:7]], -1), lane_pts
        )
        if dist >= 3.0:
            lane_objects["others"].append(text)
        else:
            if index not in lane_objects:
                lane_objects[index] = []
            lane_objects[index].append(text)

    return lane_objects, crosswalk_objects


def describe_ego2lane(lane_pts, lane_objects, ego_cur_pos=np.array([0.0, 0.0, 0.0])):
    index, dist, _, _ = closest_curve(ego_cur_pos, lane_pts)
    ego_info = f"your own car"

    if dist >= 3.5:
        lane_objects["othes"].insert(0, ego_info)
        return lane_objects, None
    else:
        if index not in lane_objects:
            lane_objects[index] = []
        lane_objects[index].insert(0, ego_info)
        return lane_objects, index


def describe_object(
    agent_names,
    agent_fullnames,
    agent_lcf_feats,
    agent_fut_trajs,
    agent_fut_masks,
    agent_fut_yaw,
    agent_fut_vel,
    agent_fut_goals,
):
    """Describe the annotated objects in the scene
    Args:
        agent_lcf_feats: (num_agents, 9) -> (x, y, yaw, vx, vy, width, length, height, type)
        agent_fut_trajs: (num_agents, fut_ts, 2)
        agent_fut_masks: (num_agents, fut_ts)
        agent_fut_yaw: (num_agents, fut_ts)
        agent_fut_vel: (num_agents, fut_ts, 2)
        agent_fut_goals: (num_agents)

    Returns:
        agent_desc: dict
    """
    # 累加未来轨迹
    agent_fut_trajs = (
        np.cumsum(agent_fut_trajs, axis=1) + agent_lcf_feats[:, :2][:, None]
    )
    agent_fut_yaw = np.cumsum(agent_fut_yaw, axis=1) + agent_lcf_feats[:, 2][:, None]
    agent_fut_vel = np.cumsum(agent_fut_vel, axis=1) + agent_lcf_feats[:, 3:5][:, None]
    # 根据agent车辆相对自车位置进行排序
    agent_dist = np.linalg.norm(agent_lcf_feats[:, :2], axis=1)  # (num_agents,)
    agent_sort_idx = np.argsort(agent_dist)

    # 过滤掉不需要的类别
    filted_idx_1 = [i for i in agent_sort_idx if agent_names[i] in SELECTED_OD_CLASSES]
    filted_idx_2 = [
        i
        for i in agent_sort_idx
        if agent_lcf_feats[i, 0] < PLANNING_ANCHOR_XLIM[1]
        and agent_lcf_feats[i, 0] > PLANNING_ANCHOR_XLIM[0]
        and agent_lcf_feats[i, 1] < PLANNING_ANCHOR_YLIM[1]
        and agent_lcf_feats[i, 1] > PLANNING_ANCHOR_YLIM[0]
    ]
    agent_sort_idx = [
        i for i in agent_sort_idx if i in filted_idx_1 and i in filted_idx_2
    ]

    agent_names = agent_names[agent_sort_idx]
    agent_fullnames = agent_fullnames[agent_sort_idx]
    agent_lcf_feats = agent_lcf_feats[agent_sort_idx]
    agent_fut_trajs = agent_fut_trajs[agent_sort_idx]
    agent_fut_masks = agent_fut_masks[agent_sort_idx]
    agent_fut_yaw = agent_fut_yaw[agent_sort_idx]
    agent_fut_vel = agent_fut_vel[agent_sort_idx]
    agent_fut_goals = agent_fut_goals[agent_sort_idx]
    agent_dist = agent_dist[agent_sort_idx]

    agent_desc = {}
    for i in range(len(agent_names)):
        agent_desc[f"agent_{i}"] = {}
        agent_desc[f"agent_{i}"][
            "type"
        ] = f"No. {i}, Class: {agent_names[i]}, Distance: {agent_dist[i]:.2f}"
        agent_desc[f"agent_{i}"]["geometry"] = {
            "position": f"({agent_lcf_feats[i, 0]:.1f}, {agent_lcf_feats[i, 1]:.1f})",
            "size": f"{agent_lcf_feats[i, 5]:.1f}m in width, {agent_lcf_feats[i, 6]:.1f}m in length",
            "heading": f"oriented towards {np.degrees(agent_lcf_feats[i, 2]):.1f} degree",
        }
        agent_desc[f"agent_{i}"]["static property"] = ""
        agent_desc[f"agent_{i}"]["dynamic property"] = {
            "operating status": _get_operating_status(
                agent_lcf_feats[i, 3:5], agent_fut_vel[i], agent_fut_masks[i]
            ),
            "speed": f"{np.linalg.norm(agent_lcf_feats[i, 3:5]):.1f}m/s",
            "future trajectory": _get_fut_traj_str(
                agent_fut_trajs[i], agent_fut_masks[i]
            ),
            "intention": _get_intention(
                agent_fut_trajs[i],
                agent_fut_yaw[i],
                agent_fut_masks[i],
                agent_fut_goals[i],
            ),
            "load status": "standard",
            "potential interaction with ego": "Yes",  # TODO
            "meta action": _get_meta_action(
                agent_lcf_feats[i],
                agent_fut_trajs[i],
                agent_fut_vel[i],
                agent_fut_masks[i],
            ),
        }
    return agent_desc


def _get_fut_traj_str(fut_trajs, fut_masks):
    """ "
    fut_trajs: (fut_ts, 2)
    fut_masks: (fut_ts)
    """
    valid_traj_points = []
    for t in range(fut_trajs.shape[0]):
        if fut_masks[t]:
            valid_traj_points.append(f"({fut_trajs[t, 0]:.1f}, {fut_trajs[t, 1]:.1f})")
    traj_str = "/".join(valid_traj_points)
    return traj_str


def _get_operating_status(cur_vel, fut_vel, fut_masks):
    """
    计算载具运行状态
    Args:
        cur_vx (float): 当前x方向速度
        cur_vy (float): 当前y方向速度
        fut_vel (np.array): 未来速度序列 (fut_ts, 2)
        fut_masks (np.array): 未来数据有效性掩码 (fut_ts,)
    Returns:
        str: 运行状态描述
    """
    # 计算当前速度
    current_speed = np.linalg.norm(cur_vel)

    # 提取有效未来速度
    valid_future_vel = fut_vel[fut_masks.astype(bool)]
    if len(valid_future_vel) == 0:
        return "unknown"

    # 计算平均速度和加速度
    future_speeds = np.linalg.norm(valid_future_vel, axis=-1)  # (fut_ts,)
    avg_speed = np.mean(future_speeds, axis=0)
    avg_acceleration = (
        (future_speeds[-1] - current_speed) / len(valid_future_vel)
        if len(valid_future_vel) > 0
        else 0
    )

    # 速度状态分类
    if avg_speed < 0.5:
        speed_status = "stopped"
    elif avg_speed < 5:
        speed_status = "moving slowly"
    elif avg_speed < 10:
        speed_status = "moving at moderate speed"
    else:
        speed_status = "moving fast"

    # 运动趋势判断
    if avg_acceleration > 0.5:
        trend = "accelerating"
    elif avg_acceleration < -0.5:
        trend = "decelerating"
    else:
        trend = "maintaining speed"

    return f"{speed_status} and {trend}"


def _get_intention(fut_trajs, fut_yaw, fut_masks, fut_goal, cur_speed_threshold=0.5):
    """
    结合目标方向信息的精细化意图判断
    Args:
        fut_trajs: (fut_ts, 2) 未来轨迹坐标
        fut_yaw: (fut_ts,) 未来航向角
        fut_masks: (fut_ts,) 有效数据掩码
        fut_goal: int 0-9 目标方向编码（0-7: 45度区间方向, 8: 静止）
        cur_speed_threshold: 静止判断阈值(m/s)
    """
    # 预处理有效数据
    fut_goal = fut_goal.astype(int)
    valid_traj = fut_trajs[fut_masks.astype(bool)]
    valid_yaw = fut_yaw[fut_masks.astype(bool)]

    if len(valid_traj) < 2 or fut_goal == 9:  # 明确静止目标
        return "stopped/waiting" if fut_goal == 9 else "unknown"

    # 阶段1：基于目标方向的粗粒度判断
    goal_angle = (fut_goal * np.pi / 4) % (2 * np.pi)  # 转换为弧度
    goal_direction = _get_direction_label(fut_goal)  # 方向标签化

    # 阶段2：轨迹位移分析
    displacement = valid_traj[-1] - valid_traj[0]
    displacement_norm = np.linalg.norm(displacement)

    # 阶段3：航向变化分析
    yaw_diff = valid_yaw[-1] - valid_yaw[0]

    # 阶段4：速度状态判断
    avg_speed = displacement_norm / len(valid_traj) if len(valid_traj) > 0 else 0

    # ----------------------------综合决策 ----------------------------
    if displacement_norm < 2.0 or avg_speed < cur_speed_threshold:
        return f"waiting ({goal_direction})" if fut_goal != 8 else "parked"

    # 判断方向一致性
    movement_angle = np.arctan2(displacement[1], displacement[0])
    angle_diff = (movement_angle - goal_angle + np.pi) % (2 * np.pi) - np.pi

    if abs(angle_diff) > np.radians(60):  # 运动方向与目标方向显著偏离
        return f"changing course to {goal_direction}"

    # 转弯意图增强判断
    if abs(yaw_diff) > np.radians(30):  # 累计偏航角变化阈值
        turn_type = "turning left" if yaw_diff > 0 else "turning right"
        return f"{turn_type} toward {goal_direction}"

    # 精细化场景判断
    if displacement_norm > 10:  # 长距离移动
        return f"going straight {goal_direction}"

    return f"moving {goal_direction}"


def _get_direction_label(goal_code):
    directions = [
        "east",
        "northeast",
        "north",
        "northwest",
        "west",
        "southwest",
        "south",
        "southeast",
        "static",
    ]
    return directions[goal_code] if goal_code < 9 else "unknown"


def _get_meta_action(lcf_feat, fut_trajs, fut_vel, fut_masks):
    """
    计算元动作（横向/纵向）
    Args:
        fut_trajs (np.array): 未来轨迹 (fut_ts, 2)
        fut_vel (np.array): 未来偏航角 (fut_ts, 2
        fut_masks (np.array): 数据有效性掩码 (fut_ts,)
    """
    valid_traj = fut_trajs[fut_masks.astype(bool)]
    fut_vel = fut_vel[fut_masks.astype(bool)]

    if len(valid_traj) < 2:
        return {"lateral": ["unknown"], "longitudinal": ["unknown"]}

    # 横向动作判断
    if (lcf_feat[0] - valid_traj[-1, 0]) > 2.0:
        lateral = "TurnRight"
    elif (lcf_feat[0] - valid_traj[-1, 0]) < -2.0:
        lateral = "TurnLeft"
    else:
        lateral = "MaintainLane"

    # 纵向动作判断
    cur_vel = np.linalg.norm(lcf_feat[3:5])
    fut_vel = np.linalg.norm(fut_vel, axis=-1)  # （fut_ts,）
    avg_acc = (fut_vel[-1] - cur_vel) / len(fut_vel) if len(fut_vel) > 0 else 0

    if avg_acc > 0.1:
        longitudinal = "Accelerate"
    elif avg_acc < -0.1:
        longitudinal = "Decelerate"
    else:
        longitudinal = "MaintainSpeed"

    return {"lateral": [lateral] * 2, "longitudinal": [longitudinal] * 2}
