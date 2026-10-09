import numpy as np
from .utils import format_number, closest_curve, check_indices_in_sublists
from .planning_utils import Traj_Generator

SELECTED_CLASSES = [
    "car",
    "truck",
    "trailer",
    "bus",
    "pedestrian",
    "bicycle",
    "motorcycle" "traffic_cone",
    "construction_vehicle",
]


def describe_expert(
    ego_fut_trajs,
    ego_fut_yaws,
    ego_fut_masks,
    lane_pts,
    agent_fut_trajs,
    agent_fut_masks,
    gt_names,
    gt_bboxes,
    gt_attrs,
    near_dist_thresh=20.0,
):
    """
    生成专家决策描述，返回字典格式
        Args:
            ego_fut_trajs: np.ndarray, shape=(N, 3), ground truth planning data
            ego_fut_yaws: np.ndarray, shape=(N,), ground truth yaw data
            ego_fut_masks: np.ndarray, shape=(N,), mask of planning data
            lane_pts: list, center line points
            agent_fut_trajs: np.ndarray, shape=(N, 6, 3), predicted trajectory
            agent_fut_masks: np.ndarray, shape=(N, 6), mask of predicted trajectory
            gt_names: list, list of object gt_names
            gt_bboxes: np.ndarray, shape=(N, 4), bounding boxes
            gt_attrs: list, list of object attributes
            near_dist_thresh: float, distance threshold of near objects. Defaults to 20.

    """
    planning_traj = np.cumsum(ego_fut_trajs, axis=-2)
    planning_yaw = ego_fut_yaws  # planning_yaw不是offset形式
    mask = ego_fut_masks.astype(bool)

    combined_data = list(
        zip(gt_names, gt_bboxes, gt_attrs, agent_fut_trajs, agent_fut_masks)
    )

    filtered_data = [
        (name, bbox, attr, traj, traj_mask)
        for name, bbox, attr, traj, traj_mask in combined_data
        if abs(bbox[0]) <= 50 and abs(bbox[1]) <= 50 and name in SELECTED_CLASSES
    ]
    all_gt_names = []
    all_dists = []
    all_xy = []
    for name, bbox, attr, traj, traj_mask in filtered_data:
        if attr == "":
            full_name = name
        else:
            attr = attr.split(".")[1]
            full_name = name + f".{attr}"
        traj = np.cumsum(traj, axis=-2)
        traj += bbox[:2]
        masked_planning = planning_traj[mask]
        masked_traj = traj[traj_mask.astype(bool)]
        dist_rec = np.linalg.norm(bbox[:2])

        # 检查是否有空数组，如果有，则不能计算距离
        if masked_planning.size == 0 or masked_traj.size == 0:
            l2_norm = dist_rec
        else:
            # 若两数组长度不同，取较小的长度来计算L2 Norm
            min_len = min(len(masked_planning), len(masked_traj))

            # 计算L2 Norm
            l2_norm = np.linalg.norm(
                masked_planning[:min_len][..., :2] - masked_traj[:min_len], axis=1
            ).min()
        dist = min(dist_rec, l2_norm)

        if dist <= near_dist_thresh:
            all_gt_names.append(full_name)
            all_dists.append(dist)
            all_xy.append(bbox[:2])

    # 计算速度信息
    ego_vel = _calculate_speed(ego_fut_trajs[:, :2], mask)
    integrate_traj_and_vel = np.hstack(
        (planning_traj[mask][:, :2], ego_vel[mask].reshape(-1, 1))
    )
    integrate_traj_and_yaw = np.hstack(
        (planning_traj[mask][:, :2], planning_yaw[mask].reshape(-1, 1))
    )
    speed_action = assign_longitude_behavior(integrate_traj_and_vel)

    # 计算路径信息
    traj_gentor = Traj_Generator()
    _, full_paths = traj_gentor.search_path(lane_pts)
    lane_change = _detect_lane_change(integrate_traj_and_yaw, lane_pts, full_paths)
    turning_behavior = assign_lateral_behavior(
        integrate_traj_and_yaw,
        turn_angle_th=20,
        uturn_angle_th=120,
        displacement_threshold=150,
        min_segment_length=6,
        max_segment_length=12,
        curvature_threshold=1,
    )
    # 组合路径行为
    path_actions = []
    if speed_action not in ["Stopped", "Unknown"]:
        if turning_behavior == "Go Straight":
            path_actions.append(lane_change)
        if not (lane_change != "Lane Keeping" and turning_behavior == "Go Straight"):
            path_actions.append(turning_behavior)
    path_action = "/".join(path_actions) if path_actions else ""

    # 轨迹点
    formatted_points = [
        f"({format_number(point[0], 2)}, {format_number(point[1], 2)})"
        for point in planning_traj[mask]
    ]
    expert_traj = f"[PT, {', '.join(formatted_points)}]"

    # 周围物体
    objects_near_path = []
    if len(all_dists):
        for i, obj in enumerate(all_gt_names):
            objects_near_path.append(
                f"{all_gt_names[i]} at ({format_number(all_xy[i][0])}, {format_number(all_xy[i][1])})"
            )

    # 创建和返回字典
    result = {
        "speed_action": speed_action,
        "path_action": path_action,
        "expert_traj": expert_traj,
        "objects_near_path": objects_near_path,
        # 添加原始文本描述以便向后兼容
        "description": f"Expert decision: {speed_action}{', ' + path_action if path_action else ''}{', ' + lane_change if lane_change else ''}\nExpert trajectory: {expert_traj}.\n"
        + (
            f"Objects near your path: {', '.join(objects_near_path)}."
            if objects_near_path
            else ""
        ),
    }

    return result


def assign_longitude_behavior(
    trajectory,
    min_segment_length=6,
    max_segment_length=12,
):
    if len(trajectory) <= 1:
        return "UnKnow"
    behaviors = []
    n = len(trajectory)
    i = 0
    while i < n:
        # 动态确定分段长度
        segment_length = min_segment_length
        if i + max_segment_length < n:
            # 计算当前段的曲率
            segment = trajectory[i : i + max_segment_length]
            headings = [point[2] for point in segment]
            heading_changes = np.abs(np.diff(headings))
            avg_heading_change = np.mean(heading_changes)

            # 根据曲率调整分段长度
            if avg_heading_change > curvature_threshold:
                segment_length = min_segment_length  # 曲率大，分段长度缩短
            else:
                segment_length = max_segment_length  # 曲率小，分段长度延长

        # 截取当前段
        segment_traj = trajectory[i : i + segment_length]
        if len(segment_traj) < 2:
            behaviors.append("UnKnow")
            break
        # 判断当前段的行为
        flag, behavior = check_start(segment_traj)
        if flag:
            behaviors.append(behavior)
        # 确认停止
        flag, behavior = check_stop(segment_traj)
        if flag:
            behaviors.append(behavior)
        # 确认加减速
        flag, behavior = check_acc(segment_traj)
        if flag:
            behaviors.append(behavior)

        # 移动到下一段
        i += segment_length
    if not behaviors:
        behaviors = ["UnKnow"]
    return behaviors[0]


def check_start(trajectory):
    """判断起步
    :param trajectory: 自车未来轨迹，格式为 [(x1, y1, v1), (x2, y2, v2), ...]
    """
    end_speed = trajectory[-1, 2]
    start_flag = (trajectory[:3, 2] <= 0.2).all()
    if start_flag and end_speed >= 2.0:
        return True, "Start"
    else:
        return False, "UnKnow"


def check_stop(trajectory, parking_distance_th=1.0):
    """判断静止
    :param trajectory: 自车未来轨迹，格式为 [(x1, y1, v1), (x2, y2, v2), ...]
    """
    dist = np.linalg.norm(trajectory[-1, :2] - trajectory[0, :2], axis=-1)
    if dist < parking_distance_th and len(trajectory) > 2:
        return True, "Stop"
    else:
        return False, "UnKnow"


def check_acc(
    trajectory,
    hard_brake_th=-3.0,
    decelerate_th=-1.0,
    accelerate_th=1.0,
    hard_accelerate_th=3.0,
):
    """判断加速度
    :param trajectory: 自车未来轨迹，格式为 [(x1, y1, v1), (x2, y2, v2), ...]
    """
    speed = trajectory[:, -1]
    acc_calc_index = ((0, 2), (4, 6))
    first_seg_index = (
        acc_calc_index[0][0],
        min(len(speed), acc_calc_index[0][1]),
    )
    first_seg_speed = speed[first_seg_index[0] : first_seg_index[1]]
    first_seg_speed_mean = first_seg_speed.mean()

    if acc_calc_index[1][0] >= len(speed):
        return False, "UnKnow"
    last_seg_index = (
        acc_calc_index[1][0],
        min(len(speed), acc_calc_index[1][1]),
    )
    last_seg_speed = speed[last_seg_index[0] : last_seg_index[1]]
    last_seg_speed_mean = last_seg_speed.mean()

    time_interval = (last_seg_index[1] + last_seg_index[0]) * 0.5 - (
        first_seg_index[1] + first_seg_index[0]
    ) * 0.5
    time_interval = time_interval * 0.5

    acc = (last_seg_speed_mean - first_seg_speed_mean) / time_interval
    if acc < hard_brake_th:
        return True, "HardBrake"
    elif acc >= hard_brake_th and acc < decelerate_th:
        return True, "Decelerate"
    elif acc >= decelerate_th and acc < accelerate_th:
        return True, "Maintain"
    elif acc >= accelerate_th and acc < hard_accelerate_th:
        return True, "Accelerate"
    else:
        return True, "HardAccelerate"


def _calculate_speed(traj, mask, dt=0.5):
    """
    计算速度
        Args:
            traj: np.ndarray, shape=(N, 2), trajectory data, offset format
            mask: np.ndarray, shape=(N,), mask of trajectory data
    """
    speeds = np.zeros(traj.shape[0])

    for i in range(traj.shape[0]):
        if mask[i]:
            displacement = np.linalg.norm(traj[i])
            speeds[i] = displacement / dt

    return speeds


def assign_lateral_behavior(
    trajectory,
    turn_angle_th=40,
    uturn_angle_th=120,
    displacement_threshold=1.0,
    min_segment_length=6,
    max_segment_length=12,
    curvature_threshold=0.1,
):
    """
    判断自车的横向行为，支持动态分段分析
    :param trajectory: 自车未来轨迹，格式为 [(x1, y1, heading1), (x2, y2, heading2), ...]
    :param turn_angle_th: 转弯角度阈值（单位：度）
    :param uturn_angle_th: 掉头角度阈值 (单位：度)
    :param displacement_threshold: 横向位移变化的阈值（单位：米）
    :param min_segment_length: 最小分段长度（时间步数）
    :param max_segment_length: 最大分段长度（时间步数）
    :param curvature_threshold: 曲率阈值，用于动态调整分段长度
    :return: 横向行为类别列表
    """
    behaviors = []
    n = len(trajectory)
    i = 0
    if n <= 1:
        return "UnKnow"
    while i < n:
        # 动态确定分段长度
        segment_length = min_segment_length
        if i + max_segment_length < n:
            # 计算当前段的曲率
            segment = trajectory[i : i + max_segment_length]
            heading_changes = np.abs(np.diff(segment[i : i + max_segment_length, 2]))
            avg_heading_change = np.mean(heading_changes)

            # 根据曲率调整分段长度
            if avg_heading_change > curvature_threshold:
                segment_length = min_segment_length  # 曲率大，分段长度缩短
            else:
                segment_length = max_segment_length  # 曲率小，分段长度延长

        # 截取当前段
        segment = trajectory[i : i + segment_length]
        if len(segment) < 2:
            behaviors.append("UnKnow")
            break
        # 判断当前段的行为
        behavior = assign_segment_behavior(
            segment,
            turn_angle_th=turn_angle_th,
            uturn_angle_th=uturn_angle_th,
            displacement_threshold=displacement_threshold,
        )
        behaviors.append(behavior)

        # 移动到下一段
        i += segment_length

    return behaviors[0]


def assign_segment_behavior(
    segment, turn_angle_th=40, uturn_angle_th=120, displacement_threshold=1.0
):
    """
    判断单段轨迹的行为
    :param segment: 单段轨迹，格式为 [(x1, y1, heading1), (x2, y2,heading2), ...]
    :param turn_angle_th: 转弯角度阈值（单位：度）
    :param uturn_angle_th: 掉头角度阈值 (单位：度)
    :param displacement_threshold: 横向位移变化的阈值（单位：米）
    :return: 单段轨迹的行为类别
    """
    initial_x, initial_y, initial_heading = segment[0]
    final_x, final_y, final_heading = segment[-1]

    """
        y (longitude)
        ^
        |
        |
        |
    (0.,0.) -------> x (lateral)
    
    """
    # 计算方向变化
    heading_change = final_heading - initial_heading
    heading_change = (heading_change / np.pi) * 180  # 由弧度制转为角度制
    heading_change = (heading_change + 180) % 360 - 180  # 归一化到 [-180, 180]

    # 计算横向位移变化
    displacement = final_x - initial_x

    # 判断行为类别
    if abs(heading_change) > uturn_angle_th:  # 掉头
        return "U-Turn"
    elif heading_change > turn_angle_th:  # 左转
        return "Left Turn"
    elif heading_change < -turn_angle_th:  # 右转
        return "Right Turn"
    elif displacement > displacement_threshold and heading_change < -5:  # 右侧偏移
        return "Deviate Right"
    elif displacement < -displacement_threshold and heading_change > 5:  # 左侧偏移
        return "Deviate Left"
    else:  # 直行
        return "Go Straight"


def _detect_lane_change(trajs, lane_centers_list, full_paths):
    # Step 1: Find which lane center is nearest to the origin
    origin = np.array([0, 0, 0])  # x, y ,yaw
    nearest_lane_center_indices, _, _, _ = closest_curve(origin, lane_centers_list)
    ref_index = nearest_lane_center_indices

    # Step 2 and 3: Check if the nearest lane center changes along the trajectory
    for coord in trajs:
        cur_index, _, _, _ = closest_curve(coord, lane_centers_list)
        if ref_index != cur_index:
            if (
                not check_indices_in_sublists(full_paths, ref_index, cur_index)
                and (
                    np.abs(
                        lane_centers_list[ref_index][0, 1]
                        - lane_centers_list[cur_index][0, 1]
                    )
                    > 1.5
                )
                and (
                    np.abs(
                        lane_centers_list[ref_index][-1, 1]
                        - lane_centers_list[cur_index][-1, 1]
                    )
                    > 1.5
                )
            ):
                if coord[1] > 0.1:
                    return "Left Lane Changing"
                elif coord[1] < -0.1:
                    return "Right Lane Changing"
            else:
                ref_index = cur_index

    return "Lane Keeping"
