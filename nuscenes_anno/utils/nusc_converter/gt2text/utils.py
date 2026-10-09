import math
import torch
import numpy as np
from math import factorial


def format_number(n, decimal_places=1):
    """Keep n decimal places

    Args:
        n (float): number to be formatted
        decimal_places (int, optional): decimal places. Defaults to 1.

    Returns:
        str: formatted number string
    """
    if abs(round(n, decimal_places)) <= 1e-2:
        return 0.0
    else:
        format_string = f"{{n:+.{decimal_places}f}}"
        return format_string.format(n=n)


def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))


def fit_bezier_Endpointfixed(points, n_control):
    n_points = len(points)
    A = np.zeros((n_points, n_control))
    t = np.arange(n_points) / (n_points - 1)
    for i in range(n_points):
        for j in range(n_control):
            A[i, j] = (
                comb(n_control - 1, j)
                * np.power(1 - t[i], n_control - 1 - j)
                * np.power(t[i], j)
            )
    A_BE = A[1:-1, 1:-1]
    _points = points[1:-1]
    _points = (
        _points
        - A[1:-1, 0].reshape(-1, 1) @ points[0].reshape(1, -1)
        - A[1:-1, -1].reshape(-1, 1) @ points[-1].reshape(1, -1)
    )

    conts = np.linalg.lstsq(A_BE, _points, rcond=None)

    control_points = np.zeros((n_control, points.shape[1]))
    control_points[0] = points[0]
    control_points[-1] = points[-1]
    control_points[1:-1] = conts[0]

    return control_points


def bezier_tangent_angles(control_points, t_list):
    n = len(control_points)
    derivative_points = [
        (n - 1) * (control_points[i + 1] - control_points[i]) for i in range(n - 1)
    ]
    angles = []  # 存储每个t值对应的角度

    for t in t_list:
        # 计算t时刻的贝塞尔曲线的一阶导数
        derivative_at_t = np.zeros(2)  # 假设是二维的情况
        for i, point in enumerate(derivative_points):
            coefficient = comb(n - 2, i) * ((1 - t) ** (n - 2 - i)) * (t**i)
            derivative_at_t += point * coefficient

        # 计算导数向量的角度
        angle = np.arctan2(derivative_at_t[1], derivative_at_t[0])

        # 将计算出的角度添加到列表中
        angles.append(angle)

    return np.array(angles)


def calculate_vector_angle(vector):
    """计算向量与水平轴的夹角（以度为单位）"""
    angle = np.arctan2(vector[1], vector[0]) * (180 / np.pi)
    return angle


def classify_lane_direction(lane_points):
    """根据向量角度变化判断车道线的走向"""
    vectors = np.diff(lane_points, axis=0)
    angles_deg = np.array([calculate_vector_angle(vector) for vector in vectors])

    # 统计落在各个角度区间的向量数量
    fwd_count = np.sum((angles_deg >= -45) & (angles_deg <= 45))
    left_count = np.sum((angles_deg > 45) & (angles_deg <= 135))
    opp_count = np.sum((angles_deg > 135) | (angles_deg <= -135))
    right_count = np.sum((angles_deg < -45) & (angles_deg > -135))

    # 根据最多的落在哪个区间来判断车道方向
    max_count = max(fwd_count, left_count, opp_count, right_count)
    if max_count == fwd_count:
        result = "with-flow"
    elif max_count == left_count:
        result = "allowing from right to left driving"
    elif max_count == opp_count:
        result = "opposite-flow"
    elif max_count == right_count:
        result = "allowing from left to right driving"

    directions = angles_deg - angles_deg[0]

    for i in range(len(directions)):
        if directions[i] < -180:
            directions[i] += 360
        elif directions[i] > 180:
            directions[i] -= 360

    # 防止异常数据，以平均值为准，乘2，看总体趋势
    direction = directions.mean() * 2

    if direction > 30:
        result += ", left turning lane"
    elif direction < -30:
        result += ", right turning lane"
    else:
        result += ", straight lane"

    return result


def analyze_position(x, y, angle_deg, desc_direction):
    # agent box在自车坐标系下
    #      ^ x
    #      |
    #      |
    #      |
    # y<-----
    direction = ""
    if x > 0:
        direction += "front"
    elif x < 0:
        direction += "back"

    if y > 2.5:
        direction += " left"
    elif y < -2.5:
        direction += " right"

    if desc_direction:
        if abs(angle_deg) < 45:
            direction += ", same direction as you, "
        elif abs(abs(angle_deg) - 180) < 45:
            direction += ", opposite direction from you, "
        elif abs(angle_deg - 90) < 45:
            direction += ", heading from right to left, "
        elif abs(angle_deg + 90) < 45:
            direction += ", heading from left to right, "

    return direction.strip()


def format_det_answer(full_name, bbox, vel, desc_direction):
    x = bbox[0]
    y = bbox[1]
    z = bbox[2]
    w = bbox[3]
    l = bbox[4]
    h = bbox[5]
    # 原始的bbox[6]在激光坐标系下，如下
    #      ^ y
    #      |
    #      |
    #      |
    #      -------> x
    #
    # 需要转换到如下的自车坐标系
    #      ^ x
    #      |
    #      |
    #      |
    # y<-----
    yaw = math.degrees(-(bbox[6] + np.pi / 2))
    vx = vel[0]
    vy = vel[1]

    position = analyze_position(x, y, yaw, desc_direction)

    answer = f"{full_name} in the {position} "
    answer += f"location: ({format_number(x)}, {format_number(y)})"
    # answer += f"length: {l:.1f}, width: {w:.1f}, height: {h:.1f}, "
    # answer += f"angles in degrees: {format_number(yaw)}"
    if np.sqrt(vx**2 + vy**2) > 0.2:
        answer += f", velocity: ({format_number(vx)}, {format_number(vy)}).  "
    else:
        answer += "."

    return answer


def calculate_distance(bbox):
    """计算bounding box中心点的欧式距离"""
    x, y, z = bbox[:3]
    return math.sqrt(x**2 + y**2 + z**2)


def point_in_rotated_rect(bbox, rect):

    rect = np.array(rect)
    point = np.array(bbox[:2])

    edge1 = rect[1] - rect[0]
    edge2 = rect[3] - rect[0]

    axis1 = edge1 / np.linalg.norm(edge1)
    axis2 = edge2 / np.linalg.norm(edge2)

    point_vector = point - rect[0]

    proj_on_axis1 = point_vector.dot(axis1)
    proj_on_axis2 = point_vector.dot(axis2)

    in_rect = 0 <= proj_on_axis1 <= np.linalg.norm(
        edge1
    ) and 0 <= proj_on_axis2 <= np.linalg.norm(edge2)

    return in_rect


def closest_curve(vehicle, curves):
    min_distance_angle = float("inf")
    min_distance = float("inf")
    min_angle_diff = float("inf")
    point_index = None
    closest_curve_index = None
    for i, curve in enumerate(curves):
        inter_curve = interpolate_lane_points(
            fit_bezier_Endpointfixed(curve[..., :2], 4), 100
        ).numpy()
        distance, angle, point_index = find_closest_point_and_tangent(
            inter_curve, vehicle[..., :2]
        )
        angle_diff = np.abs(angle_difference(angle, vehicle[-1]))
        distance_angle = distance + angle_diff
        if distance_angle < min_distance_angle:
            min_distance_angle = distance_angle
            min_distance = distance
            min_angle_diff = angle_diff
            closest_curve_index = i
            min_point_index = point_index

    return closest_curve_index, min_distance, min_point_index, min_angle_diff


def interpolate_lane_points(lane_points, n_points=100):
    t = np.arange(n_points) / (n_points - 1)
    interpolated_points = control_points_to_lane_points(lane_points, t)
    return interpolated_points


def control_points_to_lane_points(lanes, t):
    if isinstance(lanes, np.ndarray):
        lanes = torch.tensor(lanes, dtype=torch.float32)
    lanes = lanes.reshape(-1, lanes.shape[0], lanes.shape[-1])
    n_control = lanes.shape[1]
    n_points = len(t)
    A = np.zeros((n_points, n_control))
    for i in range(n_points):
        for j in range(n_control):
            A[i, j] = (
                comb(n_control - 1, j)
                * np.power(1 - t[i], n_control - 1 - j)
                * np.power(t[i], j)
            )
    bezier_A = torch.tensor(A, dtype=torch.float32).to(lanes.device)
    lanes = torch.einsum("ij,njk->nik", bezier_A, lanes)
    lanes = lanes.reshape(-1, lanes.shape[-1])
    return lanes


def angle_difference(angle1, angle2):
    # 计算两个角度之间的差值
    diff = angle2 - angle1
    # 调整差值使其在(-pi, pi)范围内
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return diff


def find_closest_point_and_tangent(lane_pts, target_pt, n_points=100):
    distances = np.linalg.norm(target_pt - lane_pts, axis=1)
    min_index = np.argmin(distances)
    min_distance = distances[min_index]

    t_list = np.linspace(0, 1, n_points)
    closest_t = t_list[min_index]
    angles = bezier_tangent_angles(lane_pts, [closest_t])

    return min_distance, angles[0], min_index


def get_obj_rel_position(loc):
    # nuscenes camera fov: 70 (except rear cam: 110)
    cf_fov = 70.0
    cf_start = 90.0 - cf_fov / 2
    cam_offset = 55
    cb_fov = 110

    cf_range = [cf_start, cf_start + cf_fov]  # [55, 125]
    cfl_range = [cf_start + cam_offset, cf_start + cf_fov + cam_offset]  # [110, 180]
    cbl_range = [
        cf_start + 2 * cam_offset,
        cf_start + cf_fov + 2 * cam_offset,
    ]  # [165, 235]
    cfr_range = [cf_start - cam_offset, cf_start + cf_fov - cam_offset]  # [0, 70]
    cbr_range = [
        cf_start - 2 * cam_offset,
        cf_start + cf_fov - 2 * cam_offset,
    ]  # [-55, 15]
    cb_range = [(cb_fov - 180) / 2, (cb_fov - 180) / 2 - cb_fov]  # [-35, -145]

    x, y = loc[0], loc[1]
    angle = math.degrees(math.atan2(y, x))
    angle1 = angle if angle >= 0 else angle + 360

    if angle1 >= cf_range[0] and angle1 < cf_range[1]:
        return "front"
    elif angle1 >= cfl_range[0] and angle1 < cfl_range[1]:
        return "front left"
    elif angle1 >= cbl_range[0] and angle1 < cbl_range[1]:
        return "back left"
    elif angle1 >= cfr_range[0] and angle1 < cfr_range[1]:
        return "front right"
    elif angle >= cbr_range[0] and angle < cbr_range[1]:
        return "back right"
    elif angle < cb_range[0] and angle >= cb_range[1]:
        return "back"  # overlap with side cams
    else:
        raise Exception("Not in any camera range!")


def check_indices_in_sublists(lst, index1, index2):
    # 遍历所有子list
    for sublist in lst:
        # 检查子list是否同时包含两个index
        if index1 in sublist and index2 in sublist:
            # 如果找到，返回True
            return True
    # 如果遍历结束都没有找到，返回False
    return False
