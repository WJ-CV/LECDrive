from importlib import invalidate_caches
import os
import math
import copy
import pickle
import json
import argparse
import numpy as np
import os.path as osp

from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils import splits
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global
from map_utils.nuscmap_extractor import NuscMapExtractor
from gt2text.od_desc import describe_object
from gt2text.tl_desc import describe_tl
from gt2text.ld_desc import describe_lanes
from gt2text.meta_action_desc import describe_expert
from gt2text.vru_desc import vru_describe

NameMapping = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}

ego_width, ego_length = 1.85, 4.084


def quart_to_rpy(qua):
    x, y, z, w = qua
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw


def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i - 1] < utimes[i] - utime):
        i -= 1
    return i


def geom2anno(map_geoms):
    MAP_CLASSES = (
        "ped_crossing",
        "divider",
        "boundary",
    )
    vectors = {}
    for cls, geom_list in map_geoms.items():
        if cls in MAP_CLASSES:
            label = MAP_CLASSES.index(cls)
            vectors[label] = []
            for geom in geom_list:
                line = np.array(geom.coords)
                vectors[label].append(line)
    return vectors


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print("total scene num: {}".format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f"{os.getcwd()}/")[-1]
                # relative path
            if not os.path.isfile(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num: {}".format(len(available_scenes)))
    return available_scenes


def get_can_bus_info(nusc, nusc_can_bus, sample):
    scene_name = nusc.get("scene", sample["scene_token"])["name"]
    sample_timestamp = sample["timestamp"]
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, "pose")
    except:
        return np.zeros(18)  # server scenes do not have can bus information.
    can_bus = []
    # during each scene, the first timestamp of can_bus may be larger than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose["utime"] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop("utime")  # useless
    pos = last_pose.pop("pos")
    rotation = last_pose.pop("orientation")
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(last_pose[key])  # 16 elements
    can_bus.extend([0.0, 0.0])
    return np.array(can_bus)


def get_global_sensor_pose(rec, nusc, inverse=False):
    lidar_sample_data = nusc.get("sample_data", rec["data"]["LIDAR_TOP"])

    ego2global = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    lidar2ego = nusc.get(
        "calibrated_sensor", lidar_sample_data["calibrated_sensor_token"]
    )

    if not inverse:
        global_from_ego = transform_matrix(
            ego2global["translation"], Quaternion(ego2global["rotation"]), inverse=False
        )
        ego_from_sensor = transform_matrix(
            lidar2ego["translation"], Quaternion(lidar2ego["rotation"]), inverse=False
        )
        pose = global_from_ego.dot(ego_from_sensor)
    else:
        sensor_from_ego = transform_matrix(
            lidar2ego["translation"], Quaternion(lidar2ego["rotation"]), inverse=True
        )
        ego_from_global = transform_matrix(
            ego2global["translation"], Quaternion(ego2global["rotation"]), inverse=True
        )
        pose = sensor_from_ego.dot(ego_from_global)
    return pose


def obtain_sensor2top(
    nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"
):
    """Obtain the info with RT matrix from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f"{os.getcwd()}/")[-1]  # relative path
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }

    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep


def nuscenes_data_prep(
    root_path,
    can_bus_root_path,
    info_prefix,
    version,
    dataset_name,
    out_dir,
    max_sweeps=10,
    lane_json_path=None,
    lane_anno_path=None,
):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    create_nuscenes_infos(
        root_path,
        out_dir,
        can_bus_root_path,
        info_prefix,
        version=version,
        max_sweeps=max_sweeps,
        lane_json_path=lane_json_path,
        lane_anno_path=lane_anno_path,
    )


def _fill_trainval_infos(
    nusc: NuScenes,
    nusc_map_extractor: NuscMapExtractor,
    nusc_can_bus: NuScenesCanBus,
    train_scenes: set,
    val_scenes: set,
    test: bool = False,
    max_sweeps: int = 10,
    fut_ts: int = 12,
    his_ts: int = 6,
    navi_fut_ts: int = 12,
    delta_t: float = 0.5,
    lane_sample_dict: dict = None,
    lane_anno_dict: dict = None,
    vru_dis_thresh: float = 20.0,
):
    """Get trainval infos from nuscenes dataset.

    Args:
        nusc (class): NuScenes dataset object.
        nusc_can_bus (class): NuScenesCanBus object.
        train_scenes (list[str]): List of training scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.
        fut_ts (int): Future time steps. Default: 12, 6s.
        his_ts (int): History time steps. Default: 6, 3s.
        navi_fut_ts (int): Navigation future time steps. Default: 12.
        delta_t (float): Delta time (s) between time steps. Default: 0.5s, i.e. 2Hz.
        lane_sample_dict (dict): Lane sample dict, key: timestamp, val: (split, segment_id, timestamp)
        vru_dis_thresh (float): VRU distance threshold. Default: 20.0m.

    Returns:
        tuple[list[dict]]: Information of training and validation sets
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []
    frame_idx = 0
    cat2idx = {}
    for idx, dic in enumerate(nusc.category):
        cat2idx[dic["name"]] = idx

    predict_helper = PredictHelper(nusc)
    sample_dict = {}
    if lane_sample_dict is not None:
        for split, segments in lane_sample_dict.items():
            for segment_id, timestamps in segments.items():
                for timestamp in timestamps:
                    sample_dict[timestamp.split(sep=".")[0]] = (
                        split,
                        segment_id,
                        timestamp.split(sep=".")[0],
                    )

    for sample in tqdm(nusc.sample):
        map_location = nusc.get(
            "log", nusc.get("scene", sample["scene_token"])["log_token"]
        )["location"]
        lidar_token = sample["data"]["LIDAR_TOP"]
        sample_data = nusc.get("sample_data", lidar_token)
        calibration = nusc.get(
            "calibrated_sensor", sample_data["calibrated_sensor_token"]
        )
        ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
        if sample["prev"] != "":
            sample_prev = nusc.get("sample", sample["prev"])
            sample_data_prev = nusc.get("sample_data", sample_prev["data"]["LIDAR_TOP"])
            ego_pose_prev = nusc.get("ego_pose", sample_data_prev["ego_pose_token"])
        else:
            ego_pose_prev = None
        if sample["next"] != "":
            sample_next = nusc.get("sample", sample["next"])
            sample_data_next = nusc.get("sample_data", sample_next["data"]["LIDAR_TOP"])
            ego_pose_next = nusc.get("ego_pose", sample_data_next["ego_pose_token"])
        else:
            ego_pose_next = None

        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        assert os.path.isfile(lidar_path)
        can_bus = get_can_bus_info(nusc, nusc_can_bus, sample)
        fut_valid_flag = True
        test_sample = copy.deepcopy(sample)
        for i in range(fut_ts):
            if test_sample["next"] != "":
                test_sample = nusc.get("sample", test_sample["next"])
            else:
                fut_valid_flag = False
        info = {
            "lidar_path": lidar_path,
            "token": sample["token"],
            "prev": sample["prev"],
            "next": sample["next"],
            "can_bus": can_bus,
            "frame_idx": frame_idx,
            "sweeps": [],
            "cams": dict(),
            "scene_token": sample["scene_token"],
            "lidar2ego_translation": calibration["translation"],
            "lidar2ego_rotation": calibration["rotation"],
            "ego2global_translation": ego_pose["translation"],
            "ego2global_rotation": ego_pose["rotation"],
            "timestamp": sample["timestamp"],
            "fut_valid_flag": fut_valid_flag,
            "map_location": map_location,
        }
        if sample["next"] == "":
            frame_idx = 0
        else:
            frame_idx += 1

        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # extract map annos
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = np.array(info["ego2global_translation"])
        lidar2global = ego2global @ lidar2ego

        translation = list(lidar2global[:3, 3])
        rotation = list(Quaternion(matrix=lidar2global).q)
        map_geoms = nusc_map_extractor.get_map_geom(map_location, translation, rotation)
        map_annos = geom2anno(map_geoms)
        info["map_annos"] = map_annos

        # obtain 6 image's information per frame
        camera_types = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
        for cam in camera_types:
            cam_token = sample["data"][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(
                nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
            )
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info["cams"].update({cam: cam_info})

        # extract openlane info
        if str(info["cams"]["CAM_FRONT"]["timestamp"]) not in sample_dict.keys():
            continue
        else:
            info["lane_info_key"] = sample_dict[
                str(info["cams"]["CAM_FRONT"]["timestamp"])
            ]
            info["lane_anno"] = lane_anno_dict[info["lane_info_key"]]

        # obtain sweeps for a single key-frame
        sample_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sample_data["prev"] == "":
                sweep = obtain_sensor2top(
                    nusc,
                    sample_data["prev"],
                    l2e_t,
                    l2e_r_mat,
                    e2g_t,
                    e2g_r_mat,
                    "lidar",
                )
                sweeps.append(sweep)
                sample_data = nusc.get("sample_data", sample_data["prev"])
            else:
                break
        info["sweeps"] = sweeps

        # obtain annotation
        if test:
            continue
        annotations = [nusc.get("sample_annotation", token) for token in sample["anns"]]
        for i in range(len(annotations)):
            if len(annotations[i]["attribute_tokens"]) == 0:
                annotations[i]["attr"] = ""
            else:
                annotations[i]["attr"] = nusc.get(
                    "attribute", annotations[i]["attribute_tokens"][0]
                )["name"]
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
        velocity = np.array([nusc.box_velocity(token)[:2] for token in sample["anns"]])
        valid_flag = np.array(
            [
                (anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0
                for anno in annotations
            ],
            dtype=bool,
        ).reshape(-1)
        # convert velo from global to lidar
        for i in range(len(boxes)):
            velo = np.array([*velocity[i], 0.0])
            velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
            velocity[i] = velo[:2]

        names = [b.name for b in boxes]
        for i in range(len(names)):
            if names[i] in NameMapping:
                names[i] = NameMapping[names[i]]
        names = np.array(names)

        fullnames = [ann_info["category_name"] for ann_info in annotations]
        fullnames = np.array(fullnames)

        gt_attrs = [ann_info["attr"] for ann_info in annotations]
        # we need to convert box size to
        # the format of our lidar coordinate system
        # which is x_size, y_size, z_size (corresponding to l, w, h)
        # gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
        gt_boxes = np.concatenate([locs, dims, rots], axis=1)
        assert len(gt_boxes) == len(annotations), f"{len(gt_boxes)}, {len(annotations)}"

        # object tracking annos: instance_ids
        instance_inds = [
            nusc.getind("instance", anno["instance_token"]) for anno in annotations
        ]

        # get future corrdinates for each box
        # [num_box, fut_ts * 2]
        num_box = len(boxes)
        gt_fut_trajs = np.zeros((num_box, fut_ts, 2))
        gt_fut_trajs_vcs = np.zeros((num_box, fut_ts, 2))
        gt_fut_yaw = np.zeros((num_box, fut_ts))
        gt_fut_yaw_vcs = np.zeros((num_box, fut_ts))
        gt_fut_vel = np.zeros((num_box, fut_ts, 2))
        gt_fut_masks = np.zeros((num_box, fut_ts))
        # gt_boxes_yaw = -(gt_boxes[:, 6] + np.pi / 2)
        gt_boxes_yaw = gt_boxes[:, 6]
        # agent lcf feat (x, y, yaw, vx, vy, width, length, height, type)
        agent_lcf_feat = np.zeros((num_box, 9))
        gt_fut_goal = np.zeros((num_box))

        for i, anno in enumerate(annotations):
            cur_box = boxes[i]
            box_lcf_vcs = Box(
                anno["translation"], anno["size"], Quaternion(anno["rotation"])
            )
            box_lcf_vcs.translate(-np.array(calibration["translation"]))
            box_lcf_vcs.rotate(Quaternion(calibration["rotation"]).inverse)
            cur_anno = anno
            lcf_anno = copy.deepcopy(anno)
            cur_vel = velocity[i]
            # x y
            agent_lcf_feat[i, 0:2] = cur_box.center[:2]
            # yaw
            agent_lcf_feat[i, 2] = gt_boxes_yaw[i]
            # vx vy
            agent_lcf_feat[i, 3:5] = velocity[i]
            # width, length, height
            agent_lcf_feat[i, 5:8] = anno["size"]
            # type
            agent_lcf_feat[i, 8] = (
                cat2idx[anno["category_name"]]
                if anno["category_name"] in cat2idx.keys()
                else -1
            )
            for j in range(fut_ts):
                if cur_anno["next"] != "":
                    anno_next = nusc.get("sample_annotation", cur_anno["next"])
                    box_next = Box(
                        anno_next["translation"],
                        anno_next["size"],
                        Quaternion(anno_next["rotation"]),
                    )
                    vel_next = nusc.box_velocity(anno_next["token"])[:2]
                    # Move box to vehicle lcf coord system.
                    box_next_vcs = copy.deepcopy(box_next)
                    box_next_vcs.translate(-np.array(lcf_anno["translation"]))
                    box_next_vcs.rotate(Quaternion(lcf_anno["rotation"]).inverse)
                    # Move box to sensor coord system.
                    box_next_vcs.translate(-np.array(calibration["translation"]))
                    box_next_vcs.rotate(Quaternion(calibration["rotation"]).inverse)
                    # Move box to ego vehicle coord system.
                    box_next.translate(-np.array(ego_pose["translation"]))
                    box_next.rotate(Quaternion(ego_pose["rotation"]).inverse)
                    # Move box to sensor coord system.
                    box_next.translate(-np.array(calibration["translation"]))
                    box_next.rotate(Quaternion(calibration["rotation"]).inverse)
                    gt_fut_trajs[i, j] = box_next.center[:2] - cur_box.center[:2]
                    gt_fut_masks[i, j] = 1
                    gt_fut_vel[i, j] = vel_next - cur_vel
                    gt_fut_trajs_vcs[i, j] = box_next_vcs.center[:2]
                    # add yaw diff
                    _, _, box_yaw = quart_to_rpy(
                        [
                            cur_box.orientation.x,
                            cur_box.orientation.y,
                            cur_box.orientation.z,
                            cur_box.orientation.w,
                        ]
                    )
                    _, _, box_yaw_next = quart_to_rpy(
                        [
                            box_next.orientation.x,
                            box_next.orientation.y,
                            box_next.orientation.z,
                            box_next.orientation.w,
                        ]
                    )
                    gt_fut_yaw[i, j] = box_yaw_next - box_yaw
                    _, _, box_lcf_yaw_vcs = quart_to_rpy(
                        [
                            box_lcf_vcs.orientation.x,
                            box_lcf_vcs.orientation.y,
                            box_lcf_vcs.orientation.z,
                            box_lcf_vcs.orientation.w,
                        ]
                    )
                    _, _, box_yaw_vcs_next = quart_to_rpy(
                        [
                            box_next_vcs.orientation.x,
                            box_next_vcs.orientation.y,
                            box_next_vcs.orientation.z,
                            box_next_vcs.orientation.w,
                        ]
                    )
                    gt_fut_yaw_vcs[i, j] = box_yaw_vcs_next - box_lcf_yaw_vcs

                    cur_anno = anno_next
                    cur_box = box_next
                    cur_vel = vel_next
                else:
                    gt_fut_trajs[i, j:] = 0
                    break
            # get agent goal
            gt_fut_coords = np.cumsum(gt_fut_trajs[i], axis=-2)
            coord_diff = gt_fut_coords[-1] - gt_fut_coords[0]
            if coord_diff.max() < 1.0:  # static
                gt_fut_goal[i] = 9
            else:
                box_mot_yaw = np.arctan2(coord_diff[1], coord_diff[0]) + np.pi
                gt_fut_goal[i] = box_mot_yaw // (np.pi / 4)  # 0-8 goal direction class

        # get ego history traj (offset format)
        ego_his_trajs = np.zeros((his_ts + 1, 3))
        ego_his_trajs_diff = np.zeros((his_ts + 1, 3))
        sample_cur = sample

        for i in range(his_ts, -1, -1):
            if sample_cur is not None:
                pose_mat = get_global_sensor_pose(sample_cur, nusc, inverse=False)
                ego_his_trajs[i] = pose_mat[:3, 3]
                has_prev = sample_cur["prev"] != ""
                has_next = sample_cur["next"] != ""
                if has_next:
                    sample_next = nusc.get("sample", sample_cur["next"])
                    pose_mat_next = get_global_sensor_pose(
                        sample_next, nusc, inverse=False
                    )
                    ego_his_trajs_diff[i] = pose_mat_next[:3, 3] - ego_his_trajs[i]
                sample_cur = (
                    nusc.get("sample", sample_cur["prev"]) if has_prev else None
                )
            else:
                ego_his_trajs[i] = ego_his_trajs[i + 1] - ego_his_trajs_diff[i + 1]
                ego_his_trajs_diff[i] = ego_his_trajs_diff[i + 1]

        # global to ego at lcf
        ego_his_trajs = ego_his_trajs - np.array(ego_pose["translation"])
        rot_mat = Quaternion(ego_pose["rotation"]).inverse.rotation_matrix
        ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
        # ego to lidar at lcf
        ego_his_trajs = ego_his_trajs - np.array(calibration["translation"])
        rot_mat = Quaternion(calibration["rotation"]).inverse.rotation_matrix
        ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
        ego_his_trajs = ego_his_trajs[1:] - ego_his_trajs[:-1]

        # get ego future traj (offset format)
        ego_fut_trajs = np.zeros((fut_ts + 1, 3))
        ego_fut_masks = np.zeros((fut_ts + 1))
        ego_fut_yaws = np.zeros((fut_ts + 1))
        ego_cur_box = Box(e2g_t, [1.73, 4.08, 1.56], Quaternion(e2g_r))
        _, _, ego_cur_box_yaw = quart_to_rpy(
            [
                ego_cur_box.orientation.x,
                ego_cur_box.orientation.y,
                ego_cur_box.orientation.z,
                ego_cur_box.orientation.w,
            ]
        )
        sample_cur = sample
        for i in range(fut_ts + 1):
            pose_mat = get_global_sensor_pose(sample_cur, nusc, inverse=False)

            sd_rec = nusc.get("sample_data", sample_cur["data"]["LIDAR_TOP"])
            rec_pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
            rec_e2g_rot = rec_pose_record["rotation"]
            rec_e2g_trans = rec_pose_record["translation"]

            ego_box = Box(rec_e2g_trans, [1.73, 4.08, 1.56], Quaternion(rec_e2g_rot))

            _, _, ego_box_yaw = quart_to_rpy(
                [
                    ego_box.orientation.x,
                    ego_box.orientation.y,
                    ego_box.orientation.z,
                    ego_box.orientation.w,
                ]
            )

            ego_fut_yaws[i] = ego_box_yaw - ego_cur_box_yaw
            ego_fut_trajs[i] = pose_mat[:3, 3]
            ego_fut_masks[i] = 1
            if sample_cur["next"] == "":
                ego_fut_trajs[i + 1 :] = ego_fut_trajs[i]
                break
            else:
                sample_cur = nusc.get("sample", sample_cur["next"])
        # global to ego at lcf
        ego_fut_trajs = ego_fut_trajs - np.array(ego_pose["translation"])
        rot_mat = Quaternion(ego_pose["rotation"]).inverse.rotation_matrix
        ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
        # ego to lidar at lcf
        ego_fut_trajs = ego_fut_trajs - np.array(calibration["translation"])
        rot_mat = Quaternion(calibration["rotation"]).inverse.rotation_matrix
        ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
        # offset from lcf -> per-step offset
        ego_fut_trajs = ego_fut_trajs[1:] - ego_fut_trajs[:-1]

        # get traget point
        n_tp = 5
        target_points = [None] * n_tp
        target_point_valid_masks = np.zeros(n_tp)
        total_dist = 0
        sample_cur = sample
        ego_prev_pos = None
        thresholds = [10, 20, 30, 40, 50]

        while 1:
            pose_mat = get_global_sensor_pose(sample_cur, nusc, inverse=False)
            ego_cur_pos = pose_mat[:3, 3]

            if ego_prev_pos is not None:
                delta = ego_cur_pos[:2] - ego_prev_pos[:2]
                delta_dist = np.linalg.norm(delta)

                # 检查所有阈值是否被跨越
                crossed = [
                    d
                    for d in thresholds
                    if (total_dist < d) and (total_dist + delta_dist >= d)
                ]
                for d in crossed:
                    ratio = (d - total_dist) / delta_dist
                    crossed_pos = ego_prev_pos[:2] + delta * ratio
                    idx = d // 10 - 1
                    if target_points[idx] is None:
                        target_points[idx] = crossed_pos
                        target_point_valid_masks[idx] = 1

                total_dist += delta_dist

            if sample_cur["next"] == "":
                break
            ego_prev_pos = ego_cur_pos.copy()
            sample_cur = nusc.get("sample", sample_cur["next"])

        target_points = np.array(
            [tp if tp is not None else [0, 0] for tp in target_points]
        )
        target_points = np.concatenate(
            [target_points, np.zeros_like(target_points[..., 0:1])], axis=-1
        )
        # global to ego at lcf
        target_points = target_points - np.array(ego_pose["translation"])
        rot_mat = Quaternion(ego_pose["rotation"]).inverse.rotation_matrix
        target_points = np.dot(rot_mat, target_points.T).T
        # ego to lidar at lcf
        target_points = target_points - np.array(calibration["translation"])
        rot_mat = Quaternion(calibration["rotation"]).inverse.rotation_matrix
        target_points = np.dot(rot_mat, target_points.T).T

        # drive command according to final fut step offset from lcf
        if ego_fut_trajs[-1][0] >= 2:
            command = np.array([1, 0, 0])  # Turn Right
        elif ego_fut_trajs[-1][0] <= -2:
            command = np.array([0, 1, 0])  # Turn Left
        else:
            command = np.array([0, 0, 1])  # Go Straight

        # get navi info from long-horizon (6s) future trajs
        ego_navi_trajs = np.zeros((navi_fut_ts + 1, 3))
        sample_cur = sample
        for i in range(navi_fut_ts + 1):
            pose_mat = get_global_sensor_pose(sample_cur, nusc, inverse=False)
            ego_navi_trajs[i] = pose_mat[:3, 3]
            if sample_cur["next"] == "":
                ego_navi_trajs[i + 1 :] = ego_navi_trajs[i]
                break
            else:
                sample_cur = nusc.get("sample", sample_cur["next"])
        # global to ego at lcf
        ego_navi_trajs = ego_navi_trajs - np.array(ego_pose["translation"])
        rot_mat = Quaternion(ego_pose["rotation"]).inverse.rotation_matrix
        ego_navi_trajs = np.dot(rot_mat, ego_navi_trajs.T).T
        # ego to lidar at lcf
        ego_navi_trajs = ego_navi_trajs - np.array(calibration["translation"])
        rot_mat = Quaternion(calibration["rotation"]).inverse.rotation_matrix
        ego_navi_trajs = np.dot(rot_mat, ego_navi_trajs.T).T

        # drive command according to final fut step offset from lcf
        ego_navi_trajs = ego_navi_trajs[1:, :2]  # discard current timestamp
        target_point = ego_navi_trajs[[-1], :2]

        # FIXME: identify the possible coordinate bug
        if target_point[0, 1] >= 20.0 and target_point[0, 0] >= 10.0:
            ego_navi_cmd = "go straight and turn right"
        elif target_point[0, 1] >= 20.0 and target_point[0, 0] <= -10.0:
            ego_navi_cmd = "go straight and turn left"
        elif target_point[0, 1] < 20.0 and target_point[0, 0] >= 10.0:
            ego_navi_cmd = "turn right"
        elif target_point[0, 1] < 20.0 and target_point[0, 0] <= -10.0:
            ego_navi_cmd = "turn left"
        else:
            ego_navi_cmd = "go straight"

        ego_lcf_feat = np.zeros(9)
        _, _, ego_yaw = quart_to_rpy(ego_pose["rotation"])
        ego_position = np.array(ego_pose["translation"])
        if ego_pose_prev is not None:
            _, _, ego_yaw_prev = quart_to_rpy(ego_pose_prev["rotation"])
            ego_position_prev = np.array(ego_pose_prev["translation"])
        if ego_pose_next is not None:
            _, _, ego_yaw_next = quart_to_rpy(ego_pose_next["rotation"])
            ego_position_next = np.array(ego_pose_next["translation"])
        assert (ego_pose_prev is not None) or (
            ego_pose_next is not None
        ), "prev and next tokens all empty"
        if ego_pose_prev is not None:
            ego_w = (ego_yaw - ego_yaw_prev) / delta_t
            ego_v = np.linalg.norm(ego_position[:2] - ego_position_prev[:2]) / delta_t
            ego_vx, ego_vy = ego_v * math.cos(ego_yaw + np.pi / 2), ego_v * math.sin(
                ego_yaw + np.pi / 2
            )
        else:
            ego_w = (ego_yaw_next - ego_yaw) / delta_t
            ego_v = np.linalg.norm(ego_position_next[:2] - ego_position[:2]) / delta_t
            ego_vx, ego_vy = ego_v * math.cos(ego_yaw + np.pi / 2), ego_v * math.sin(
                ego_yaw + np.pi / 2
            )

        ref_scene = nusc.get("scene", sample["scene_token"])
        try:
            pose_msgs = nusc_can_bus.get_messages(ref_scene["name"], "pose")
            steer_msgs = nusc_can_bus.get_messages(
                ref_scene["name"], "steeranglefeedback"
            )
            pose_uts = [msg["utime"] for msg in pose_msgs]
            steer_uts = [msg["utime"] for msg in steer_msgs]
            ref_utime = sample["timestamp"]
            pose_index = locate_message(pose_uts, ref_utime)
            pose_data = pose_msgs[pose_index]
            steer_index = locate_message(steer_uts, ref_utime)
            steer_data = steer_msgs[steer_index]
            # initial speed
            v0 = pose_data["vel"][0]  # [0] means longitudinal velocity  m/s
            # curvature (positive: turn left)
            steering = steer_data["value"]
            # flip x axis if in left-hand traffic (singapore)
            flip_flag = True if map_location.startswith("singapore") else False
            if flip_flag:
                steering *= -1
            Kappa = 2 * steering / 2.588
        except:
            delta_x = ego_his_trajs[-1, 0] + ego_fut_trajs[0, 0]
            delta_y = ego_his_trajs[-1, 1] + ego_fut_trajs[0, 1]
            v0 = np.sqrt(delta_x**2 + delta_y**2)
            Kappa = 0

        # ego lcf feat (vx, vy, ax, ay, w, length, width, vel, steer)
        ego_lcf_feat[:2] = np.array([ego_vx, ego_vy])
        ego_lcf_feat[2:4] = can_bus[7:9]
        ego_lcf_feat[4] = ego_w
        ego_lcf_feat[5:7] = np.array([ego_length, ego_width])
        ego_lcf_feat[7] = v0
        ego_lcf_feat[8] = Kappa

        # get agent describe
        od_describe = describe_object(
            names,
            fullnames,
            agent_lcf_feat,
            gt_fut_trajs,
            gt_fut_masks,
            gt_fut_yaw,
            gt_fut_vel,
            gt_fut_goal,
        )
        # get traffic light info
        traffic_elems = lane_anno_dict[info["lane_info_key"]]["annotation"][
            "traffic_element"
        ]
        traffic_elems_convert = convert_traffic_elements(traffic_elems)

        # get lane describe
        lane_describe = describe_lanes(
            info["lane_info_key"],
            lane_anno_dict[info["lane_info_key"]],
        )

        # get traffic light describe
        tl_descirbe = describe_tl(lane_anno_dict[info["lane_info_key"]])
        lane_describe["traffic_light"] = tl_descirbe

        # get expert meta action describe
        lane_pts = [
            lane["points"]
            for lane in lane_anno_dict[info["lane_info_key"]]["annotation"][
                "lane_centerline"
            ]
        ]
        expert_describe = describe_expert(
            ego_fut_trajs,
            ego_fut_yaws[1:],
            ego_fut_masks[1:],
            lane_pts,
            gt_fut_trajs,
            gt_fut_masks,
            names,
            gt_boxes,
            gt_attrs,
        )

        info.update(
            {
                "gt_boxes": gt_boxes,
                "gt_names": names,
                "gt_fullnames": fullnames,
                "gt_attrs": gt_attrs,
                "gt_velocity": velocity.reshape(-1, 2),
                "num_lidar_pts": np.array([a["num_lidar_pts"] for a in annotations]),
                "num_radar_pts": np.array([a["num_radar_pts"] for a in annotations]),
                "valid_flag": valid_flag,
                "instance_inds": instance_inds,
                "gt_agent_fut_trajs": gt_fut_trajs.reshape(-1, fut_ts * 2).astype(
                    np.float32
                ),
                "gt_agent_fut_masks": gt_fut_masks.reshape(-1, fut_ts).astype(
                    np.float32
                ),
                "gt_agent_lcf_feat": agent_lcf_feat.astype(np.float32),
                "gt_agent_fut_yaw": gt_fut_yaw.astype(np.float32),
                "gt_agent_fut_goal": gt_fut_goal.astype(np.float32),
                "gt_ego_his_trajs": ego_his_trajs[:, :2].astype(np.float32),
                "gt_ego_fut_trajs": ego_fut_trajs[:, :2].astype(np.float32),
                "gt_ego_fut_yaws": ego_fut_yaws[1:].astype(np.float32),
                "gt_ego_fut_masks": ego_fut_masks[1:].astype(np.float32),
                "gt_ego_fut_cmd": command.astype(np.float32),
                "gt_ego_lcf_feat": ego_lcf_feat.astype(np.float32),
                "gt_agent_fut_trajs_vcs": gt_fut_trajs_vcs.astype(np.float32),
                "gt_agent_fut_yaw_vcs": gt_fut_yaw_vcs.astype(np.float32),
                "gt_ego_navi_cmd": ego_navi_cmd,
                "images": [info["cams"][idx]["data_path"] for idx in camera_types],
                "target_points": target_points,
                "taget_point_valid_masks": target_point_valid_masks,
                "traffic_elems": traffic_elems_convert,
                "od_describe": od_describe,
                "ld_describe": lane_describe,
                "expert_describe": expert_describe,
            }
        )
        info["vru_describe"] = vru_describe(info, vru_dis_thresh=vru_dis_thresh)
        if info["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)
    return train_nusc_infos, val_nusc_infos


def convert_traffic_elements(traffic_elems):
    result = {"traffic_light": [], "road_sign": []}

    TRAFFIC_LIGHT_STATES = {0: "unknown", 1: "red", 2: "green", 3: "yellow"}

    ROAD_SIGN_TYPES = {
        0: "unknown",
        4: "go_straight",
        5: "turn_left",
        6: "turn_right",
        7: "no_left_turn",
        8: "no_right_turn",
        9: "u_turn",
        10: "no_u_turn",
        11: "slight_left",
        12: "slight_right",
    }

    for elem in traffic_elems:
        elem_data = {
            "bbox": elem["points"].tolist(),
            "confidence": elem.get("confidence", 1.0),
        }

        # 处理交通灯
        if elem["category"] == 1:
            elem_data["state"] = TRAFFIC_LIGHT_STATES.get(elem["attribute"], "unknown")
            result["traffic_light"].append(elem_data)

        # 处理道路标志
        elif elem["category"] == 2:
            elem_data["type"] = ROAD_SIGN_TYPES.get(elem["attribute"], "unknown")
            result["road_sign"].append(elem_data)

    return result


def create_nuscenes_infos(
    root_path,
    out_path,
    can_bus_root_path,
    info_prefix,
    version="v1.0-trainval",
    max_sweeps=10,
    roi_size=(100, 60),
    lane_json_path=None,
    lane_anno_path=None,
):
    """Create info file of nuscenes dataset.

    Given raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        out_path (str): Path of the output info file.
        can_bus_root_path (str): Path of the can bus data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data. Default: 'v1.0-trainval'.
        max_sweeps (int): Max number of sweeps. Default: 10.
        lane_json_path (str): Path to lane json file. Default: None.
    """
    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert version in available_vers
    print(version, root_path)
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    nusc_map_extractor = NuscMapExtractor(root_path, roi_size)
    nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
    if version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
        out_path = osp.join(out_path, "mini")
    elif version == "v1.0-trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    else:
        raise ValueError("Unknown version")
    os.makedirs(out_path, exist_ok=True)

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set(
        [
            available_scenes[available_scene_names.index(s)]["token"]
            for s in train_scenes
        ]
    )
    val_scenes = set(
        [available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes]
    )

    test = "test" in version
    if test:
        print(f"Test scene num {len(train_scenes)}")
    else:
        print(f"Train scene num {len(train_scenes)}, val scene num {len(val_scenes)}")

    lane_sample_dict = None
    if lane_json_path is not None and os.path.isfile(lane_json_path):
        with open(lane_json_path, "r") as f:
            lane_sample_dict = json.load(f)

    lane_anno_dict = None
    if lane_anno_path is not None and os.path.isfile(lane_anno_path):
        with open(lane_anno_path, "rb") as f:
            lane_anno_dict = pickle.load(f)

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc,
        nusc_map_extractor,
        nusc_can_bus,
        train_scenes,
        val_scenes,
        test,
        max_sweeps=max_sweeps,
        lane_sample_dict=lane_sample_dict,
        lane_anno_dict=lane_anno_dict,
    )
    metadata = dict(version=version)

    if test:
        print(f"test sample: {len(train_nusc_infos)}")
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path, f"{info_prefix}_infos_test.pkl")
        with open(info_path, "wb") as f:
            pickle.dump(data, f)
    else:
        print(
            f"train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}"
        )
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path, f"{info_prefix}_infos_train.pkl")
        with open(info_path, "wb") as f:
            pickle.dump(data, f)
        data["infos"] = val_nusc_infos
        info_val_path = osp.join(out_path, f"{info_prefix}_infos_val.pkl")
        with open(info_val_path, "wb") as f:
            pickle.dump(data, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Data converter arg parser")
    parser.add_argument(
        "--root-path",
        default="data/nuScenes",
        type=str,
        help="Root path of nusc dataset.",
    )
    parser.add_argument(
        "--canbus", type=str, default="data/nuScenes", help="Root path of nusc canbus."
    )
    parser.add_argument(
        "--lane_json_path",
        type=str,
        default="nuscenes_anno/utils/nusc_converter/dataset/data_dict_subset_B.json",
        help="Path of lane json file.",
    )
    parser.add_argument(
        "--lane_anno_path",
        type=str,
        default="nuscenes_anno/utils/nusc_converter/dataset/data_dict_sample.pkl",
        help="Path of lane annotation file.",
    )
    parser.add_argument(
        "--version",
        type=str,
        required=False,
        default="v1.0-mini",
        help="Dataset verision(v1.0-mini or v1.0-trainval).",
    )
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=10,
        required=False,
        help="Specify sweeps of lidar per example.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="nuscenes_anno/utils/nusc_converter/nusc_data",
        help="Directory to save output data.",
    )
    parser.add_argument(
        "--info-prefix", type=str, default="nusc", help="Prefix of info file."
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of threads to be used."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.version != "v1.0-mini":
        train_version = f"{args.version}-trainval"
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.info_prefix,
            version=train_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            lane_json_path=args.lane_json_path,
            lane_anno_path=args.lane_anno_path,
        )
        test_version = f"{args.version}-test"
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.info_prefix,
            version=test_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            lane_json_path=args.lane_json_path,
            lane_anno_path=args.lane_anno_path,
        )
    elif args.version == "v1.0-mini":
        train_version = f"{args.version}"
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.info_prefix,
            version=train_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            lane_json_path=args.lane_json_path,
            lane_anno_path=args.lane_anno_path,
        )
