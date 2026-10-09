import numpy as np
from .utils import get_obj_rel_position


def vru_describe(
    info: dict,
    vru_dis_thresh: float = 20.0,
    vru_classes: list = ["bicycle", "motorcycle", "pedestrian"],
):
    vru_list = []
    num_objects = info["gt_boxes"].shape[0]
    for i in range(num_objects):
        obj_loc = info["gt_boxes"][i, :2]
        obj_cls = info["gt_names"][i]
        if obj_cls not in vru_classes or np.linalg.norm(obj_loc) >= vru_dis_thresh:
            continue
        obj_rel_loc = get_obj_rel_position(obj_loc)
        # only consider VRUs at front
        if obj_rel_loc != "front":
            continue

        lat_dis, lon_dis = obj_loc[0], obj_loc[1]
        lat_pos = ""
        if lat_dis <= -2.0:
            lat_pos = f" and {int(abs(lat_dis))} meters to the left"
        elif lat_dis >= 2.0:
            lat_pos = f" and {int(abs(lat_dis))} meters to the right"
        vru_desc = (
            f"a {obj_cls} located {int(abs(lon_dis))} meters ahead of me{lat_pos}"
        )
        vru_list.append(vru_desc)
    return ", and ".join(vru_list)
