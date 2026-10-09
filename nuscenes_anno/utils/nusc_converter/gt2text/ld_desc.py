import numpy as np

from .utils import format_number, fit_bezier_Endpointfixed, classify_lane_direction

te_category = ["traffic_light", "road_sign"]

te_attribute = [
    "unknown",
    "red",
    "green",
    "yellow",
    "go_straight",
    "turn_left",
    "turn_right",
    "no_left_turn",
    "no_right_turn",
    "u_turn",
    "no_u_turn",
    "slight_left",
    "slight_right",
]


def describe_lanes(lane_info_key, lane_info):
    lane_describe = {}
    lane_describe["lane_info_key"] = lane_info_key

    lanes_part = []
    lanes_red = []
    for i in range(len(lane_info["annotation"]["lane_centerline"])):
        bezier_lane = fit_bezier_Endpointfixed(
            lane_info["annotation"]["lane_centerline"][i]["points"], 4
        )
        desc = classify_lane_direction(
            lane_info["annotation"]["lane_centerline"][i]["points"]
        )
        bezier_lane_desc = (
            f"{desc} ["
            f"({format_number(bezier_lane[0][0])}, {format_number(bezier_lane[0][1])}), "
            f"({format_number(bezier_lane[1][0])}, {format_number(bezier_lane[1][1])}), "
            f"({format_number(bezier_lane[2][0])}, {format_number(bezier_lane[2][1])}), "
            f"({format_number(bezier_lane[3][0])}, {format_number(bezier_lane[3][1])})"
            "]"
        )
        category = [""]
        for j in range(len(lane_info["annotation"]["topology_lcte"][i])):
            if lane_info["annotation"]["topology_lcte"][i][j] != 0:
                if (
                    te_attribute[
                        lane_info["annotation"]["traffic_element"][j]["attribute"]
                    ]
                    not in category
                ):
                    category += [
                        te_category[
                            lane_info["annotation"]["traffic_element"][j]["category"]
                            - 1
                        ]
                    ]
                    category += [
                        te_attribute[
                            lane_info["annotation"]["traffic_element"][j]["attribute"]
                        ]
                    ]
                    if (
                        te_attribute[
                            lane_info["annotation"]["traffic_element"][j]["attribute"]
                        ]
                        == "red"
                    ):
                        lanes_red.append(bezier_lane)
        lanes_part.append(" ".join([bezier_lane_desc] + category))

    lane_describe["descripation"] = lanes_part
    lane_describe["red_lanes"] = lanes_red
    return lane_describe


def describe_crosswalks(crosswalks):
    crosswalks_part = []
    for i in range(len(crosswalks)):
        crosswalk = (
            f"Crosswalk ["
            f"({format_number(crosswalks[i][0][0])}, {format_number(crosswalks[i][0][1])}), "
            f"({format_number(crosswalks[i][1][0])}, {format_number(crosswalks[i][1][1])}), "
            f"({format_number(crosswalks[i][2][0])}, {format_number(crosswalks[i][2][1])}), "
            f"({format_number(crosswalks[i][3][0])}, {format_number(crosswalks[i][3][1])})"
            "]"
        )
        crosswalks_part.append(crosswalk)
    return crosswalks_part
