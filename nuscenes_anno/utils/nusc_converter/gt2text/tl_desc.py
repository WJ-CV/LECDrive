def describe_tl(lane_info):
    desc_tl = "Traffic Light Existing: False"
    for i in range(len(lane_info["annotation"]["traffic_element"])):
        if lane_info["annotation"]["traffic_element"][i]["category"] == 1:
            desc_tl = "Traffic Light Existing: True"
            break

    return desc_tl
