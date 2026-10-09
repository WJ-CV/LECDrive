#!/usr/bin/env python

import glob
import math
import os
import pickle
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from nuplan.common.actor_state.state_representation import StateSE2
from pyquaternion import Quaternion
from tqdm import tqdm

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)


def traverse_folder(folder_path):
    motions = []
    for navlog_path in tqdm(
        glob.glob(os.path.join(folder_path, "**", "*.pkl"), recursive=True)
    ):
        # load pickle
        with open(navlog_path, "rb") as ifp:
            try:
                scene_dict_list = pickle.load(ifp)
            except Exception as e:
                print(f"{navlog_path} failed as {e}")

        # load egopose
        if len(scene_dict_list) <= 1:
            continue

        for m0, m1 in zip(scene_dict_list[:-1], scene_dict_list[1:]):
            state0 = convert_frame_to_state(m0)
            state1 = convert_frame_to_state(m1)

            local_motion = convert_absolute_to_relative_se2_array(
                StateSE2(*state0),
                np.array([state1], dtype=np.float64),
            )
            motions.append(local_motion)
    return np.vstack(motions)


def convert_frame_to_state(frame_dict):
    ego_translation = frame_dict["ego2global_translation"]
    ego_quaternion = Quaternion(*frame_dict["ego2global_rotation"])
    return np.array(
        [
            ego_translation[0],
            ego_translation[1],
            ego_quaternion.yaw_pitch_roll[0],
        ],
        dtype=np.float64,
    )


def visualize_trajectories_with_heading(
    motions, title="Motions with Heading", arrow_scale=0.2
):

    plt.figure(figsize=(30, 10))

    # indices = np.random.choice(motions.shape[0], size=int(1e5), replace=False)
    indices = np.array(range(motions.shape[0]))
    motions = motions[indices]
    x, y, heading = motions[:, 0], motions[:, 1], motions[:, 2]

    rc = [random_color() for _ in range(len(motions))]

    plt.scatter(x, y, color=rc, s=5, alpha=0.2)
    dx = arrow_scale * np.cos(heading)
    dy = arrow_scale * np.sin(heading)
    plt.quiver(
        x,
        y,
        dx,
        dy,
        angles="xy",
        scale_units="xy",
        scale=1,
        color=rc,
        headlength=0,
        headaxislength=0,
        headwidth=0,
        width=0.001,
        alpha=0.01,
    )

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True)
    plt.title(title)
    plt.axis("equal")
    plt.savefig("output/motions_with_heading.png")
    plt.close()


def random_color():
    return (random.random(), random.random(), random.random())


if __name__ == "__main__":
    # motions = traverse_folder(sys.argv[1])
    # with open("output/motions.pkl", "wb") as ofp:
    #     pickle.dump(motions, ofp)

    with open("output/motions.pkl", "rb") as ifp:
        motions = pickle.load(ifp)
        visualize_trajectories_with_heading(motions)
