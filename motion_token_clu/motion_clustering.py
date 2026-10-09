import argparse
import math
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm


def wrap_angle(radians: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(radians), torch.cos(radians))


def cal_polygon_contour(
    pos: Tensor,  # [n_agent, n_step, n_target, 2]
    head: Tensor,  # [n_agent, n_step, n_target]
    width_length: Tensor,  # [n_agent, 1, 1, 2]
) -> Tensor:  # [n_agent, n_step, n_target, 4, 2]
    x, y = pos[..., 0], pos[..., 1]  # [n_agent, n_step, n_target]
    width, length = width_length[..., 0], width_length[..., 1]  # [n_agent, 1 ,1]

    half_cos = 0.5 * head.cos()  # [n_agent, n_step, n_target]
    half_sin = 0.5 * head.sin()  # [n_agent, n_step, n_target]
    length_cos = length * half_cos  # [n_agent, n_step, n_target]
    length_sin = length * half_sin  # [n_agent, n_step, n_target]
    width_cos = width * half_cos  # [n_agent, n_step, n_target]
    width_sin = width * half_sin  # [n_agent, n_step, n_target]

    left_front_x = x + length_cos - width_sin
    left_front_y = y + length_sin + width_cos
    left_front = torch.stack((left_front_x, left_front_y), dim=-1)

    right_front_x = x + length_cos + width_sin
    right_front_y = y + length_sin - width_cos
    right_front = torch.stack((right_front_x, right_front_y), dim=-1)

    right_back_x = x - length_cos + width_sin
    right_back_y = y - length_sin - width_cos
    right_back = torch.stack((right_back_x, right_back_y), dim=-1)

    left_back_x = x - length_cos - width_sin
    left_back_y = y - length_sin + width_cos
    left_back = torch.stack((left_back_x, left_back_y), dim=-1)

    polygon_contour = torch.stack(
        (left_front, right_front, right_back, left_back), dim=-2
    )

    return polygon_contour


def compute_distance(X, bs=512) -> torch.Tensor:
    n_samples = X.shape[0]  # X: Nx4x2
    distances = torch.zeros([n_samples, n_samples], device=X.device, dtype=torch.bfloat16)
    for i in tqdm(range(0, n_samples, bs), desc="Computing densities"):
        end_idx = min(i + bs, n_samples)
        batch_X = X[i:end_idx]  # [batch_size, 4, 2]
        # [batch_size, n_samples]
        distances[i:end_idx] = torch.norm(
            batch_X.unsqueeze(1) - X.unsqueeze(0), dim=-1
        ).mean(dim=-1)
    return distances


def density_based_cluster(
    X,  # [n_trajs, 4, 2], bbox of the vector
    N,  # int
    tol,  # float
    a_pos,  # [n_trajs, 1, 3], the complete segment
    word_count_threshold=20,
    cal_mean_heading=True,
):
    n_total = X.shape[0]
    ret_traj_list = []

    # 计算desities
    X_cuda = X.to(torch.bfloat16).cuda()
    distances = compute_distance(X_cuda)  # NxN
    distances = distances.cpu()

    densities = (distances < tol).sum(dim=-1) - 1  # N

    groups = []
    res_mask = torch.ones(X.shape[0], dtype=bool)
    for i in range(N):
        if i == 0:
            choice_index = 0
        else:
            choice_index = torch.argmax(densities).item()

        if densities[choice_index] <= word_count_threshold:
            break

        tmp_mask = distances[choice_index] > tol
        if cal_mean_heading:
            ret_traj = a_pos[~tmp_mask].mean(0, keepdim=True)
        else:
            ret_traj = a_pos[[choice_index]]

        groups.append(
            {
                "center": ret_traj,
                "motions": a_pos[~tmp_mask],
            }
        )
        res_mask = res_mask & tmp_mask
        densities[~res_mask] = -1  # reset densities

        ret_traj_list.append(ret_traj)

        remain = res_mask.sum() * 100.0 / n_total
        n_inside = (~tmp_mask).sum().item()
        print(f"{i=}, {remain=:.6f}%, {n_inside=}")

    return torch.cat(ret_traj_list, dim=0), groups  # [N, 1, 3]


def visualize_motions_with_heading(
    groups, name="group", arrow_scale=0.2,
):

    plt.figure(figsize=(30, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups) + 1))
    for k, gp in enumerate(groups):
        motions = gp.squeeze(1)

        x, y, heading = motions[:, 0], motions[:, 1], motions[:, 2]
        plt.scatter(x, y, color=colors[k], alpha=0.5)
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
            color=colors[k],
            headlength=0,
            headaxislength=0,
            headwidth=0,
            width=0.001,
            alpha=0.05,
        )

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True)
    plt.title(name)
    plt.axis("equal")
    plt.savefig(f"output/{name}.png")
    plt.close()


def main(args):
    with open(args.motion_file, "rb") as f:
        data = pickle.load(f)
        data = torch.tensor(data)
        if len(data.shape) == 2:
            data = data.unsqueeze(1)
        indices = torch.sort(torch.randperm(data.shape[0])[:args.subsamples])[0]
        print(indices)
        data = data[indices]
        data = torch.cat((torch.zeros(1, 1, 3), data), dim=0)

    width_length = torch.tensor([2.0, 4.8])
    width_length = width_length.unsqueeze(0)  # [1, 2]

    contour = cal_polygon_contour(
        pos=data[:, 0, :2], head=data[:, 0, 2], width_length=width_length
    )

    # ret_traj = Kdisk_cluster(X=contour, N=num_cluster, tol=tol, a_pos=data)
    ret_traj, groups = density_based_cluster(X=contour, N=args.num_cluster, tol=args.tol, a_pos=data)

    flag = f"{int(args.subsamples//10000)}W_tol{args.tol}_C{args.num_cluster}_{len(groups)}"
    visualize_motions_with_heading([g["motions"] for g in groups], name=f"motions_{flag}")
    visualize_motions_with_heading([g["center"] for g in groups], name=f"centers_{flag}")
    ret_traj[:, :, -1] = wrap_angle(ret_traj[:, :, -1])

    contour = cal_polygon_contour(
        pos=ret_traj[:, :, :2],  # [N, 1, 2]
        head=ret_traj[:, :, 2],  # [N, 1]
        width_length=width_length.unsqueeze(0),
    )

    save_res = {"token": ret_traj.numpy(), "contour": contour.numpy()}
    with open(f"exp/centers_{flag}.pkl", "wb") as f:
        pickle.dump(save_res, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Process motion file and perform clustering.")
    parser.add_argument("--motion_file", type=str, default="exp/motions.pkl", help="Path to the input motion file.")
    parser.add_argument("--tol", type=float, default=0.05, help="Tolerance value for processing (default: 0.05).")
    parser.add_argument("--num_cluster", type=int, default=2048, help="Number of clusters (default: 2048).")
    parser.add_argument("--subsamples", type=int, default=120000, help="Number of sub samples (default: 12W).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
