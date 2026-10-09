import math
import pickle
import sys
from pathlib import Path

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


def Kdisk_cluster(
    X,  # [n_trajs, 4, 2], bbox of the last point of the segment
    N,  # int
    tol,  # float
    a_pos,  # [n_trajs, 1, 3], the complete segment
    cal_mean_heading=True,
):
    n_total = X.shape[0]
    ret_traj_list = []

    for i in range(N):
        if i == 0:
            choice_index = 0  # always include [0, 0, 0]
        else:
            choice_index = torch.randint(0, X.shape[0], (1,)).item()
        x0 = X[choice_index]
        # res_mask = torch.sum((X - x0) ** 2, dim=[1, 2]) / 4.0 > (tol**2)
        res_mask = (
            torch.norm(X - x0, dim=-1).mean(-1) > tol
        )  # 大于阈值的轮廓点保留下来，作为下一次筛选范围
        if cal_mean_heading:
            ret_traj = a_pos[~res_mask].mean(0, keepdim=True)
        else:
            ret_traj = a_pos[[choice_index]]

        X = X[res_mask]
        if X.shape[0] == 0:
            break
        a_pos = a_pos[res_mask]
        ret_traj_list.append(ret_traj)

        remain = X.shape[0] * 100.0 / n_total
        n_inside = (~res_mask).sum().item()
        print(f"{i=}, {remain=:.8f}%, {n_inside=}")

    return torch.cat(ret_traj_list, dim=0)  # [N, 1, 3]


def adaptive_sigama_selection(X, k=20):
    """
    自适应选择带宽参数
    """
    n_samples = X.shape[0]
    k = min(k, n_samples - 1)

    sample_size = min(1000, n_samples)
    sample_indices = torch.randperm(n_samples)[:sample_size]
    sample_X = X[sample_indices]

    # 计算每个样本点到其他点的距离
    distances = []
    for i in range(sample_size):
        dist = torch.norm(sample_X[i : i + 1] - sample_X, dim=-1).mean(dim=-1)

        dist = dist[dist > 0]
        if len(dist) >= k:
            # 取第k近的距离
            kth_dist = torch.topk(dist, k, largest=False)[0][-1]
            distances.append(kth_dist)

    if distances:
        sigama = torch.min(torch.stack(distances)).item()
    else:
        sigama = 0.01

    return sigama


def compute_density_gpu(
    X,  # [n_trajs, 4, 2], bbox of the vector
    sigama=0.01,
    batch_size=1000,
):
    n_samples = X.shape[0]
    densities = torch.zeros(n_samples, device=X.device)

    print("Selecting sigama...")
    sigama = adaptive_sigama_selection(X)
    print(f"Selected sigama: {sigama:.4f}")

    for i in tqdm(range(0, n_samples, batch_size), desc="Computing densities"):
        end_idx = min(i + batch_size, n_samples)
        batch_X = X[i:end_idx]  # [batch_size,4,2]

        if hasattr(torch, "cdist"):
            batch_X_flat = batch_X.reshape(batch_X.shape[0], -1)
            X_flat = X.reshape(X.shape[0], -1)
            distances = torch.cdist(batch_X_flat, X_flat) / 4.0  # 平均到4个点
        else:
            distances = torch.norm(batch_X.unsqueeze(1) - X.unsqueeze(0), dim=-1).mean(
                dim=-1
            )  # [batch_size, n_samples]

        kernel_values = torch.exp(-0.5 * (distances / sigama) ** 2)
        densities[i:end_idx] = kernel_values.sum(dim=1)  # [batch_size,]

    # 归一化
    densities = densities / densities.sum()  # [n,]
    # densities = densities.to(X.device)

    return densities


def density_based_cluster(
    X,  # [n_trajs, 4, 2], bbox of the vector
    N,  # int
    tol,  # float
    a_pos,  # [n_trajs, 1, 3], the complete segment
    sigama=0.01,
    cal_mean_heading=False,
):
    n_total = X.shape[0]
    ret_traj_list = []

    # 计算desities
    densities = compute_density_gpu(X, sigama)

    groups = []
    # for i in tqdm(range(N), desc="Clustering"):
    for i in range(N):
        if i == 0:
            choice_index = 0
        else:
            choice_index = torch.argmax(densities).item()

        x0 = X[choice_index]
        res_mask = (
            torch.norm(X - x0, dim=-1).mean(-1) > tol
        )  # 大于阈值的轮廓点保留下来，作为下一次筛选范围
        if cal_mean_heading:
            ret_traj = a_pos[~res_mask].mean(0, keepdim=True)
        else:
            ret_traj = a_pos[[choice_index]]

        groups.append(
            {
                "center": ret_traj,
                "motions": a_pos[~res_mask],
            }
        )
        X = X[res_mask]
        a_pos = a_pos[res_mask]
        densities = densities[res_mask]

        ret_traj_list.append(ret_traj)

        remain = X.shape[0] * 100.0 / n_total
        n_inside = (~res_mask).sum().item()
        print(f"{i=}, {remain=:.6f}%, {n_inside=}")

    visualize_trajectories_with_heading(groups)
    return torch.cat(ret_traj_list, dim=0)  # [N, 1, 3]


def visualize_trajectories_with_heading(
    groups, title="Motions with Heading", arrow_scale=0.02
):

    plt.figure(figsize=(18, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups) + 1))
    for k, gp in enumerate(groups):
        motions = gp["motions"]
        for motion in tqdm(motions):
            motion = motion.squeeze(0)
            x, y, heading = motion[0], motion[1], motion[2]
            plt.scatter(x, y, color=colors[k], alpha=0.4)
            dx = arrow_scale * math.cos(heading)
            dy = arrow_scale * math.sin(heading)
            plt.arrow(
                x - dx,
                y - dy,
                dx,
                dy,
                head_width=0.0,
                head_length=0.1,
                fc=colors[-1],
                ec=colors[-1],
                alpha=0.2,
            )

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True)
    plt.title(title)
    plt.axis("equal")
    plt.savefig("output/groups.png")
    plt.close()


def main(motion_file, save_path, num_cluster=2048):
    with open(motion_file, "rb") as f:
        data = pickle.load(f)
        data = torch.tensor(data)
        if len(data.shape) == 2:
            data = data.unsqueeze(1)
        data = torch.cat((torch.zeros(1, 1, 3), data), dim=0)

    width_length = torch.tensor([2.0, 4.8])
    width_length = width_length.unsqueeze(0)  # [1, 2]

    contour = cal_polygon_contour(
        pos=data[:, 0, :2], head=data[:, 0, 2], width_length=width_length
    )

    tol = 0.10
    # ret_traj = Kdisk_cluster(X=contour, N=num_cluster, tol=tol, a_pos=data)
    ret_traj = density_based_cluster(X=contour, N=16, tol=tol, a_pos=data)
    ret_traj[:, :, -1] = wrap_angle(ret_traj[:, :, -1])

    contour = cal_polygon_contour(
        pos=ret_traj[:, :, :2],  # [N, 1, 2]
        head=ret_traj[:, :, 2],  # [N, 1]
        width_length=width_length.unsqueeze(0),
    )

    save_res = {"traj": {}, "token_all": {}}
    save_res["traj"] = ret_traj.numpy()
    save_res["token_all"] = contour.numpy()

    with open(save_path, "wb") as f:
        pickle.dump(save_res, f)


if __name__ == "__main__":
    # main(sys.argv[1], sys.argv[2])
    main("output/motions.pkl", "output/clusters.pkl")
