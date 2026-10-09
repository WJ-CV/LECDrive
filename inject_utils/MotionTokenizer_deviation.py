import pickle
from typing import Tuple, Union
from torch import nn
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import re
import ast

class MotionTokenizer(nn.Module):
    """
    The default length/width of the ego car is 4.8m/2m
    """

    def __init__(self, vocab_path: str, horizon: int=8, p_noise=0., topk=1):
        super().__init__()
        self.register_vocabulary(vocab_path=vocab_path)
        self.ego_car_size = {"car_length": 4.8, "car_width": 2}
        self.horizon = horizon
        self.p_noise = p_noise
        self.topk = topk
        assert topk >= 1 and p_noise <= 1 and p_noise >= 0

    def register_vocabulary(self, vocab_path):
        """load the motion vocabulary"""
        with open(vocab_path, "rb") as ifp:
            # motion_vocab: vocab_size * 4 * 2
            infos = pickle.load(ifp)
        # corners = infos["token"]["veh"].astype(np.float32)  # 2048 x 4 x 2
        # !! use float64 as default, change the module precision with traing framework
        # corners = infos["token"]["veh"].astype(np.float32)  # 2048 x 4 x 2
        # corners = infos["token"]["veh"]  # 2048 x 4 x 2
        if 'contour' in infos:
            corners = infos["contour"].squeeze(1)  # 2048 x 4 x 2
        else:
            corners = infos["token"]["veh"]  # 2048 x 4 x 2
        dxy = (
            corners[:, 0] - corners[:, 3]
        )  # clockwise: front-left, front-right, back-right, back-left
        orients = np.arctan2(dxy[:, 1], dxy[:, 0])
        coord_yaw = np.hstack((corners.mean(1), orients[:, None]))  # (x, y, yaw)

        self.register_buffer("vocab_contour", torch.from_numpy(corners))
        self.register_buffer("vocab_motions", torch.from_numpy(coord_yaw))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab_contour)

    @property
    def dtype(self) -> torch.dtype:
        return self.vocab_motions.dtype

    def tokenize(self, trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """trajectory: N x T x 3 (x, y, heading)"""
        assert trajectory.size(1) == self.horizon

        # matching strategy
        # tokens = self.catk(trajectory, self.motion_vocab)
        # tokens = self.topk(trajectory, self.motion_vocab)
        tokens, rollouts = self.gready(trajectory, self.vocab_contour, self.vocab_motions, self.p_noise, self.topk)
        return tokens, rollouts

    def detokenize(self, tokens: torch.Tensor, init_p=None) -> torch.Tensor:
        """
        tokens: (B, T)
        returns:
            trajectory: (B, T, 3)
        """
        B, T, dtype = tokens.size(0), tokens.size(1), self.dtype

        if init_p is None:
            curr_pos = torch.zeros((B, 2), device=tokens.device, dtype=dtype)
            curr_yaw = torch.zeros((B), device=tokens.device, dtype=dtype)
        else:
            curr_pos = init_p[:, :2].clone().to(tokens.device).to(dtype)
            curr_yaw = init_p[:, 2].clone().to(tokens.device).to(dtype)
        rollouts = torch.zeros((B, T, 3), device=tokens.device, dtype=dtype)

        for t in range(T):
            indices = tokens[:, t]
            motions = self.vocab_motions[indices]

            # Convert the seletected motion vectors to the global CRS
            inverse_rot_matrix = MotionTokenizer.rotation_matrix(-curr_yaw)  # (B, 2, 2)
            curr_pos += torch.einsum("nd,ndk->nk", motions[:, :2], inverse_rot_matrix)
            curr_yaw += motions[:, -1]

            rollouts[:, t, :2] = curr_pos
            rollouts[:, t, -1] = curr_yaw

        return rollouts

    def get_contour(self, tokens: Union[torch.Tensor, None] = None) -> torch.Tensor:
        if tokens is None:
            return self.vocab_contour.to(dtype=torch.float32, device=tokens.device)
        else:
            return self.vocab_contour[tokens].to(dtype=torch.float32, device=tokens.device)

    def get_motion(self, tokens: Union[torch.Tensor, None] = None) -> torch.Tensor:
        if tokens is None:
            return self.vocab_motions.to(dtype=torch.float32, device=tokens.device)
        else:
            return self.vocab_motions[tokens].to(dtype=torch.float32, device=tokens.device)

    @staticmethod
    def _check_heading_diff(trajectory):
        """Truncate excessive orientation differences
        # check the validate heading diff between timesteps
        trajectory: N x T x 3
        """
        headings = trajectory[..., -1]
        hdiff = torch.diff(headings, dim=1, prepend=torch.zeros_like(headings[:, :1]))
        hmask = torch.abs(hdiff) < 1.5
        return hmask

    @staticmethod
    def gready(trajectory: torch.Tensor, contours: torch.Tensor, motions: torch.Tensor, p_noise=0., topk=1) -> Tuple[torch.Tensor, torch.Tensor]:
        """The global CRS is built on frame(0)"""
        B, T, _ = trajectory.shape
        tokens = torch.zeros((B, T), dtype=torch.long, device=trajectory.device)

        if T < 8:
            curr_pos = trajectory[:, 0, :2].clone()
            curr_yaw = trajectory[:, 0, -1].clone()
        else:
            curr_pos = torch.zeros_like(trajectory[:, 0, :2])
            curr_yaw = torch.zeros_like(trajectory[:, 0, -1])
        rollouts = torch.zeros_like(trajectory)

        for t in range(T):
            next_pos = trajectory[:, t, :2]  # B x 2
            next_yaw = trajectory[:, t, -1]  # B

            pos_delta = next_pos - curr_pos
            yaw_delta = wrap_angle(next_yaw - curr_yaw)

            # Convert the motion delta vectors to the local@t CRS
            # Multiplied on the right, rotate coordinate reference system with yaw
            rot_matrix = MotionTokenizer.rotation_matrix(curr_yaw)
            local_pos_t = torch.einsum("nd,ndk->nk", pos_delta, rot_matrix)
            local_pose_t = torch.cat((local_pos_t, yaw_delta[:, None]), dim=-1)

            # distances = MotionTokenizer.distance_pose(local_pos_t, vocabulary)
            kwargs = {"car_length": 4.8, "car_width": 2}
            distances = MotionTokenizer.distance_contour(
                MotionTokenizer.pose2contour(local_pose_t, **kwargs),
                contours,
            )

            if np.random.rand() < p_noise:
                # probs = torch.softmax(distances, dim=-1)
                # indices = torch.multinomial(probs, 1)
                # mot_t = motions[indices]  # B x 3

                row = torch.arange(B)
                col = torch.randint(0, topk, (B,), dtype=torch.long)
                indices = torch.argsort(distances, dim=-1)[row, col]
                mot_t = motions[indices]  # B x 3
            else:
                indices = torch.argmin(distances, dim=-1)
                mot_t = motions[indices]  # B x 3

            # Convert the seletected motion vectors to the global CRS
            inverse_rot_matrix = MotionTokenizer.rotation_matrix(-curr_yaw)  # (B, 2, 2)
            curr_pos += torch.einsum("nd,ndk->nk", mot_t[:, :2], inverse_rot_matrix)
            curr_yaw += mot_t[:, -1]
            # print(f"   {pos_delta=}")
            # print(f"   {yaw_delta=}")
            # print(f"{local_pose_t=}")
            # print(f"   {mot_t[0]=}")

            rollouts[:, t, :2] = curr_pos
            rollouts[:, t, -1] = curr_yaw
            tokens[:, t] = indices

        # print("  ".join(["\n====", "Finish", "===="]))
        return tokens, rollouts

    @staticmethod
    def rotation_matrix(theta: torch.Tensor) -> torch.Tensor:
        """When the matrix is multiplied on the left, it rotates the vector;
        when the matrix is multiplied on the right, it rotates the coordinate system.

        Args:
            theta: (B,)
        Return:
            matrix: (B, 2, 2)
        """
        assert theta.dim() == 1, f"Wrong shape: {theta.size()}"
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # rot_matrix = torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta])
        rot_matrix = torch.stack(
            [
                torch.stack([cos_theta, -sin_theta], dim=-1),  # (B, 2)
                torch.stack([sin_theta, cos_theta], dim=-1),  # (B, 2)
            ],
            dim=-2,
        )  # shape (B, 2, 2)
        return rot_matrix

    @staticmethod
    def distance_pose(pose_t: torch.Tensor, vocabulary: torch.Tensor) -> torch.Tensor:
        # NOTE: The performance of this version is not good enough.
        """
        args:
            pose_t: B x 3
            vocabulary: N x 3
        return:
            distance: B x N
        """
        position_diff = torch.norm(
            pose_t[:, None, :2] - vocabulary[None, :, :2], dim=-1
        )
        heading_diff = pose_t[:, None, -1] - vocabulary[None, :, -1]  # B x N
        heading_diff = torch.abs(
            torch.atan2(torch.sin(heading_diff), torch.cos(heading_diff))
        )
        return position_diff + 4.8 * 2 * heading_diff

    @staticmethod
    def distance_contour(
        pose_t_contour: torch.Tensor, vocabulary_contour: torch.Tensor
    ):
        """
        pose_t: B x 4 x 2
        vocabulary: N x 4 x 2
        """
        return torch.norm(
            pose_t_contour[:, None, :, :] - vocabulary_contour[None, :, :, :], dim=-1
        ).sum(-1)

    @staticmethod
    def pose2contour(pose: torch.Tensor, car_length, car_width) -> torch.Tensor:
        """convert pose to contour

        Args:
            pose (torch.Tensor): B x 3(X, Y, Yaw)

        Returns:
            contour (torch.Tensor): B x 4 x 2
        """
        # clockwise: front-left, front-right, back-right, back-left
        x, y, yaw = pose[:, 0], pose[:, 1], pose[:, 2]
        width, length = car_width, car_length

        half_cos = 0.5 * yaw.cos()  # [n_agent, n_step, n_target]
        half_sin = 0.5 * yaw.sin()  # [n_agent, n_step, n_target]
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

    @staticmethod
    def contour2pose(corners: torch.Tensor) -> torch.Tensor:
        # clockwise: front-left, front-right, back-right, back-left
        dxy = corners[:, 0] - corners[:, 3]
        orients = torch.arctan2(dxy[:, 1], dxy[:, 0])
        pose = torch.cat((corners.mean(1), orients[:, None]), dim=-1)
        return pose


def wrap_angle(radians):
    return torch.atan2(torch.sin(radians), torch.cos(radians))


def plot_trajectory(trajectory, rollout, save_path):
    # Plot trajectory
    plt.figure(figsize=(8, 6))

    # Extract x, y, and yaw
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    yaw = trajectory[:, 2]

    # Compute unit vectors for heading arrows
    u = np.cos(yaw)
    v = np.sin(yaw)

    plt.plot(x, y, "o-", label="Trajectory")
    plt.quiver(
        x,
        y,
        u,
        v,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="r",
        width=0.005,
        label="Traj Yaw",
    )

    if rollout is not None:
        # Extract x, y, and yaw
        rx = rollout[:, 0]
        ry = rollout[:, 1]
        ryaw = rollout[:, 2]

        # Compute unit vectors for heading arrows
        ru = np.cos(ryaw)
        rv = np.sin(ryaw)

        plt.plot(rx, ry, "x-", label="Rollout")
        plt.quiver(
            rx,
            ry,
            ru,
            rv,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="g",
            width=0.005,
            label="Roll Yaw",
        )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Trajectory with Heading Arrows")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig(save_path)
    plt.close()


def load_catk_examples(pkl_path) -> torch.Tensor:
    with open(pkl_path, "rb") as ifp:
        dat = pickle.load(ifp)
    # print(f"{dat.keys()=}")
    # vehicle mask
    mask = torch.logical_and(dat["type"] == 0, dat["ego_mask"])
    # trajectory = torch.cat([dat['gt_pos_raw'][mask], dat['gt_head_raw'][mask][:, :, None]], dim=-1).cpu()
    pos = dat["gt_pos_raw"][mask]  # 10, 18, 2
    yaw = dat["gt_head_raw"][mask]  # 10, 18
    pos = normalize_trajectory(pos, yaw)
    yaw = wrap_angle(yaw - yaw[:, :1])

    # move more than 5 meters
    # mask = torch.norm(pos[:, -1] - pos[:, 0], dim=-1) > 5
    # pos = pos[mask]
    # yaw = yaw[mask]

    trajectory = torch.cat([pos, yaw[:, :, None]], dim=-1).cpu().double()
    return trajectory


def mock_example(radius=16, horizon=9) -> torch.Tensor:
    angles = np.linspace(0, np.pi / 12 * 5, horizon)  # 0->75

    x = radius * np.sin(angles)
    y = radius * (1 - np.cos(angles))
    trajectory = np.column_stack((x, y))
    trajectory = np.hstack((trajectory, angles[:, None]))[1:, :]

    trajectory = torch.from_numpy(np.array([trajectory]))
    return trajectory


def normalize_trajectory(trajectory, yaw_angles):
    """
    Normalize the trajectory based on the position and yaw angle at time 0.

    Args:
    - trajectory: Tensor of shape (N, T, 2) representing the trajectories.
    - yaw_angles: Tensor of shape (N, T) representing the yaw angles.

    Returns:
    - normalized_trajectory: Tensor of shape (N, T, 2) representing the normalized trajectories.
    """
    # Extract the position and yaw angle at time 0
    initial_positions = trajectory[:, 0, :]  # Shape: (N, 2)
    initial_yaw_angles = yaw_angles[:, 0]  # Shape: (N)

    # Convert yaw angles to rotation matrices
    cos_yaw = torch.cos(initial_yaw_angles)
    sin_yaw = torch.sin(initial_yaw_angles)
    rotation_matrices = torch.stack(
        [
            torch.stack([cos_yaw, -sin_yaw], dim=-1),
            torch.stack([sin_yaw, cos_yaw], dim=-1),
        ],
        dim=-2,
    )  # Shape: (N, 2, 2)

    # Subtract the initial position from the trajectory
    centered_trajectory = trajectory - initial_positions.unsqueeze(
        1
    )  # Shape: (N, T, 2)

    # Apply rotation to align with the global coordinate system
    normalized_trajectory = torch.einsum(
        "ntd,ndk->ntk", centered_trajectory, rotation_matrices
    )  # Shape: (N, T, 2)

    return normalized_trajectory

def format_tensor_2dec(t):
    return "[" + ", ".join(f"{v.item():.2f}" for v in t) + "]"

def convert_tokens_to_motion_format(tokens, deviation):
    """将每个 token 转换为 <MTxxx> 格式"""
    return [f"<MT{tokens[i].item()}>+{format_tensor_2dec(deviation[i, :2])}" for i in range(len(tokens))]

pre_prompt="You are an autonomous driving trajectory prediction system. Predict the ego vehicle's future trajectory for the next 4 seconds (8 steps, 0.5s per step) based on: \n1. Visual perception from front-left, front, and front-right camera views \n2. "
Output_requirements = "Output requirements:\n- Predict 8 future motion steps (t+1 to t+8) - Each step must be formatted as: <MTxxx> + [dx, dy] - Use [...] to encapsulate the sequence - Deviation values [dx, dy] should be rounded to two decimal places - Output only the motion sequence, without any extra text"

def motion_tokenizer_his(traj_full, user_prompt, topk=2):
    user_content = user_prompt["text"]
    user_content = user_content.split("Output requirements:")[0] + Output_requirements
    post_content = user_content.split("\n\n3.")[1]


    trajectory = torch.from_numpy(np.array([traj_full]))
    trajectory_his = trajectory[:, :4, :]
    mt = MotionTokenizer(
        vocab_path='inject_utils/motion_vocab.pkl', horizon=trajectory_his.size(1), p_noise=1.0, topk=topk
    )
    tokens, rollouts = mt.tokenize(trajectory_his)

    deviation = trajectory_his - rollouts
    deviation = torch.round(deviation, decimals=2)
    # ###
    # deroll = mt.detokenize(tokens, init_p=trajectory_his[:, 0, :])
    # assert torch.isclose(deroll, rollouts).all()
    # print(f"Displacement: {torch.norm(trajectory_his[:, :, :2] - deroll[:, :, :2], dim=-1).mean(dim=-1).mean()}.")

    rm_tokens = tokens[:, 1:]
    deviation = deviation[:, 1:]
    his_motion_tokens = convert_tokens_to_motion_format(rm_tokens[0], deviation[0])
    fill_content = f"Historical motion context (last 3 steps): - t-2: {his_motion_tokens[0]} - t-1: {his_motion_tokens[1]} - t-0: {his_motion_tokens[2]} Each <MTxxx> is a motion token representing a discrete motion prototype. The 2D deviation [dx, dy] is a continuous offset applied to the motion token"

    new_content = pre_prompt + fill_content + "\n3." + post_content
    user_prompt["text"] = new_content
    return user_prompt

def motion_tokenizer(traj_content, topk=5):
    trajectory = json.loads(traj_content)
    trajectory = torch.from_numpy(np.array([trajectory]))

    valid_mask = MotionTokenizer._check_heading_diff(trajectory)
    if not valid_mask.all():
        mask = valid_mask.all(dim=-1)
        print(f"before: {trajectory.shape=}")
        trajectory = trajectory[mask]
        print(f"after: {trajectory.shape=}")
        # assert valid_mask.all(), f"Invalid: {trajectory[valid_mask]}"

    mt = MotionTokenizer(
        vocab_path='inject_utils/motion_vocab.pkl', horizon=trajectory.size(1), p_noise=1.0, topk=topk
    )
    tokens, rollouts = mt.tokenize(trajectory)

    deviation = trajectory - rollouts
    deviation = torch.round(deviation, decimals=2)

    motion_tokens = convert_tokens_to_motion_format(tokens[0], deviation[0])
    
    return str(str(motion_tokens).replace("'", ""))

def motion_detokenizer(text):
    matches = re.findall(r'<MT(\d+)>', text)
    numbers = list(map(int, matches))
    re_tokens = torch.tensor([numbers])

    mt = MotionTokenizer(
        vocab_path='inject_utils/motion_vocab.pkl', horizon=re_tokens.size(1)
    )

    deroll = mt.detokenize(re_tokens)
    re_traj = deroll[0, :, :-1]
    re_traj = re_traj.round(decimals=2).tolist()

    deviations = re.findall(r'\[[-\d\.]+,\s*[-\d\.]+\]', text)
    if not deviations:
        return re_traj
    else:
        deviations = '[' + ', '.join(deviations) + ']'
        offset = ast.literal_eval(deviations)
        
        rollout_traj = np.array(re_traj) + np.array(offset)
        return rollout_traj.round(decimals=2).tolist(), np.array(re_traj).round(decimals=2).tolist()

if __name__ == "__main__":
    # trajectory = load_catk_examples("dataset/catk_test_example.pkl")
    # trajectory = mock_example(16, 9)
    json_path = 'cache_files/navsim_cache/navsim_test_4s_3v_1f_yaw_system_user_prompt.json'
    with open(json_path, 'r') as js:
        json_data = json.load(js)
    for data in json_data:
        trajectory = json.loads(data["messages"][-1]["content"])
        trajectory = torch.from_numpy(np.array([trajectory]))

        # check the validate heading diff between timesteps
        valid_mask = MotionTokenizer._check_heading_diff(trajectory)
        if not valid_mask.all():
            mask = valid_mask.all(dim=-1)
            print(f"before: {trajectory.shape=}")
            trajectory = trajectory[mask]
            print(f"after: {trajectory.shape=}")
            # assert valid_mask.all(), f"Invalid: {trajectory[valid_mask]}"

        for index in range(len(trajectory)):
            plot_trajectory(trajectory[index, ...], None, f"motionToken_output/{index:05d}_traj.png")

        mt = MotionTokenizer(
            vocab_path='inject_utils/motion_vocab.pkl', horizon=trajectory.size(1), p_noise=1., topk=5
        )
        tokens, rollouts = mt.tokenize(trajectory)

        motion_tokens = convert_tokens_to_motion_format(tokens[0])

        data["messages"][-1]["content"] = str(str(motion_tokens).replace("'", ""))

        re_tokens = torch.tensor([[int(token.strip('<MT>')) for token in motion_tokens]])

        deroll = mt.detokenize(re_tokens)
        assert torch.isclose(deroll, rollouts).all()
        print(f"Displacement: {torch.norm(trajectory[:, :, :2] - deroll[:, :, :2], dim=-1).mean(dim=-1).mean()}.")

        print(f"{trajectory.shape=}")
        print(f"{trajectory=}")
        
        print(f"{re_tokens.shape=}")
        print(f"{rollouts=}")
        print(f"{re_tokens=}")

        for index in range(len(trajectory)):
            plot_trajectory(
                trajectory[index, ...],
                rollouts[index, ...],
                f"motionToken_output/{index:05d}_token.png",
            )
    with open("navsim_test_4s_3v_1f_motiontoken_system_user_prompt.json", 'w') as js:
        json.dump(json_data, js, indent=4)