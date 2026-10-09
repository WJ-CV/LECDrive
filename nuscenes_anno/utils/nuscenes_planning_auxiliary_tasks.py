import os
import sys

sys.path.append("/evad-lab/evad-datasets/users/lg/users/maoyuchen/vla-r1")
from open_r1.dataloader.nuscenes_planning import NuScenesPlanningSupervisedDataset
from trl import ScriptArguments
from open_r1.configs import NuScenesDataConfig
from open_r1.dataloader.utils import safe_round, preprocess_meta_action_str
from PIL import Image


class NuScenesPlanningAuxiliaryTasksDataset(NuScenesPlanningSupervisedDataset):
    def __init__(
        self,
        data_path="data/nuScenes",
        script_args=ScriptArguments(
            dataset_name="data/nuScenes",
        ),
        data_args=NuScenesDataConfig(
            version="v1.0-mini",  # Replace with your dataset version
            split="val",  # Replace with your dataset split
            obs_len=4,
            fut_len=6,
            meta_dir="data/nuScenes",  # Replace with your meta directory
            rebuild=False,
            use_target=True,
            use_action=True,
            use_traj=True,
            use_nav=True,
            num_path_points=10,
            path_point_distance=1.0,
            decimal=2,
            query_prompt_type="json",
            sft_indices=[],
        ),
    ):
        super(NuScenesPlanningAuxiliaryTasksDataset, self).__init__(
            data_path=data_path, script_args=script_args, data_args=data_args
        )

    def __getitem__(self, idx):
        info = self._samples[idx]

        obs_lat_action = info["obs_lat_action"]  # lateral control
        obs_lon_action = info["obs_lon_action"]  # longitudinal control
        lat_action = info["lat_action"]  # lateral control
        lon_action = info["lon_action"]  # longitudinal control
        speed = info["speed"]  # speed of the vehicle
        obs_traj_list = [
            [safe_round(x, self.decimal), safe_round(y, self.decimal)]
            for (x, y) in info["obs_trajectory"]
        ]
        fut_traj_list = [
            [safe_round(x, self.decimal), safe_round(y, self.decimal)]
            for (x, y) in info["fut_trajectory"]
        ]

        images = [os.path.join(self.nusc.dataroot, frame) for frame in info["frames"]]

        return {
            "images": images,
            "speed": speed,
            "obs_lat_action": obs_lat_action,
            "obs_lon_action": obs_lon_action,
            "obs_trajectory": obs_traj_list,
            "fut_trajectory": fut_traj_list,
            "lat_action": lat_action,
            "lon_action": lon_action,
        }


# Example usage
if __name__ == "__main__":

    dataset = NuScenesPlanningAuxiliaryTasksDataset()
    sample = dataset[0]
    print(sample)
