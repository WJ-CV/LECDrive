import os
import cv2
import pickle
import shutil
from pathlib import Path
import numpy as np
################
bev_base_path = 'outputs_navsim/navsim_test/02_24_12_12_46_sota_resume_w_sam0.5_w_da1.5_1e-5_ep2-89.81+++++/visualization_bev/'
front_base_path = 'outputs_navsim/navsim_test/02_24_12_12_46_sota_resume_w_sam0.5_w_da1.5_1e-5_ep2-89.81+++++/visualization/'
################
navsim_logs_path='/e2e-data/evad-osc-datasets/datasets/navsim_v1.1/navsim_logs/test/'
base_path = '/e2e-data/evad-osc-datasets/datasets/navsim_v1.1/sensor_blobs/test/'

# navsim_logs = os.listdir(navsim_logs_path)
navsim_logs=[
    "2021.05.25.14.16.10_veh-35_01690_02183.pkl",
    # "2021.06.28.16.29.11_veh-38_03263_03766.pkl",
    # "2021.09.09.14.18.22_veh-48_00322_00895.pkl",
    # "2021.09.16.14.39.34_veh-42_00297_00935.pkl",
    # "2021.10.06.08.16.17_veh-52_00181_00574.pkl",
    # "2021.10.06.08.16.17_veh-52_00922_01296.pkl",
    # "2021.10.06.08.16.17_veh-52_01949_02501.pkl",
    # "2021.09.16.14.39.34_veh-42_00032_00186.pkl"
    ]
for pkl_log in navsim_logs:
    pkl_path = navsim_logs_path + pkl_log

    scenes_pkl = os.path.basename(pkl_path)
    scenes_name = os.path.splitext(scenes_pkl)[0]

    video_base_path = front_base_path.replace('visualization', 'visualization_video')
    target_path = os.path.join(video_base_path, scenes_name)
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # 视频设置
    video_path = target_path + "/merged_3v_bev_video.mp4"
    fps = 5
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = None


    with open(pkl_path, 'rb') as pkl:
        pkl_data = pickle.load(pkl)

    frame_idx = 0
    for data in pkl_data:
        token = data['token']
        frame_idx += 1

        cam_front = data['cams']['CAM_F0']['data_path']
        front_name = cam_front.split('/')[-1]
        img_name = front_name[:-4]
        vis_name = img_name + '_' + token + '.jpg'

        bev_img_path = Path(bev_base_path) / vis_name

        if not bev_img_path.exists():
            continue

        bev_new_name = str(frame_idx)+'_'+'bev'+'_'+ token + '.jpg'
        # shutil.copy(bev_img_path, target_path + bev_new_name)


        front_vis_path = front_base_path + vis_name
        front_new_name = str(frame_idx)+'_'+'front'+'_'+ token + '.jpg'
        # shutil.copy(front_vis_path, target_path + front_new_name)

        CAM_L0_path = base_path + data['cams']['CAM_L0']['data_path']
        Left_new_name = str(frame_idx)+'_'+'left'+'_'+ token + '.jpg'
        # shutil.copy(CAM_L0_path, target_path + Left_new_name)

        CAM_R0_path = base_path + data['cams']['CAM_R0']['data_path']
        Right_new_name = str(frame_idx)+'_'+'right'+'_'+ token + '.jpg'
        # shutil.copy(CAM_R0_path, target_path + Right_new_name)


        img_L = cv2.imread(CAM_L0_path)
        img_F = cv2.imread(front_vis_path)
        img_R = cv2.imread(CAM_R0_path)

        overlap = 480 
        h, w_L, _ = img_L.shape
        h, w_F, _ = img_F.shape
        h, w_R, _ = img_R.shape

        img_L_crop = img_L[:, :w_L - overlap]   # L图裁掉右边
        img_R_crop = img_R[:, overlap:]         # R图裁掉左边

        merged = np.hstack((img_L_crop, img_F, img_R_crop))
        merged_resized = cv2.resize(merged, (800, 250), interpolation=cv2.INTER_AREA)
        merged_name = str(frame_idx)+'_'+'merged'+'_'+ token + '.jpg'
        # cv2.imwrite(target_path + merged_name, merged_resized)


        img_top = merged_resized
        img_bottom = cv2.imread(bev_img_path)

        overlap = 110  # 重叠高度
        h1, w1, _ = img_top.shape
        h2, w2, _ = img_bottom.shape

        img_top_crop = img_top[:, :]
        img_bottom_crop = img_bottom[overlap:-100, 112:-88]

        merged_bev = np.vstack((img_top_crop, img_bottom_crop))
        merged_bev_name = '/' + str(frame_idx)+'_'+'merged_bev'+'_'+ token + '.jpg'
        cv2.imwrite(target_path + merged_bev_name, merged_bev)

        if video is None:
            h, w, _ = merged_bev.shape
            video = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        # 写入视频
        video.write(merged_bev)

        print(f"frame {frame_idx} added")

    # 释放视频
    if video:
        video.release()

    print("Video saved to:", video_path)