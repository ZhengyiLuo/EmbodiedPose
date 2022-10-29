import shutil
import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import joblib

from tqdm import tqdm
import pickle as pk
import joblib
from scipy.spatial.transform import Rotation as sRot
from collections import defaultdict
import json
from tqdm import tqdm

from copycat.utils.image_utils import read_img_dir, write_frames_to_video

# prox_base = "/hdd/zen/data/video_pose/prox/qualitative/"
# for prox_dir in glob.glob(osp.join(prox_base, "recordings", "*")):
#     video_name = prox_dir.split('/')[-1]
#     img_list = sorted(glob.glob(osp.join(prox_dir, "Color", "*")))
#     frames = read_img_list(img_list)
#     frames = [f[:, ::-1, :] for f in frames]
#     out_name = osp.join(prox_base, "videos_full", f"{video_name}.mp4")
#     write_frames_to_video(frames, out_file_name=out_name)
#     print(f"done {out_name}")

#     del frames
#     import gc
#     gc.collect()


# hrnet_base = "/hdd/zen/data/video_pose/prox/hrnet_res"
# for prox_dir in glob.glob(osp.join(hrnet_base, "images", "*")):
#     video_name = prox_dir.split('/')[-1]
#     if video_name == "MPH1Library_00145_01":
#         img_list = sorted(glob.glob(osp.join(prox_dir, "*")))
#         import ipdb
#         ipdb.set_trace()
#         frames = read_img_list(img_list)
#         frames = [f[:, ::-1, :] for f in frames]
#         out_name = osp.join(hrnet_base, "videos", f"{video_name}.mp4")

#         write_frames_to_video(frames, out_file_name=out_name)
#         print(f"done {out_name}")

#         del frames
#         import gc
#         gc.collect()

# import cv2
# TRIM = 90
# prox_base = "/hdd/zen/data/video_pose/prox/qualitative/"
# for prox_dir in tqdm(glob.glob(osp.join(prox_base, "recordings", "*"))):
#     video_name = prox_dir.split('/')[-1]
#     vid = cv2.VideoCapture(
#         f"/hdd/zen/data/video_pose/prox/qualitative/videos_full/{video_name}.mp4"
#     )
#     image_frames = []
#     while (vid.isOpened()):
#         ret, frame = vid.read()
#         if ret == True:
#             image_frames.append(frame)
#         else:
#             break
#     image_frames = np.array(image_frames)[91:-90, :, ]
#     write_frames_to_video(image_frames, out_file_name=f"/hdd/zen/data/video_pose/prox/qualitative/videos_chop/{video_name}.mp4")
#     print(f"done {video_name}")

#     del image_frames
#     import gc
#     gc.collect()


# import cv2
# TRIM = 90
# prox_base = "/hdd/zen/data/video_pose/prox/qualitative/"
# for prox_dir in tqdm(glob.glob(osp.join(prox_base, "videos_chop", "*"))):
#     cmd = f"ffmpeg -i {prox_dir} -vf hflip -c:a copy {prox_dir}"
#     os.system(cmd)

# import cv2

# TRIM = 90
# prox_base = "/hdd/zen/data/video_pose/prox/dekr_res/"
# for prox_dir in tqdm(glob.glob(osp.join(prox_base, "videos", "*"))):
#     video_name = prox_dir.split('/')[-1].split(".")[0]
#     vid = cv2.VideoCapture(prox_dir)
#     print(prox_dir)
#     try:
#         image_frames = []
#         while (vid.isOpened()):
#             ret, frame = vid.read()
#             if ret == True:
#                 image_frames.append(frame)
#             else:
#                 break
#         image_frames = np.array(image_frames)[91:-90, :, ]
#         write_frames_to_video(
#             image_frames,
#             out_file_name=
#             f"/hdd/zen/data/video_pose/prox/dekr_res/videos_chop/{video_name}.mp4"
#         )
#         print(f"done {video_name}")

#         del image_frames
#         import gc
#         gc.collect()
#     except:
#         print("fail at", prox_dir)

import cv2
TRIM = 90
data_base = "/hdd/zen/data/video_pose/prox/qualitative/renderings/sceneplus/tcn_voxel_4_1/prox"
for video_full_name in tqdm(glob.glob(osp.join(data_base, "prox", "*"))):
    vid = cv2.VideoCapture(video_full_name)
    video_file_name = video_full_name.split("/")[-1]
    try:
        image_frames = []
        while (vid.isOpened()):
            ret, frame = vid.read()
            if ret == True:
                image_frames.append(frame)
            else:
                break
        image_frames = np.array(image_frames)[90:, :, ]
        write_frames_to_video(
            image_frames,
            out_file_name=
            osp.join(data_base, "prox_chop", video_file_name)
        )
        print(f"done {video_full_name}")

        del image_frames
        import gc
        gc.collect()
    except:
        print("fail at", video_full_name)


# import cv2
# data_base = "temp"
# os.makedirs(osp.join(data_base,  "tcn_voxel_1") , exist_ok=True)
# for images_dir in tqdm(glob.glob(osp.join(data_base, "tcn_voxel_1", "*"))):
    
#     image_frames = read_img_dir(images_dir)
#     if len(image_frames) == 0:
#         continue
#     print(f"Processing {images_dir}")
#     image_frames = np.array(image_frames)[90:, :, ]
#     write_frames_to_video(
#         image_frames,
#         out_file_name=
#         osp.join(data_base, "videos", images_dir.split("/")[-1] + ".mp4")
#     )
    

#     del image_frames
#     import gc
#     gc.collect()