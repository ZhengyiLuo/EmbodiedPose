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


def convert_to_h36m(joints):
    joints = joints / 1000
    j3ds_acc, j3d_h36m_acc = [], []
    for j3ds in joints:
        j3d_h36m = j3ds[j_ind]
        j3d_s24 = np.zeros([24, 4])
        j3d_s24[global_idx, :3] = j3d_h36m
        j3d_s24[global_idx, 3] = 1
        j3ds_acc.append(j3d_s24)
        j3d_h36m_acc.append(j3ds[j_ind_h36m_correct])
    j3ds_acc, j3d_h36m_acc = np.array(j3ds_acc), np.array(j3d_h36m_acc)
    return j3ds_acc, j3d_h36m_acc


ids = np.load('/hdd/zen/data/video_pose/GPA/c2gimgid.npy')
gpa_base_dir = "/hdd/zen/data/video_pose/GPA/"
cam_ids = np.load('/hdd/zen/data/video_pose/GPA/cam_ids_750k.npy')
# joints_3d = json.load(open('/hdd/zen/data/video_pose/GPA/xyz_gpa12_cntind_world_cams.json', "r"))
joints_3d_full = json.load(
    open('/hdd/zen/data/video_pose/GPA/gpa_videojson_full.json', "r"))

global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 13, 9, 10, 11, 8, 7,
              6]  # Fixing top of head
j_ind = [0, 29, 30, 31, 24, 25, 26, 3, 5, 6, 7, 17, 18, 19, 9, 10,
         11]  # Upper one spine joint
j_ind_h36m_correct = [
    0, 24, 25, 26, 29, 30, 31, 2, 5, 6, 7, 17, 18, 19, 9, 10, 11
]


def convert_to_h36m(joints):
    joints = joints / 1000
    j3ds_acc, j3d_h36m_acc = [], []
    for j3ds in joints:
        j3d_h36m = j3ds[j_ind]
        j3d_s24 = np.zeros([24, 4])
        j3d_s24[global_idx, :3] = j3d_h36m
        j3d_s24[global_idx, 3] = 1
        j3ds_acc.append(j3d_s24)
        j3d_h36m_acc.append(j3ds[j_ind_h36m_correct])
    j3ds_acc, j3d_h36m_acc = np.array(j3ds_acc), np.array(j3d_h36m_acc)
    return j3ds_acc, j3d_h36m_acc


gpa_dataset_full = {}
agg_keys = ['markers', 'joints_3d_mm', "joints_2d", "file_name"]

video_byframe_data = defaultdict(list)
for i in range(len(joints_3d_full['annotations'])):
    video_name = "_".join(
        (joints_3d_full['annotations'][i]['take_name'],
         joints_3d_full['annotations'][i]['subject_id'], cam_ids[i]))
    video_byframe_data[video_name].append(joints_3d_full['annotations'][i])

for video_key, frames_data in video_byframe_data.items():
    video_data = defaultdict(list)
    for k in [
            'camera_p0', 'camera_p1', 'camera_p2', 'camera_p3', 'subject_id',
            'take_name'
    ]:
        video_data[k] = frames_data[0][k]
    for frame_data in frames_data:
        for k in agg_keys:
            video_data[k].append(frame_data[k])

    for k in agg_keys:
        video_data[k] = np.array(video_data[k])
    j3ds_acc, j3d_h36m_acc = convert_to_h36m(
        np.array(video_data['joints_3d_mm']))
    video_data["S24"] = j3ds_acc
    video_data["S17"] = j3d_h36m_acc
    video_data["S34"] = np.array(video_data['joints_3d_mm']) / 1000
    gpa_dataset_full[video_key] = video_data

joblib.dump(gpa_dataset_full,
            osp.join(gpa_base_dir, 'gpa_dataset_full_take1.pkl'))
# gap_annot = joblib.load(osp.join(gpa_base_dir, 'gpa_dataset_full_take1.pkl'))
# import ipdb; ipdb.set_trace()