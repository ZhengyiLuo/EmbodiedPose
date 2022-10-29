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
import joblib
from embodiedpose.models.humor.utils.velocities import estimate_velocities
from copycat.smpllib.np_smpl_humanoid_batch import Humanoid_Batch
from embodiedpose.models.humor.utils.humor_mujoco import reorder_joints_to_humor, MUJOCO_2_SMPL

humanoid_batch = Humanoid_Batch(data_dir="/hdd/zen/data/SMPL/smpl_models/")

h36m_grad = joblib.load('/hdd/zen/data/video_pose/h36m/data_fit/h36m_test_30_fitted_grad_full.p')
# h36m_grad = joblib.load('/hdd/zen/data/video_pose/h36m/data_fit/h36m_train_30_fitted_grad_full.p')
h36m_base = "/hdd/zen/data/video_pose/h36m/raw_data"
new_dict = {}
for k, v in tqdm(h36m_grad.items()):
    B = v['part'].shape[0]
    j2ds = v['part'].copy()
    j3ds = v['S'].copy()

    j2ds_25 = np.zeros([B, 25, 3])
    j3ds_14 = np.zeros([B, 14, 3])

    j2ds_25[:, 1] = j2ds[:, [11, 14], :].mean(axis=1)
    j2ds_25[:, [2, 3, 4, 5, 6, 7]] = j2ds[:, [14, 15, 16, 11, 12, 13], :]
    j2ds_25[:, [8, 9, 10, 11]] = j2ds[:, [0, 4, 5, 6], :]
    j2ds_25[:, [12, 13, 14]] = j2ds[:, [1, 2, 3], :]

    full_R = np.array(v['cam']['R'])
    full_t = np.array(v['cam']['t']) / 1000

    j3ds = (j3ds - full_t) @ full_R
    j3ds_14[:, 0] = j3ds[:, [11, 14], :].mean(axis=1)
    j3ds_14[:, [1, 2, 3, 4, 5, 6]] = j3ds[:, [14, 15, 16, 11, 12, 13], :]
    j3ds_14[:, [7, 8, 9, 10]] = j3ds[:, [0, 4, 5, 6], :]
    j3ds_14[:, [11, 12, 13]] = j3ds[:, [1, 2, 3], :]

    K = np.zeros([3, 3])
    K[:2, :2] = np.diag(v['cam']['f'])
    K[:2, 2] = v['cam']['c']
    K[2, 2] = 1

    beta = v['shape'].copy()
    pose_aa_np = v['pose'].copy()
    trans_np = v['trans'].copy()
    gender = "neutral"

    humanoid_batch.update_model(torch.from_numpy(beta[0:1]), torch.tensor([0]))
    trans_np = (trans_np - full_t) @ full_R - humanoid_batch._offsets[:, 0]
    pose_aa_np[:, :3] = sRot.from_matrix(
        np.matmul(full_R.T,
                  sRot.from_rotvec(
                      pose_aa_np[:, :3]).as_matrix())).as_rotvec()

    pose_mat = sRot.from_rotvec(pose_aa_np.reshape(B * 24,
                                                   3)).as_matrix().reshape(
                                                       B, 24, 3, 3)
    pose_body = pose_mat[:, 1:22]
    root_orient = pose_mat[:, 0:1].squeeze()

    input_vecs = np.concatenate([trans_np, pose_aa_np], axis=1)[None, ]
    wbpos = humanoid_batch.fk_batch_grad(input_vecs)[..., MUJOCO_2_SMPL, :]
    joints = wbpos.squeeze()[:, :22].reshape(-1, 66)

    trans_vel, joints_vel, root_orient_vel = estimate_velocities(
        torch.from_numpy(trans_np[None]),
        torch.from_numpy(root_orient[None]),
        torch.from_numpy(joints[None, ]),
        30,
        aa_to_mat=False)
    beta_16 = np.concatenate([beta, np.zeros([B, 6])], axis=1)

    new_dict[k] = {
        "joints2d": j2ds_25,
        "joints": joints,
        "joints_gt": j3ds_14,
        "seq_name": k,
        "pose_aa": pose_aa_np,
        "pose_6d": pose_mat[:, :, :, :2].reshape(B, 24, 6),
        "pose_body": pose_body,
        "root_orient": root_orient,
        "trans": trans_np,
        'betas': beta_16,
        "gender": "neutral",
        "seq_name": k,
        "trans_vel": trans_vel.squeeze().numpy(),
        "joints_vel": joints_vel.squeeze().numpy(),
        "root_orient_vel": root_orient_vel.squeeze().numpy(),
        "cam": {
            "scene_name": None,
            "img_w": 1000,
            "img_h": 1000,
            "full_R": full_R,
            "full_t": full_t,
            "K": K
        }
    }

joblib.dump(
    new_dict,
    "/hdd/zen/data/video_pose/h36m/data_fit/h36m_test_30_fk.p")

# joblib.dump(
#     new_dict,
#     "/hdd/zen/data/video_pose/h36m/data_fit/h36m_train_30_grad_full_2d.p")

# np.random.seed(0)
# keys = np.random.choice(list(new_dict.keys()), 30, replace=False)
# joblib.dump(
#     {k: new_dict[k]
#      for k in keys},
#     "/hdd/zen/data/video_pose/h36m/data_fit/h36m_test_30_fk_valid.p")
