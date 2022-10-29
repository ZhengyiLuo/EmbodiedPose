from smplx.lbs import vertices2joints
from torch.optim.lr_scheduler import StepLR
from scipy.ndimage import gaussian_filter1d
from copycat.khrylib.utils.transformation import (
    quaternion_slerp,
    quaternion_from_euler,
    euler_from_quaternion,
)
from tqdm import tqdm
from torch.autograd import Variable
from collections import defaultdict
from copycat.utils.transform_utils import (
    convert_aa_to_orth6d,
    convert_orth_6d_to_aa,
    vertizalize_smpl_root,
    rotation_matrix_to_angle_axis,
    rot6d_to_rotmat,
    convert_orth_6d_to_mat,
    angle_axis_to_rotation_matrix,
    angle_axis_to_quaternion,
)
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
from mujoco_py import load_model_from_path
from copycat.khrylib.utils import *
import joblib
from scipy.spatial.transform import Rotation as sRot

import glob
import os
import sys
import pdb
import os.path as osp
import argparse
# from copycat.smpllib.smpl_parser import SMPL_Parser
from copycat.data_process.grad_h36m_convert import smooth_smpl_quat_window

from embodiedpose.models.smpl_multi import SMPL as SMPL_Multi

sys.path.append(os.getcwd())


def remove_nan(np_array):
    nan_idx = np.nonzero(np.isnan(np_array[:, 0]))
    np_array[nan_idx, :] = np_array[nan_idx[0] + 1, :]
    return np_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="train")
    args = parser.parse_args()
    data = args.data
    device = (torch.device("cuda", index=0)
              if torch.cuda.is_available() else torch.device("cpu"))

    gpa_base_dir = "/hdd/zen/data/video_pose/GPA/"

    smpl_p = SMPL_Multi(
        "/hdd/zen/dev/copycat/Copycat/data/smpl",
        gender="neutral",
        joint_regressor_extra=
        "/hdd/zen/dev/ActMix/actmix/DataGen/MotionCapture/VIBE/data/vibe_data/J_regressor_extra.npy"
    )
    smpl_p = smpl_p.to(device)

    idx = 0
    # data_fit = joblib.load('/hdd/zen/data/video_pose/GPA/smpl_fits_test/data_split000.pkl')
    out_dir = osp.join(gpa_base_dir, "smpl_fits_grad")
    data_files = glob.glob("/hdd/zen/data/video_pose/GPA/smpl_fits_full/*.pkl")
    gpa_take1 = joblib.load(
        osp.join(gpa_base_dir, 'gpa_dataset_full_take1.pkl'))

    num_epochs = 5000
    lixel_order = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    subindex = np.array([
        True, True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, False, True, True, False, False, False, False, False,
        False
    ])
    # for k, data_item in tqdm(data_fit.items()):
    for data_file in data_files:
        data_out = {}
        data_item = joblib.load(data_file)
        k = data_file.split("/")[-1].split(".")[0]
        out_file_name = osp.join(out_dir, k + ".pkl")
        if osp.isfile(out_file_name):
            print("Already processed: ", out_file_name)
            continue
        data_item = {k: remove_nan(v) for k, v in data_item.items()}
        pose_aa, betas, trans = torch.from_numpy(data_item['pose']).to(device), \
            torch.from_numpy(np.mean(data_item['betas'], axis = 0)).to(device), torch.from_numpy(data_item['trans']).to(device)

        # with torch.no_grad():
        #     vertices, joints = smpl_p.get_joints_verts(pose_aa, th_betas=betas, th_trans=trans)
        #     length = joints.shape[0]

        length = pose_aa.shape[0]
        # pose_aa = smooth_smpl_quat_window(pose_aa, ratio = 0.5).reshape(-1, 72)

        gt_kps = torch.from_numpy(
            gpa_take1[k]['S24']).to(device).float()[:length, :, :3].clone()
        pose_aa_torch = torch.tensor(pose_aa).float().to(device)
        # trans = gaussian_filter1d(trans, 3, axis=0); trans = torch.from_numpy(trans).to(device).float()

        print("---" + k)
        pose_aa_torch_new = Variable(pose_aa_torch, requires_grad=True)
        trans_new = Variable(trans, requires_grad=True)
        shape_new = Variable(betas, requires_grad=True)

        optimizer_pose = torch.optim.SGD([pose_aa_torch_new], lr=1)
        optimizer_shape = torch.optim.SGD([shape_new], lr=100)
        optimizer_trans = torch.optim.SGD([trans_new], lr=5)

        # optimizer_mesh = torch.optim.Adadelta([pose_aa_torch_new, shape_new], lr=50)
        # optimizer_trans = torch.optim.Adadelta([trans_new], lr=5)
        # optimizer_regressor = torch.optim.Adadelta([J_regressor_new], lr=0.00001)

        # optimizer_mesh = torch.optim.Adagrad([pose_aa_torch_new, shape_new], lr=0.05)
        # optimizer_trans = torch.optim.Adagrad([trans_new], lr=0.01)

        # optimizer_mesh = torch.optim.AdamW([pose_aa_torch_new, shape_new], lr=0.001)
        # optimizer_trans = torch.optim.AdamW([trans_new], lr=0.0005)

        scheduler_pose = StepLR(optimizer_pose, step_size=1, gamma=0.9999)
        scheduler_shape = StepLR(optimizer_shape, step_size=1, gamma=0.9995)
        scheduler_trans = StepLR(optimizer_trans, step_size=1, gamma=0.9995)
        # scheduler_regressor = StepLR(optimizer_regressor, step_size=1, gamma=0.9995)

        for i in range(num_epochs):
            scheduler_pose.step()
            scheduler_trans.step()
            scheduler_shape.step()
            # scheduler_regressor.step()

            shapes = shape_new.repeat((pose_aa.shape[0], 1))
            shapes.retain_grad()

            vertices, joints = smpl_p.get_joints_verts(pose_aa_torch_new,
                                                       th_betas=shapes,
                                                       th_trans=trans_new)
            smpl_j3d_torch = joints

            # loss = torch.abs(smpl_j3d_torch - gt_kps).mean()
            loss = torch.norm(smpl_j3d_torch[:, subindex] -
                              gt_kps[:, subindex],
                              dim=2).mean()

            # print(loss_l2.item() * 1000, 'Epoch:', i,'LR:', scheduler_mesh.get_lr(), scheduler_trans.get_lr())
            if i % 100 == 0:
                print(
                    loss.item() * 1000,
                    "Epoch:",
                    i,
                    "LR:",
                    scheduler_pose.get_lr(),
                    scheduler_shape.get_lr(),
                    scheduler_trans.get_lr(),
                )
            optimizer_pose.zero_grad()
            optimizer_shape.zero_grad()
            optimizer_trans.zero_grad()
            # optimizer_regressor.zero_grad()
            loss.backward()
            optimizer_pose.step()
            optimizer_shape.step()
            optimizer_trans.step()
            # optimizer_regressor.step()
        shapes = shape_new.repeat((pose_aa.shape[0], 1))

        data_out = copy.deepcopy(gpa_take1[k])
        data_out["pose"] = pose_aa_torch_new.detach().cpu().numpy()
        data_out["shape"] = (shapes.detach().cpu().numpy())
        data_out["trans"] = trans_new.detach().cpu().numpy()
        joblib.dump(data_out, out_file_name)

        torch.cuda.empty_cache()
        import gc
        gc.collect()
    # # import ipdb; ipdb.set_trace()
    # joblib.dump(
    #     data_out,
    #     f'/hdd/zen/data/video_pose/GPA/smpl_fits/gpa_dataset_smplx_grad_fitted_test.pkl'
    # )
