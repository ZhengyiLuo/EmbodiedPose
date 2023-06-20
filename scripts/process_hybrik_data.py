import argparse
import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import numpy as np
import pickle as pk
from scipy.spatial.transform import Rotation as sRot
import matplotlib.pyplot as plt
import joblib
from uhc.utils.math_utils import smpl_op_to_op
from embodiedpose.models.humor.body_model.utils import smpl_to_openpose
from embodiedpose.models.humor.utils.humor_mujoco import SMPL_2_OP, OP_14_to_OP_12
from uhc.smpllib.smpl_parser import SMPL_Parser, SMPLH_BONE_ORDER_NAMES, SMPLH_Parser
import torch

def xyxy2xywh(bbox): # from HybrIK
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default='sample_data/res_wild/res.pk')
    parser.add_argument("--output", default='sample_data/wild_processed.pkl')
    args = parser.parse_args()

    data_dir = "data/smpl"
    smpl_parser_n = SMPL_Parser(model_path=data_dir, gender="neutral")
    smpl_parser_m = SMPL_Parser(model_path=data_dir, gender="male")
    smpl_parser_f = SMPL_Parser(model_path=data_dir, gender="female")

    smpl2op_map = smpl_to_openpose("smpl",
                                    use_hands=False,
                                    use_face=False,
                                    use_face_contour=False,
                                    openpose_format='coco25')
    smpl_2op_submap = smpl2op_map[smpl2op_map < 22]

    res_data = pk.load(open(args.input, "rb"))

    pose_mat = np.array(res_data['pred_thetas'])
    trans_orig = np.array(res_data['transl']).squeeze()
    bbox = np.array(res_data['bbox']).squeeze()

    B = pose_mat.shape[0]
    pose_aa  = sRot.from_matrix(pose_mat.reshape(-1, 3, 3)).as_rotvec().reshape(B, -1)
    pose_aa_orig = pose_aa.copy()

    ## Apply the rotation to make z the up direction
    transform = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False)
    new_root = (transform * sRot.from_rotvec(pose_aa[:, :3])).as_rotvec()
    pose_aa[:, :3] = new_root
    transform.as_matrix(), sRot.from_rotvec(pose_aa[0, :3]).as_matrix()
    trans = trans_orig.dot(transform.as_matrix().T)
    diff_trans = (trans[0, 2] - 0.92)
    trans[:, 2] = trans[:, 2] - diff_trans

    scale = (bbox[:, 2] - bbox[:, 0]) / 256
    trans[:, 1] = trans[:, 1] / scale
    beta = res_data['pred_betas'][0]

    kp_25  = np.zeros([B, 25, 3])
    kp_25_idxes = np.arange(25)[SMPL_2_OP][OP_14_to_OP_12] # wow this needs to get better...

    uv_29 = res_data['pred_uvd'][:, :24]
    pts_12 =smpl_op_to_op(uv_29[:, smpl_2op_submap, :])
    kp_25[:, kp_25_idxes] = pts_12

    for i in range(B):
        bbox_xywh = xyxy2xywh(bbox[i])
        kp_25[i] = kp_25[i] * bbox_xywh[2]
        kp_25[i, :, 0] = kp_25[i, :, 0] + bbox_xywh[0]
        kp_25[i, :, 1] = kp_25[i, :, 1] + bbox_xywh[1]
        kp_25[i, :, 2] = 1 # probability

    ### Assemblle camera
    idx = 0 # Only need to use the first frame, since after that Embodied pose will take over tracking. 
    height, width = res_data['height'][idx], res_data['width'][idx]
    focal = 1000.0
    bbox_xywh = xyxy2xywh(bbox[idx])
    focal = focal / 256 * bbox_xywh[2] # A little hacky
    focal = (2 * focal / min(height, width))

    full_R = sRot.from_euler("xyz", np.array([np.pi/2, 0, 0])).as_matrix()
    full_t = np.array([0,  -diff_trans,  0])
    K =  np.array([[            res_data['height'][0]/2 * (focal/scale[idx]),      0,  res_data['width'][0]/2 ],
                [   0.              ,      res_data['height'][0]/2 * (focal/scale[idx]), res_data['height'][0]/2],
                [   0.              ,      0,           1.              ]])

    cam = {
        "full_R": full_R, 
        "full_t": full_t, 
        "K": K,
        'img_w': res_data['width'][0],
        'img_h': res_data['height'][0],
        'scene_name': None
    }

    pose_mat = sRot.from_rotvec(pose_aa.reshape(B * 24, 3)).as_matrix().reshape(B, 24, 3, 3)
    pose_body = pose_mat[:, 1:22]
    root_orient = pose_mat[:, 0:1]
    new_dict = {}
    start = 0
    end = B
    key = '00'
    new_dict[key] = {
        "joints2d": kp_25[start:end].copy(),
        "pose_body": pose_body[start:end], 
        "root_orient": root_orient[start:end],
        "trans": trans.squeeze()[start:end],
        "pose_aa" : pose_aa.reshape(-1, 72)[start:end],
        "joints": np.zeros([B, 22, 3]),
        "seq_name": "01",
        "pose_6d": np.zeros([B, 24, 6]), 
        'betas': beta,
        "gender": "neutral", 
        "seq_name": key,
        "trans_vel": np.zeros([B, 1, 3]), 
        "joints_vel": np.zeros([B, 22, 3]), 
        "root_orient_vel": np.zeros([B, 1, 3]), 
        "points3d": None,
        "cam": cam
    }
    joblib.dump(new_dict, args.output)