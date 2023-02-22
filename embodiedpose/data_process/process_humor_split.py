import argparse
from enum import EnumMeta
import json
import os
import sys
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

import glob
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot
import torch
from tqdm import tqdm
from mujoco_py import load_model_from_path

sys.path.append(os.getcwd())

from uhc.utils.torch_ext import dict_to_numpy
from uhc.utils.transform_utils import smooth_smpl_quat_window
from uhc.utils.image_utils import get_chunk_with_overlap
from scipy.ndimage import gaussian_filter1d
from uhc.smpllib.smpl_robot import Robot
from uhc.smpllib.torch_smpl_humanoid import Humanoid
import mujoco_py
from uhc.utils.transform_utils import (convert_aa_to_orth6d, rotation_matrix_to_angle_axis)
from uhc.smpllib.smpl_parser import SMPL_Parser
from uhc.utils.torch_geometry_transforms import (angle_axis_to_rotation_matrix)

from uhc.smpllib.smpl_mujoco import smpl_to_qpose_torch
from embodiedpose.models.humor.utils.humor_mujoco import reorder_joints_to_humor, MUJOCO_2_SMPL
from embodiedpose.models.humor.utils.velocities import estimate_velocities
from uhc.utils.config_utils.copycat_config import Config as CC_Config

np.random.seed(1)
LEFT_RIGHT_IDX = [
    0,
    2,
    1,
    3,
    5,
    4,
    6,
    8,
    7,
    9,
    11,
    10,
    12,
    14,
    13,
    15,
    17,
    16,
    19,
    18,
    21,
    20,
    23,
    22,
]
FEMALE_SUBJ_IDS = [162, 3452, 159, 3403]
SEQ_LENGTHS = {
    "BasementSittingBooth_00142_01": 1581,
    "BasementSittingBooth_00145_01": 2182,
    "BasementSittingBooth_03403_01": 2399,
    "BasementSittingBooth_03452_01": 2456,
    "MPH11_00034_01": 2174,
    "MPH11_00150_01": 2187,
    "MPH11_00151_01": 1458,
    "MPH11_00153_01": 2027,
    "MPH11_00157_01": 2000,
    "MPH11_00169_01": 1855,
    "MPH11_03515_01": 1914,
    "MPH112_00034_01": 1837,
    "MPH112_00150_01": 1752,
    "MPH112_00151_01": 1290,
    "MPH112_00157_01": 1241,
    "MPH112_00169_01": 1765,
    "MPH112_03515_01": 1570,
    "MPH16_00157_01": 1785,
    "MPH16_03301_01": 1518,
    "MPH1Library_00034_01": 2211,
    "MPH1Library_00145_01": 3483,
    "MPH1Library_03301_01": 1509,
    "MPH8_00034_01": 2949,
    "MPH8_00168_01": 3071,
    "MPH8_03301_01": 2126,
    "N0SittingBooth_00162_01": 1484,
    "N0SittingBooth_00169_01": 1096,
    "N0SittingBooth_00169_02": 1243,
    "N0SittingBooth_03301_01": 1088,
    "N0SittingBooth_03403_01": 1353,
    "N0Sofa_00034_01": 3038,
    "N0Sofa_00034_02": 1446,
    "N0Sofa_00141_01": 2266,
    "N0Sofa_00145_01": 2104,
    "N0Sofa_03403_01": 1603,
    "N3Library_00157_01": 967,
    "N3Library_00157_02": 714,
    "N3Library_03301_01": 827,
    "N3Library_03301_02": 652,
    "N3Library_03375_01": 1100,
    "N3Library_03375_02": 464,
    "N3Library_03403_01": 670,
    "N3Library_03403_02": 984,
    "N3Office_00034_01": 2152,
    "N3Office_00139_01": 1337,
    "N3Office_00139_02": 2254,
    "N3Office_00150_01": 2599,
    "N3Office_00153_01": 3060,
    "N3Office_00159_01": 2037,
    "N3Office_03301_01": 2044,
    "N3OpenArea_00157_01": 996,
    "N3OpenArea_00157_02": 1325,
    "N3OpenArea_00158_01": 1311,
    "N3OpenArea_00158_02": 1915,
    "N3OpenArea_03301_01": 1056,
    # "N3OpenArea_03403_01": 872,
    "Werkraum_03301_01": 904,
    "Werkraum_03403_01": 901,
    "Werkraum_03516_01": 1991,
    "Werkraum_03516_02": 1531,
}
SEQS = list(SEQ_LENGTHS.keys())

TRIM_EDGES = 90
IMG_WIDTH = 1920
OP_FLIP_MAP = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]


def left_to_rigth_euler(pose_euler):
    pose_euler[:, :, 0] = pose_euler[:, :, 0] * -1
    pose_euler[:, :, 2] = pose_euler[:, :, 2] * -1
    pose_euler = pose_euler[:, LEFT_RIGHT_IDX, :]
    return pose_euler


def transform_smpl(smpl_dict, R, t, seq_length, offset=[0, 0, 0]):
    offset = torch.tensor(offset).float()
    offset_rep = offset.repeat(seq_length, 1)[:, :, None]

    R_rep = R.repeat(seq_length, 1, 1)
    pose_orth = angle_axis_to_rotation_matrix(torch.from_numpy(smpl_dict['pose'].astype(np.float32)[:, :3]).reshape(-1, 3))
    pose_orth = pose_orth[:, :3, :3]
    pose_orth = torch.bmm(R_rep, pose_orth)
    trans = torch.from_numpy(smpl_dict['trans'].astype(np.float32)).reshape(-1, 3, 1)
    trans = torch.bmm(R_rep, (trans + offset_rep)) - offset_rep
    trans = trans[:, :, 0] + t[None]

    pose = np.array(rotation_matrix_to_angle_axis(pose_orth).reshape(seq_length, 3))
    trans = np.array(trans)
    return pose, trans


def smpl_2_entry(
    seq_name,
    smpl_dict,
):
    pose_aa = smpl_dict["pose"]
    trans = smpl_dict["trans"]
    seq_len = pose_aa.shape[0]
    shape = smpl_dict["shape"] if "shape" in smpl_dict else np.zeros([seq_len, 10])
    # mean_shape = shape.mean(axis=0)
    mean_shape = shape[0:10].mean(axis=0)
    # import ipdb; ipdb.set_trace()
    gender = smpl_dict["gender"] if "gender" in smpl_dict else "neutral"
    joints2d = smpl_dict["joints2d"] if "joints2d" in smpl_dict else None
    points3d = smpl_dict["points3d"] if "points3d" in smpl_dict else None
    th_betas = smpl_dict['shape']

    seq_length = pose_aa.shape[0]
    if seq_length < 10:
        return None
    pose_aa = torch.from_numpy(pose_aa).float()
    pose_seq_6d = convert_aa_to_orth6d(pose_aa).reshape(-1, 144)
    smpl_robot.load_from_skeleton(torch.from_numpy(mean_shape[None,]), gender=[0], objs_info=None)
    model = mujoco_py.load_model_from_xml(smpl_robot.export_xml_string().decode("utf-8"))
    humanoid = Humanoid(model=model)
    qpos = smpl_to_qpose_torch(pose_aa, model, trans=torch.from_numpy(trans), count_offset=True)
    fk_result = humanoid.qpos_fk(qpos, to_numpy=False)

    root_orient = pose_aa[:, :3]
    joints_humor = reorder_joints_to_humor(fk_result['wbpos'].clone(), model, cc_cfg.robot_cfg.get("model", "smpl"))[:, :66]

    trans_vel, joints_humor_vel, root_orient_vel = estimate_velocities(torch.from_numpy(trans[None]), root_orient[None], joints_humor[None], 30)
    trans_vel = trans_vel[0]
    joints_humor_vel = joints_humor_vel[0]
    root_orient_vel = root_orient_vel[0]

    root_orient_mat = angle_axis_to_rotation_matrix(root_orient)[:, :3, :3]
    pose_body = pose_aa[:, 3:].reshape(-1, 23, 3)[:, :21]
    pose_body = angle_axis_to_rotation_matrix(pose_body.reshape(-1, 3))
    pose_body = pose_body.reshape(-1, 21, 4, 4)[:, :, :3, :3]

    camera_params = load_camera_params(seq_name[:-9])

    entry = {
        # "expert": fk_result,
        "pose_aa": pose_aa,
        "pose_6d": pose_seq_6d,
        "pose_body": pose_body,
        "trans": trans,
        "trans_vel": trans_vel,
        "root_orient": root_orient_mat,
        "root_orient_vel": root_orient_vel,
        "joints": joints_humor,
        "joints_vel": joints_humor_vel,
        "betas": mean_shape,
        "seq_name": seq_name,
        "gender": gender,
        "joints2d": joints2d,
        "points3d": points3d,
        "cam": {
            "full_R": camera_params['full_R'],
            "full_t": camera_params['full_t'],
            "K": camera_params['K'],
            "img_w": 1980,
            "img_h": 1080,
            "scene_name": seq_name[:-9]
        }
    }
    return dict_to_numpy(entry)


def read_keypoints(keypoint_fn):
    '''
    Only reads body keypoint data of first person.
    '''
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    if len(data['people']) == 0:
        print('WARNING: Found no keypoints in %s! Returning zeros!' % (keypoint_fn))
        return np.zeros((25, 3), dtype=np.float)

    person_data = data['people'][0]
    body_keypoints = np.array(person_data['pose_keypoints_2d'], dtype=np.float)
    body_keypoints = body_keypoints.reshape([-1, 3])

    return body_keypoints


def load_camera_params(scene_name):
    prox_path = "/hdd/zen/data/video_pose/prox/qualitative"

    with open(f'{prox_path}/calibration/Color.json', 'r') as f:
        cameraInfo = json.load(f)
        K = np.array(cameraInfo['camera_mtx']).astype(np.float32)

    with open(f'{prox_path}/cam2world/{scene_name}.json', 'r') as f:
        camera_pose = np.array(json.load(f)).astype(np.float32)
        R = camera_pose[:3, :3]
        tr = camera_pose[:3, 3]
        R = R.T
        tr = -np.matmul(R, tr)

    with open(f'{prox_path}/alignment/{scene_name}.npz', 'rb') as f:
        aRt = np.load(f)
        aR = aRt['R']
        atr = aRt['t']
        aR = aR.T
        atr = -np.matmul(aR, atr)
    full_R = R.dot(aR)
    full_t = R.dot(atr) + tr

    cam_params = {"K": K, "R": R, "tr": tr, "aR": aR, "atr": atr, "full_R": full_R, "full_t": full_t}
    return cam_params


def load_humanoid():
    cc_cfg = CC_Config(cfg_id="copycat_e_1", base_dir="../Copycat")
    smpl_robot = Robot(
        cc_cfg.robot_cfg,
        data_dir=osp.join(cc_cfg.base_dir, "data/smpl"),
        masterfoot=cc_cfg.masterfoot,
    )
    model = mujoco_py.load_model_from_xml(smpl_robot.export_xml_string().decode("utf-8"))
    humanoid = Humanoid(model=model)
    return smpl_robot, humanoid, cc_cfg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=["prox", "proxd"], default="proxd")
    args = parser.parse_args()

    smpl_robot, humanoid, cc_cfg = load_humanoid()

    data_res = {}
    seq_length = -1
    video_annot = {}
    seq_counter = 0
    single_seq_length = 60
    fix_feet = True

    # humor_base = "/home/shun/dev/cmu/humor/"
    # prox_base = '/data/dataset/prox_dataset/'

    humor_base = "/hdd/zen/dev/copycat/humor/"
    prox_base = '/hdd/zen/data/video_pose/prox/qualitative/'

    pbar = tqdm(SEQS)
    for idx, seq in enumerate(pbar):
        pbar.set_description(seq)

        # if seq != "N3Library_03301_01":
        #     continue
        # if seq != "MPH1Library_00034_01":
        #     continue
        # color_images = glob.glob(
        #     osp.join(prox_base, "recordings", seq, "Color", "*.jpg"))
        result_path = osp.join(humor_base, f'out/{args.dataset}_fitting_sub/results_out/{seq}_*/stage3_results.npz')
        # result_path = osp.join(humor_base, f'out/{args.dataset}_fitting_sub2/results_out/{seq}_*/proxd_results.npz')
        obs_path = osp.join(humor_base, f'out/{args.dataset}_fitting_sub/results_out/{seq}_*/observations.npz')
        motion_files = sorted(glob.glob(result_path))
        obs_files = sorted(glob.glob(obs_path))

        if len(motion_files) == 0:
            print("Empty motion files for seq:", seq, idx)
            continue

        with open(osp.join(prox_base, f'cam2world/{seq[:-9]}.json'), 'r') as f:
            camera_pose = np.array(json.load(f)).astype(np.float32)
            R = torch.from_numpy(camera_pose[:3, :3])
            t = torch.from_numpy(camera_pose[:3, 3])

        with open(osp.join(prox_base, f'alignment/{seq[:-9]}.npz'), 'rb') as f:
            aRt = np.load(f)
            aR = torch.from_numpy(aRt['R'])
            at = torch.from_numpy(aRt['t'])

        seq_length = SEQ_LENGTHS[seq] - TRIM_EDGES * 2
        cur_subj_id = seq.split('_')[1]
        gender = 'female' if int(cur_subj_id) in FEMALE_SUBJ_IDS else 'male'
        smpl_dict = {'pose': np.zeros((seq_length, 72)), 'trans': np.zeros((seq_length, 3)), 'shape': np.zeros((seq_length, 16)), 'joints2d': np.zeros((seq_length, 25, 3)), 'gender': gender}
        if args.dataset == 'proxd':
            smpl_dict['points3d'] = np.zeros((seq_length, 4096, 3))

        files = zip(motion_files, obs_files)

        chunk_bounds, selects = get_chunk_with_overlap(seq_length, 60, 10, True)
        # print(chunk_bounds)
        # chunk_bounds, selects = get_chunk_with_overlap(seq_length, 60, 0, True)
        overlap = 0
        for i, (motion_file, obs_file) in enumerate(files):
            curr_seq = chunk_bounds[i]

            if curr_seq[0] > seq_length:
                print("skipping", curr_seq[0])
                break
            if curr_seq[0] != 0:
                overlap = 5  # choose the middle frame
            else:
                overlap = 0
            start = curr_seq[0] + overlap
            end = curr_seq[1]
            motion = np.load(motion_file)
            obs = np.load(obs_file)
            len_seq = smpl_dict['joints2d'][start:end].shape[0]
            smpl_dict['joints2d'][start:end] = obs["joints2d"][overlap:(len_seq + overlap)]

            smpl_dict['pose'][start:end, :3] = motion['root_orient'][overlap:(len_seq + overlap)]
            smpl_dict['pose'][start:end, 3:66] = motion['pose_body'][overlap:(len_seq + overlap)]
            smpl_dict['trans'][start:end] = motion['trans'][overlap:(len_seq + overlap)]

            smpl_dict['shape'][start:end] = motion['betas']  # Betas!!
            # smpl_dict['shape'][start:end] = motion['betas'][overlap:(len_seq + overlap)] # Betas!!
            if args.dataset == 'proxd':
                smpl_dict['points3d'][start:end] = obs["points3d"][overlap:(len_seq + overlap)]
            # print(start, end, motion_file)
            if curr_seq[0] != 0:
                window = 15
                smpl_dict['trans'][(curr_seq[0] - window):(curr_seq[0] + window)] = gaussian_filter1d(smpl_dict['trans'][(curr_seq[0] - window):(curr_seq[0] + window)], 1, axis=0)
                smpl_dict['pose'][(curr_seq[0] - window):(curr_seq[0] + window)] = smooth_smpl_quat_window(smpl_dict['pose'][(curr_seq[0] - window):(curr_seq[0] + window)], sigma=2).reshape(-1, 72)

        print(chunk_bounds.tolist())
        kp2d_files = sorted(glob.glob(osp.join(prox_base, "keypoints", seq, "*")))
        kp_data = np.array([read_keypoints(kp2d_file) for kp2d_file in kp2d_files[TRIM_EDGES:-TRIM_EDGES]])
        kp_data = kp_data[:, OP_FLIP_MAP, :]
        kp_data[:, :, 0] = IMG_WIDTH - kp_data[:, :, 0]
        smpl_dict['joints2d'] = kp_data

        ####################################################### Use HRNet
        kp_2d = np.load(open(f"/hdd/zen/data/video_pose/prox/dekr_res/{seq}.pkl", "rb"))[TRIM_EDGES:-TRIM_EDGES]
        smpl_dict['joints2d'][:, 1:15] = kp_2d

        # if (seq_length != end):
        # import ipdb; ipdb.set_trace()

        # Transform from the camera to world coordinate system
        pose, trans = transform_smpl(smpl_dict, R, t, seq_length, offset=humanoid.model.body_pos[1])
        smpl_dict['pose'][:, :3] = pose
        smpl_dict['trans'] = trans

        # Align the ground plane to the xy plane of the world coordinate system
        pose, trans = transform_smpl(smpl_dict, aR, at, seq_length, offset=humanoid.model.body_pos[1])
        smpl_dict['pose'][:, :3] = pose
        smpl_dict['trans'] = trans
        entry = smpl_2_entry(seq, smpl_dict)
        data_res[seq] = entry
        # import ipdb; ipdb.set_trace()
        # break
        # if entry is not None:
        #
        #     counter += 1

    print(data_res.keys())
    output_file_name = osp.join(prox_base, f'thirdeye_anns_{args.dataset}_overlap.pkl')
    # output_file_name = osp.join(prox_base, f'thirdeye_anns_{args.dataset}_single.pkl')
    # output_file_name = osp.join(prox_base,
    # f'thirdeye_anns_{args.dataset}_single2.pkl')
    # output_file_name = osp.join(prox_base, f'thirdeye_anns_{args.dataset}_overlap_stage3.pkl')
    print(output_file_name, len(data_res))
    joblib.dump(data_res, open(output_file_name, "wb"))
