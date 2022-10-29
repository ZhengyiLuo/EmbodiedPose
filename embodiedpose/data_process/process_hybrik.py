import glob
import os
import os.path as osp
import sys
import joblib

import numpy as np
import torch

sys.path.append(os.getcwd())

from embodiedpose.utils.torch_geometry_transforms import quaternion_to_angle_axis as quat2aa

TRIM_EDGES = 90

if __name__ == '__main__':
    seq_filenames = sorted(glob.glob('/home/shun/dev/cmu/HybrIK/data/PROX/*'))

    data = {}
    for seq_filename in seq_filenames:
        seq_name = osp.basename(seq_filename)
        female_subjects_ids = [162, 3452, 159, 3403]
        subject_id = int(seq_filename.split('_')[-2])
        if subject_id in female_subjects_ids:
            gender = 'female'
        else:
            gender = 'male'
        npy_filenames = sorted(glob.glob(osp.join(seq_filename, '*.npz')))
        smpl_dict = {
            'betas': [],
            'trans': [],
            'pose_aa': [],
            'joints3d': [],
            'gender': gender,
        }
        for npy_filename in npy_filenames:
            result = np.load(npy_filename)
            smpl_dict['betas'].append(result['pred_shape'])
            smpl_dict['trans'].append(result['pred_3d_root'] / 1000.0)
            pose_aa = quat2aa(torch.from_numpy(
                result['pred_theta_quat'])).numpy().reshape(-1)
            smpl_dict['pose_aa'].append(pose_aa)
            smpl_dict['joints3d'].append(result['pred_3d_kpt'])

        smpl_dict['betas'] = np.stack(smpl_dict['betas'][0],
                                      axis=0)[:10].mean(axis=0)
        smpl_dict['pose_aa'] = np.stack(
            smpl_dict['pose_aa'])[TRIM_EDGES:-TRIM_EDGES]
        smpl_dict['trans'] = np.stack(
            smpl_dict['trans'])[TRIM_EDGES:-TRIM_EDGES]
        smpl_dict['joints3d'] = np.stack(
            smpl_dict['joints3d'])[TRIM_EDGES:-TRIM_EDGES]

        data[seq_name] = smpl_dict
        print(seq_name)

    joblib.dump(data,
                open('/data/dataset/prox_dataset/hybrik_processed.pkl', "wb"))
