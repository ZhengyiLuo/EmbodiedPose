import argparse
import glob
import numpy as np
import os
import sys
import pickle
import joblib

sys.path.append(os.getcwd())

from uhc.smpllib.smpl_robot import Robot
from uhc.smpllib.torch_smpl_humanoid import Humanoid
import mujoco_py

from uhc.utils.config_utils.copycat_config import Config as CC_Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='lemo', choices=['lemo', 'prox', 'proxd'])
    args = parser.parse_args()

    TRIM_EDGES = 90
    prox_path = "/data/dataset/prox_dataset/"
    if args.dataset == 'prox':
        results_path = '/data/dataset/prox_dataset/PROX_v2/**/*.pkl'
    elif args.dataset == 'proxd':
        results_path = '/data/dataset/prox_dataset/PROXD_v2/**/*.pkl'
    elif args.dataset == 'lemo':
        results_path = '/data/dataset/prox_dataset/LEMO_results/**/*.pkl'
    results_filepaths = glob.glob(results_path)

    data = {}
    for result_filepath in results_filepaths:
        with open(result_filepath, 'rb') as f:
            results = pickle.load(f)
        betas = results['betas']
        global_orient = results['global_orient']
        body_pose = results['body_pose']
        transl = results['transl']
        seq_length = transl.shape[0] - 2 * TRIM_EDGES

        pose_aa = np.concatenate((global_orient, body_pose, np.zeros((body_pose.shape[0], 6))), axis=1)

        female_subjects_ids = [162, 3452, 159, 3403]
        subject_id = int(result_filepath.split('_')[-2])
        if subject_id in female_subjects_ids:
            gender = 'female'
        else:
            gender = 'male'

        seq_name = result_filepath.split('/')[-2]

        smpl_dict = {}
        smpl_dict['gender'] = gender
        smpl_dict['betas'] = betas
        smpl_dict['pose_aa'] = pose_aa[TRIM_EDGES:-TRIM_EDGES]
        smpl_dict['trans'] = transl[TRIM_EDGES:-TRIM_EDGES]
        data[seq_name] = smpl_dict

    joblib.dump(data, f'/data/dataset/prox_dataset/{args.dataset}_processed.pkl')
    print(args.dataset)
    print(len(data.keys()))
    print('Saved')
