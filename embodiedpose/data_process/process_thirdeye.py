import os
import argparse
import joblib
import glob
import os.path as osp

import torch
import mujoco_py

from uhc.smpllib.smpl_mujoco import qpos_to_smpl
from uhc.utils.config_utils.copycat_config import Config as CC_Config
from uhc.smpllib.smpl_robot import Robot
from uhc.smpllib.torch_smpl_humanoid import Humanoid


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_id", default=None)
    parser.add_argument("--epoch", default=-1, type=int)
    parser.add_argument("--data", default="prox", type=str)
    args = parser.parse_args()

    data_root = '/hdd/zen/data/video_pose/prox/qualitative/'
    res_dir = f"results/scene+/{args.cfg_id}/results/"
    humor_path = osp.join(data_root, "thirdeye_anns_proxd_overlap_full.pkl")
    if args.data == "prox":
        thirdeye_path = osp.join(res_dir, f"{args.epoch}_thirdeye_anns_prox_overlap_no_clip_coverage_full.pkl")
    elif args.data == "proxd":
        thirdeye_path = osp.join(res_dir, f"{args.epoch}_thirdeye_anns_proxd_overlap_full_coverage_full.pkl")

    humor_results = joblib.load(humor_path)
    thirdeye_results = joblib.load(thirdeye_path)

    data = {}
    success_cnt = 0
    smpl_robot, humanoid, cc_cfg = load_humanoid()
    for seq_name in humor_results.keys():
        mean_shape = humor_results[seq_name]['betas']
        qpos = thirdeye_results[seq_name]['pred']
        if thirdeye_results[seq_name]['succ']:
            success_cnt += 1

        smpl_robot.load_from_skeleton(torch.from_numpy(mean_shape[None,]), gender=[0], objs_info=None)
        model = mujoco_py.load_model_from_xml(smpl_robot.export_xml_string().decode("utf-8"))

        pose_aa, trans = qpos_to_smpl(qpos, model, cc_cfg.robot_cfg.get("model", "smpl"))

        female_subjects_ids = [162, 3452, 159, 3403]
        subject_id = int(seq_name.split('_')[-2])
        if subject_id in female_subjects_ids:
            gender = 'female'
        else:
            gender = 'male'
        smpl_dict = {}
        smpl_dict['gender'] = gender
        smpl_dict['betas'] = mean_shape
        smpl_dict['pose_aa'] = pose_aa
        smpl_dict['trans'] = trans
        smpl_dict['fail'] = thirdeye_results[seq_name]['fail_safe']
        if smpl_dict['fail']:
            print('fail!')
        data[seq_name] = smpl_dict

        print(seq_name, "gender:", gender)
    data['success_rate'] = success_cnt / 60
    out_path = osp.join(res_dir, f'thirdeye_final_processed.pkl')
    joblib.dump(data, out_path)
    print('Saved')
    print(thirdeye_path, out_path)
