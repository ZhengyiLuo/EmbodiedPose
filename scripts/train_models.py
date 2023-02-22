'''
File: /train_scene.py
Created Date: Tuesday February 15th 2022
Author: Zhengyi Luo
Comment:
-----
Last Modified: Tuesday February 15th 2022 7:39:02 pm
Modified By: Zhengyi Luo at <zluo2@cs.cmu.edu>
-----
Copyright (c) 2022 Carnegie Mellon University, KLab
-----
'''
import argparse

import sys
import pickle
import time
import joblib
import glob
import pdb
import os.path as osp
import os

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(os.getcwd())

import torch
import numpy as np

from uhc.utils.flags import flags
import wandb

from embodiedpose.agents import agent_dict
from embodiedpose.utils.video_pose_config import Config


def main_loop():
    if args.render:
        agent.pre_epoch_update(start_epoch)
        agent.sample(1e8)
    else:
        for epoch in range(start_epoch, cfg.num_epoch):
            agent.optimize_policy(epoch)
            """clean up gpu memory"""
            torch.cuda.empty_cache()

        agent.logger.info("training done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument("--num_threads", type=int, default=30)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_log", action="store_true", default=False)
    parser.add_argument("--show_noise", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    if args.render:
        args.num_threads = 1
    cfg = Config(cfg_id=args.cfg, create_dirs=not (args.render or args.epoch > 0))

    if args.debug:
        args.num_threads = 1
        args.no_log = True
        # cfg.get("data_specs", {})["train_files_path"] = [[
        #     "/hdd/zen/data/ActBound/AMASS/amass_copycat_take5_test.pkl",
        #     "amass"
        # ]]
        cfg.get("data_specs", {})["train_files_path"] = [["/hdd/zen/data/video_pose/h36m/data_fit/h36m_test_30_gt_fk.p", "scene_pose"]]

    cfg.update(args)
    flags.debug = args.debug

    if not args.no_log:
        wandb.init(
            project=cfg.proj_name,
            resume=not args.resume is None,
            id=args.resume,
            notes=cfg.notes,
        )
        wandb.config.update(vars(cfg), allow_val_change=True)
        wandb.config.update(args, allow_val_change=True)
        wandb.run.name = args.cfg
        wandb.run.save()

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = (torch.device("cuda", index=args.gpu_index) if torch.cuda.is_available() else torch.device("cpu"))
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    start_epoch = int(args.epoch)
    """create agent"""
    agent_class = agent_dict[cfg.agent_name]
    agent = agent_class(
        cfg=cfg,
        dtype=dtype,
        device=device,
        mode="train",
        checkpoint_epoch=args.epoch,
    )
    main_loop()
