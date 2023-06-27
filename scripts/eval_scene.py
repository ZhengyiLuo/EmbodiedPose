'''
File: /eval_scene.py
Created Date: Wednesday February 16th 2022
Author: Zhengyi Luo
Comment:
-----
Last Modified: Wednesday February 16th 2022 9:08:54 pm
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
import mujoco_py
import io

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(os.getcwd())

import torch
import numpy as np

from uhc.utils.flags import flags

from uhc.utils.image_utils import write_frames_to_video
import wandb
from uhc.utils.copycat_visualizer import CopycatVisualizer

from embodiedpose.agents import agent_dict
from embodiedpose.utils.video_pose_config import Config
from scipy.spatial.transform import Rotation as sRot
from uhc.smpllib.smpl_eval import compute_metrics
from uhc.utils.math_utils import smpl_op_to_op
import matplotlib.pyplot as plt


class SceneVisulizer(CopycatVisualizer):

    def update_pose(self):

        # if self.env_vis.viewer._record_video:
        #     print(self.fr)
        # print(self.fr)
        expert = self.agent.env.expert
        lim = self.agent.env.converter.new_nq
        # self.data["pred"][self.fr][-14:] = expert["obj_pose"][self.fr]

        self.env_vis.data.qpos[:lim] = self.data["pred"][self.fr]
        self.env_vis.data.qpos[lim:(lim * 2)] = self.data["gt"][self.fr]
        self.env_vis.data.qpos[(lim * 2):] = self.data["target"][self.fr]

        if (self.agent.cfg.render_rfc and self.agent.cc_cfg.residual_force and self.agent.cc_cfg.residual_force_mode == "explicit"):

            self.render_virtual_force(self.data["vf_world"][self.fr])

        # self.env_vis.data.qpos[env.model.nq] += 1.0
        # if args.record_expert:
        # self.env_vis.data.qpos[:env.model.nq] = self.data['gt'][self.fr]
        if self.agent.cfg.hide_im:
            self.env_vis.data.qpos[2] = 100.0

        if self.agent.cfg.hide_expert:
            self.env_vis.data.qpos[lim + 2] = 100.0
        if self.agent.cfg.shift_expert:
            self.env_vis.data.qpos[lim] += 3

        # self.env_vis.data.qpos[lim * 2 + 2] += 2
        if self.agent.cfg.shift_kin:
            # self.env_vis.data.qpos[lim * 2] += 0.5
            self.env_vis.data.qpos[lim * 2 + 1] += 50
        if self.fr == 146:
            self.agent.env.context_dict

        self.env_vis.data.qpos[lim * 2 + 2] += 50
        self.env_vis.data.qpos[lim + 2] += 50

        if self.agent.cfg.focus:
            self.env_vis.viewer.cam.lookat[:2] = self.env_vis.data.qpos[:2]

        # if cfg.model_specs.get("use_tcn", False):
        # self.env_vis.model.geom_pos[1:15] = self.body_pos[self.fr]

        if self.fr == 1:
            full_R, full_t = self.agent.env.camera_params['full_R'], self.agent.env.camera_params['full_t']

            distance = np.linalg.norm(full_t)
            x_axis = full_R.T[:, 0]
            pos_3d = -full_R.T.dot(full_t)
            rotation = sRot.from_matrix(full_R).as_euler("XYZ", degrees=True)
            self.env_vis.viewer.cam.lookat[:] = pos_3d + x_axis
            self.env_vis.viewer.cam.azimuth = 90 - rotation[2]
            self.env_vis.viewer.cam.elevation = -8.0
        self.env_vis.sim_forward()
        print(f"Current frame: {self.fr}", end='\r')

    def display_coverage(self):

        res_dir = osp.join(
            self.agent.cfg.output_dir,
            f"{self.agent.cfg.epoch}_{self.agent.test_data_loaders[0].name}_coverage_full.pkl",
        )
        print(res_dir)
        data_res = joblib.load(res_dir)
        print(len(data_res))

        # data_res = {k:v  for k, v in data_res.items() if v['mpjpe'] > 100}

        def vis_gen():
            keys = sorted(list(data_res.keys()))
            keys = list(data_res.keys())
            # keys = [k for k in keys if "eat" in k.lower()]
            keys = [k for k in keys if k in ["N0Sofa_00145_01"]]
            for take_key in keys:

                loader = agent.test_data_loaders[0]
                context_sample = loader.get_sample_from_key(take_key=take_key, full_sample=True, return_batch=True)
                self.agent.env.load_context(self.agent.policy_net.init_context(context_sample))
                self.env_vis.reload_sim_model(
                    # self.agent.env.smpl_robot.export_vis_string_self(
                    #     num=3, num_cones=16).decode("utf-8"))
                    self.agent.env.smpl_robot.export_vis_string_self(num=3, num_cones=0).decode("utf-8"))

                self.setup_viewing_angle()
                v = data_res[take_key]

                # v['pred'][:, 7:] = gaussian_filter1d(v['pred'][:, 7:], sigma=1, axis = 0)
                # v['pred'][:, :3] = gaussian_filter1d(v['pred'][:, :3], sigma=1, axis = 0)

                if cfg.model_specs.get("use_tcn", False):
                    self.body_pos = v['world_body_pos'] + v['world_trans']
                    # tcn_bpos = smpl_op_to_op(v['world_body_pos'])
                    # tcn_bpos -= tcn_bpos[..., 7:8, :]
                    # gt_jpos = v['gt_jpos'].reshape(-1, 12, 3)
                    # gt_jpos_local = gt_jpos - gt_jpos[..., 7:8, :]
                    # mpjpe_tcn = np.linalg.norm(tcn_bpos - gt_jpos_local, axis = -1).mean() * 1000
                    # print(f"TCN MPJPE: {mpjpe_tcn:.5f}" )
                metric_res = compute_metrics(v, None)
                metric_res = {k: np.mean(v) for k, v in metric_res.items()}
                print_str = " \t".join([f"{k}: {v:.3f}" for k, v in metric_res.items()])
                print(print_str)
                print(f"{v['percent']:.3f} |  {take_key}")

                self.num_fr = len(v["pred"])
                self.set_video_path(
                    image_path=osp.join(
                        self.agent.cfg.output,
                        str(take_key),
                        f"{take_key}_{self.agent.cfg.id}_{self.agent.cfg.epoch}_%04d.png",
                    ),
                    video_path=osp.join(
                        self.agent.cfg.output,
                        f"{take_key}_{self.agent.cfg.id}_{self.agent.cfg.epoch}_%01d.mp4",
                    ),
                )
                yield v

        self.data_gen = iter(vis_gen())
        self.data = next(self.data_gen)
        self.show_animation()

    def data_generator(self):
        if self.agent.cfg.mode != "disp_stats":
            for loader in self.agent.test_data_loaders:
                for take_key in loader.data_keys:
                    print(f"Generating for {take_key} seqlen: {loader.get_sample_len_from_key(take_key)}")
                    context_sample = loader.get_sample_from_key(take_key, full_sample=True, return_batch=True)
                    self.agent.env.load_context(self.agent.policy_net.init_context(context_sample, random_cam=not 'cam' in context_sample))
                    # import ipdb; ipdb.set_trace()
                    # joblib.dump(self.agent.env.context_dict['joints2d'], "a2.pkl")

                    eval_res = self.agent.eval_seq(take_key, loader)

                    print("Agent Mass:", mujoco_py.functions.mj_getTotalmass(self.agent.env.model), f"Seq_len: {eval_res['gt'].shape[0]}")
                    if cfg.model_specs.get("use_tcn", False):
                        self.body_pos = eval_res['world_body_pos'] + eval_res['world_trans']

                    metric_res = compute_metrics(eval_res, None)
                    metric_res = {k: np.mean(v) for k, v in metric_res.items()}
                    print_str = " \t".join([f"{k}: {v:.3f}" for k, v in metric_res.items()])
                    print("!!!Metrics computed against GT, for PROX it this is results from HuMoR")
                    print("!!!Metrics computed against GT, for PROX it this is results from HuMoR")
                    print("!!!Metrics computed against GT, for PROX it this is results from HuMoR")
                    print(print_str)
                    
                    
                    self.env_vis.reload_sim_model(
                        # self.agent.env.smpl_robot.export_vis_string_self(
                        #     num=3, num_cones=16).decode("utf-8"))
                        self.agent.env.smpl_robot.export_vis_string_self(num=3, num_cones=0).decode("utf-8"))

                    self.setup_viewing_angle()
                    self.set_video_path(
                        image_path=osp.join(
                            self.agent.cfg.output,
                            str(take_key),
                            f"{take_key}_{self.agent.cfg.id}_{self.agent.cfg.epoch}_%04d.png",
                        ),
                        video_path=osp.join(
                            self.agent.cfg.output,
                            f"{take_key}_{self.agent.cfg.id}_{self.agent.cfg.epoch}_%01d.mp4",
                        ),
                    )
                    os.makedirs(osp.join(self.agent.cfg.output, str(take_key)), exist_ok=True)
                    self.num_fr = eval_res["pred"].shape[0]

                    yield eval_res
        else:
            yield None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--num_threads", type=int, default=30)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument("--show_noise", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_log", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--data", type=str, default="sample_data/thirdeye_anns_prox_overlap_no_clip.pkl")
    parser.add_argument("--mode", type=str, default="vis")
    parser.add_argument("--render_rfc", action="store_true", default=False)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--hide_expert", action="store_true", default=False)
    parser.add_argument("--no_fail_safe", action="store_true", default=False)
    parser.add_argument("--focus", action="store_true", default=False)
    parser.add_argument("--output", type=str, default="test")
    parser.add_argument("--shift_expert", action="store_true", default=False)
    parser.add_argument("--shift_kin", action="store_false", default=True)
    parser.add_argument("--smplx", action="store_true", default=False)
    parser.add_argument("--hide_im", action="store_true", default=False)
    parser.add_argument("--filter_res", action="store_true", default=False)
    parser.add_argument("--no_filter_2d", action="store_true", default=False)
    args = parser.parse_args()

    cfg = Config(cfg_id=args.cfg, create_dirs=False)
    cfg.update(args)

    flags.debug = args.debug
    flags.no_filter_2d = args.no_filter_2d
    cfg.no_log = True
    if args.no_fail_safe:
        cfg.fail_safe = False

    prox_path = cfg.data_specs['prox_path']
    cfg.output = osp.join(prox_path, "renderings/sceneplus", f"{cfg.id}")
    os.makedirs(cfg.output, exist_ok=True)

    if cfg.mode == "vis":
        cfg.num_threads = 1

    dtype = torch.float64
    torch.set_default_dtype(dtype)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    print(f"Using: {device}")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.smplx and cfg.robot_cfg["model"] == "smplh":
        cfg.robot_cfg["model"] = "smplx"

    cfg.data_specs["train_files_path"] = [(cfg.data,"scene_pose")]
    cfg.data_specs["test_files_path"] = [(cfg.data,"scene_pose")]

    if cfg.mode == "vis":
        cfg.num_threads = 1
    """create agent"""
    agent_class = agent_dict[cfg.agent_name]
    agent = agent_class(cfg=cfg, dtype=dtype, device=device, checkpoint_epoch=cfg.epoch, mode="test")

    if args.mode == "stats":
        agent.eval_policy(epoch=cfg.epoch, dump=True)
    elif args.mode == "disp_stats":
        vis = SceneVisulizer(agent.env.smpl_robot.export_vis_string().decode("utf-8"), agent)
        vis.display_coverage()
    else:
        vis = SceneVisulizer(
            agent.env.smpl_robot.export_vis_string().decode("utf-8"),
            agent,
        )

        vis.show_animation()
