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
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Process, Queue
import imageio
from tqdm import tqdm


# view_angle_dict = {
#  'BasementSittingBooth':]
#  'MPH11',
#  'MPH112',
#  'MPH16',
#  'MPH1Library',
#  'MPH8',
#  'N0SittingBooth',
#  'N0Sofa',
#  'N3Library',
#  'N3Office',
#  'N3OpenArea',
#  'Werkraum'
# }

class SceneVisulizer(CopycatVisualizer):

    def reload_sim_model(self, xml_str):
        del self.sim
        del self.model
        del self.data
        del self.viewer
        del self._viewers
        self.model = mujoco_py.load_model_from_xml(xml_str)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.init_qpos = self.sim.data.qpos.copy()
        self.init_qvel = self.sim.data.qvel.copy()
        self.viewer = None
        self._viewers = {}
        self._get_viewer("image")._hide_overlay = True
        self.reset()
        print("Reloading Vis Sim")

    def start_video_recording(self, filename, fps = 30):
        print(f"============ Writing video to {filename} fps:{fps}============")
        self.writer = imageio.get_writer(filename, fps=fps, macro_block_size=None)


    def end_video_recording(self):
        self.writer.close()
        del self.writer

    def render(self):
        # size = (1920, 1080)
        size = (960, 540)
        self.env_vis.render(mode = "image", width = size[0], height = size[1])
        data = np.asarray(
                self.env_vis.viewer.read_pixels(size[0], size[1], depth=False)[::-1, :, :],
                dtype=np.uint8,
            )
        self.writer.append_data(data)
        # self.env_vis.render(mode = "human", width = size[0], height = size[1])
        # print(self.env_vis.viewer.cam.elevation,
        #       self.env_vis.viewer.cam.distance,
        #       self.env_vis.viewer.cam.azimuth, self.env_vis.viewer.cam.lookat)


    def update_pose(self):
        lim = self.agent.env.converter.new_nq
        self.env_vis.data.qpos[:lim] = self.data["pred"][self.fr]
        self.env_vis.sim_forward()

    def render_coverage(self):

        res_dir = osp.join(
            self.agent.cfg.output_dir,
            f"{self.agent.cfg.epoch}_{self.agent.test_data_loaders[0].name}_coverage_full.pkl",
        )
        print(res_dir)
        data_res = joblib.load(res_dir)
        print(len(data_res))

        # data_res = {k:v  for k, v in data_res.items() if v['mpjpe'] > 100}

        keys = sorted(list(data_res.keys()))
        # keys = [k for k in keys if "eat" in k.lower()]
        for take_key in keys:
            data_out = osp.join(self.agent.cfg.output, args.data + f"_{args.epoch}")
            os.makedirs(data_out, exist_ok=True)
            video_path=osp.join(
                data_out,
                    f"{take_key}.mp4",
            )

            # if osp.isfile(video_path):
            #     print("Already rendered!!!!: ", video_path)
            #     continue

            loader = agent.test_data_loaders[0]
            context_sample = loader.get_sample_from_key(take_key=take_key,
                                                        full_sample=True,
                                                        return_batch=True)
            self.agent.env.load_context(
                self.agent.policy_net.init_context(context_sample))
            self.env_vis.reload_sim_model(
                self.agent.env.smpl_robot.export_vis_string_self(
                    num=0, num_cones=0).decode("utf-8"))
            v = data_res[take_key]

            # v['pred'][:, 7:] = gaussian_filter1d(v['pred'][:, 7:], sigma=2, axis = 0)
            # v['pred'][:, :3] = gaussian_filter1d(v['pred'][:, :3], sigma=2, axis = 0)

            # print_str = " \t".join([f"{k}: {v:.3f}" for k, v in compute_metrics(v, None).items()])
            # print(print_str)
            # print(f"{v['percent']:.3f} |  {take_key}")
            self.data = v

            self.start_video_recording(video_path)
            self.fr = 0
            full_R, full_t = self.agent.env.camera_params[
                'full_R'], self.agent.env.camera_params['full_t']


            distance = np.linalg.norm(full_t)
            x_axis = full_R.T[:, 0]
            pos_3d = -full_R.T.dot(full_t)
            rotation = sRot.from_matrix(full_R).as_euler("XYZ", degrees=True)
            self.env_vis.viewer.cam.lookat[:] = pos_3d + x_axis
            self.env_vis.viewer.cam.azimuth = 90 - rotation[2]
            self.env_vis.viewer.cam.elevation = -8.0
            self.env_vis.viewer.cam.distance = 4

            if take_key.startswith("BasementSittingBooth"):
                self.env_vis.viewer.cam.elevation = -26; self.env_vis.viewer.cam.distance = 4; self.env_vis.viewer.cam.azimuth = 92
                self.env_vis.viewer.cam.lookat[:] = np.array([ 1.3350197899440235 , -0.30740768832559073 , 1.192416647672627  ])

            for frame_num in tqdm(range(len(v["pred"]))):
                self.update_pose()
                self.render()
                self.fr += 1
            self.end_video_recording()
            import gc
            gc.collect()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--num_threads", type=int, default=30)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--show_noise", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_log", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--data", type=str, default="proxd")
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
    args = parser.parse_args()

    cfg = Config(cfg_id=args.cfg, create_dirs=False)
    cfg.update(args)

    flags.debug = args.debug
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

    cfg.data_specs["train_files_path"] = [(
        '/hdd/zen/data/video_pose/prox/qualitative/singles/thirdeye_anns_proxd_single01.pkl',
        "scene_pose")]

    # cfg.data_specs["test_files_path"] = [('/hdd/zen/data/video_pose/prox/qualitative/thirdeye_anns_proxd_single12.pkl', "scene_pose")]
    # cfg.data_specs["test_files_path"] = [(
    #     '/hdd/zen/data/video_pose/prox/qualitative/thirdeye_anns_proxd_overlap.pkl',
    #     "scene_pose")]
    if cfg.data == "prox":
        cfg.data_specs["test_files_path"] = [(
            '/hdd/zen/data/video_pose/prox/qualitative/thirdeye_anns_prox_overlap_no_clip.pkl',
            "scene_pose")]
    elif cfg.data == "proxd":
        cfg.data_specs["test_files_path"] = [(
            '/hdd/zen/data/video_pose/prox/qualitative/thirdeye_anns_proxd_overlap_full.pkl',
            "scene_pose")]
    elif cfg.data == "prox_part":
        cfg.data_specs["test_files_path"] = [(
            '/hdd/zen/data/video_pose/prox/qualitative/thirdeye_anns_proxd_overlap.pkl',
            "scene_pose")]
    elif cfg.data == "prox_op":
        cfg.data_specs["test_files_path"] = [(
            '/hdd/zen/data/video_pose/prox/qualitative/thirdeye_anns_proxd_overlap_op.pkl',
            "scene_pose")]
    elif cfg.data == "h36m":
        cfg.data_specs["test_files_path"] = [
            ('/hdd/zen/data/video_pose/h36m/data_fit/h36m_test_30_fk.p',
             "scene_pose")
        ]
    elif cfg.data == "h36m_gt":
        cfg.data_specs["test_files_path"] = [
            ('/hdd/zen/data/video_pose/h36m/data_fit/h36m_test_30_gt_fk.p',
             "scene_pose")
        ]
    elif cfg.data == "h36m_part":
        cfg.data_specs["test_files_path"] = [(
            '/hdd/zen/data/video_pose/h36m/data_fit/h36m_test_30_fk_valid.p',
            "scene_pose")]
    elif cfg.data.startswith("prox"):
        cfg.data_specs["test_files_path"] = [(
            f"/hdd/zen/data/video_pose/prox/qualitative/singles/thirdeye_anns_proxd_single{cfg.data[-2:]}.pkl",
            "scene_pose")]
    elif cfg.data.startswith("amass"):
        cfg.data_specs["test_files_path"] = [(
            f"/hdd/zen/data/ActBound/AMASS/amass_copycat_take5_train.pkl",
            "amass")]
    elif cfg.data == "panotic":
        cfg.data_specs["test_files_path"] = [
            ('/hdd/zen/data/video_pose/panotic_go/smpl_processed_full.pkl',
             "scene_pose")
        ]

    if cfg.mode == "vis":
        cfg.num_threads = 1
    """create agent"""
    agent_class = agent_dict[cfg.agent_name]
    agent = agent_class(cfg=cfg,
                        dtype=dtype,
                        device=device,
                        checkpoint_epoch=cfg.epoch,
                        mode="test")

    if args.mode == "stats":
        agent.eval_policy(epoch=cfg.epoch, dump=True)
    elif args.mode == "disp_stats":
        vis = SceneVisulizer(
            agent.env.smpl_robot.export_vis_string().decode("utf-8"), agent)
        vis.render_coverage()
    else:
        vis = SceneVisulizer(
            agent.env.smpl_robot.export_vis_string().decode("utf-8"),
            agent,
        )

        vis.show_animation()
