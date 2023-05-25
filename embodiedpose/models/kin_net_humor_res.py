import glob
import os
import sys
import pdb
import os.path as osp
import json

sys.path.append(os.getcwd())
from sys import flags
from torch import nn
import torch
from collections import defaultdict
import joblib
import pickle
import time
import wandb
from tqdm import tqdm
import numpy as np
import math
from uhc.khrylib.utils import to_device, create_logger
from uhc.utils.flags import flags

from uhc.smpllib.torch_smpl_humanoid import Humanoid
from uhc.losses.loss_function import (
    compute_mpjpe_global,
    pose_rot_loss,
    root_pos_loss,
    root_orientation_loss,
    end_effector_pos_loss,
    linear_velocity_loss,
    angular_velocity_loss,
    action_loss,
    position_loss,
    orientation_loss,
    compute_error_accel,
    compute_error_vel,
)
from uhc.utils.torch_ext import gather_vecs
from uhc.utils.transformation import quaternion_about_axis, quaternion_matrix
import uhc.utils.pytorch3d_transforms as tR
from uhc.khrylib.models.rnn import RNN
from uhc.utils.torch_geometry_transforms import (angle_axis_to_rotation_matrix as aa2mat, rotation_matrix_to_angle_axis as mat2aa)
from uhc.models.kin_net_base import KinNetBase
from uhc.smpllib.smpl_mujoco import smpl_to_qpose, smpl_to_qpose_torch
from uhc.khrylib.models.mlp import MLP, MLPWithInputSkips

from embodiedpose.models.humor.utils.humor_mujoco import reorder_joints_to_humor
from embodiedpose.models.humor.humor_model import HumorModel
from embodiedpose.models.humor.utils.torch import load_state as load_humor_state
from embodiedpose.models.humor.torch_humor_loss import points3d_loss, kl_normal
from embodiedpose.models.humor.body_model.utils import smpl_to_openpose
from embodiedpose.models.humor.utils.velocities import estimate_velocities
from uhc.khrylib.rl.core.running_norm import RunningNorm
import copy
from embodiedpose.models.uhm_model import UHMModel
from embodiedpose.utils.scene_utils import load_simple_scene, get_sdf
from embodiedpose.models.humor.utils.humor_mujoco import SMPL_2_OP, MUJOCO_2_SMPL
from embodiedpose.models.kin_tcn import TemporalModel, TemporalModelOptimized1f
from uhc.utils.torch_ext import isNpArray

FPS = 30
PROX_SCENE = ['BasementSittingBooth', 'MPH11', 'MPH112', 'MPH16', 'MPH1Library', 'MPH8', 'N0SittingBooth', 'N0Sofa', 'N3Library', 'N3Office', 'N3OpenArea', 'Werkraum']


def smpl_op_to_op(pred_joints2d):
    new_2d = torch.cat([     pred_joints2d[..., [1, 4], :].mean(axis = -2, keepdims = True), \
                             pred_joints2d[..., 1:7, :], \
                             pred_joints2d[..., [7, 8, 11], :].mean(axis = -2, keepdims = True), \
                             pred_joints2d[..., 9:11, :], \
                             pred_joints2d[..., 12:, :]], \
                             axis = -2)
    return new_2d


class KinNetHumorRes(KinNetBase):
    # Kin net humor res
    def __init__(self, cfg, data_sample, device, dtype, agent, mode="train"):
        super(KinNetBase, self).__init__()
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.model_specs = cfg.model_specs
        self.mode = mode
        self.epoch = 0
        self.gt_rate = 0
        self.agent = agent
        self.sim = {}
        self.use_rnn = self.model_specs.get("use_rnn", False)

        self.motion_prior = UHMModel(in_rot_rep="mat", out_rot_rep=self.model_specs.get("out_rot_rep", "aa"), latent_size=24, model_data_config="smpl+joints", steps_in=1, use_gn=False)

        to_device(device, self.motion_prior)

        for param in self.motion_prior.parameters():
            param.requires_grad = False

        if self.model_specs.get("use_rvel", False):
            self.motion_prior.data_names.append("root_orient_vel")
            self.motion_prior.input_dim_list += [3]

        if self.model_specs.get("use_bvel", False):
            self.motion_prior.data_names.append("joints_vel")
            self.motion_prior.input_dim_list += [66]
        self.use_tcn = self.model_specs.get("use_tcn", False)
        if self.use_tcn:
            tcn_arch = self.model_specs.get("tcn_arch", "3,3,3")
            self.filter_widths = filter_widths = [int(x) for x in tcn_arch.split(',')]
            self.num_context = int(np.prod(filter_widths))

        self.humor_models = [HumorModel(in_rot_rep="mat", out_rot_rep="aa", latent_size=48, model_data_config="smpl+joints+contacts", steps_in=1, use_vposer=cfg.get("use_vposer", True))]
        load_humor_state("sample_data/humor/best_model.pth", self.humor_models[0], map_location=device)
        for param in self.humor_models[0].parameters():
            param.requires_grad = False
        to_device(device, self.humor_models[0])

        self.smpl2op_map = smpl_to_openpose(self.motion_prior.bm_dict['neutral'].model_type, use_hands=False, use_face=False, use_face_contour=False, openpose_format='coco25')
        self.smpl_2op_submap = self.smpl2op_map[self.smpl2op_map < 22]

        self.qpos_lm = 74  # TODO : get this from the SMPL model
        self.qvel_lm = 75
        self.pose_start = 7
        self.mlp_hsize = mlp_hsize = self.model_specs.get("mlp_hsize", [1024, 512])
        self.htype = htype = self.model_specs.get("mlp_htype", "relu")
        self.humor_keys = ['pose_body', 'root_orient', 'root_orient_vel', 'trans', 'trans_vel', 'joints', 'joints_vel']

        self.load_humanoid()
        data_sample = self.init_states(data_sample, random_cam=True)
        self.get_dim(data_sample)
        self.setup_logging()
        self.sept_root_mlp = self.model_specs.get("sept_root_mlp", False)
        self.use_mcp = self.model_specs.get("use_mcp", False)

        if self.use_rnn:
            rnn_hdim = self.model_specs.get("rnn_hdim", 128)

            if not self.sept_root_mlp:
                state_dim = np.sum(self.is_root_obs == 0) + np.sum(self.is_root_obs == 1)
                env_feat_dim = np.sum(self.is_root_obs == 2)
                env_net_dim = mlp_hsize[-1] if env_feat_dim > 0 else 0
                self.action_rnn = RNN(state_dim, rnn_hdim, "gru")
                self.action_rnn.set_mode("step")

                if self.use_mcp:
                    self.nets = nn.ModuleList()
                    num_primitive = 6

                    for i in range(num_primitive):
                        action_mean = nn.Linear(mlp_hsize[-1], self.body_action_dim)
                        action_mean.weight.data.mul_(0.1)
                        action_mean.bias.data.mul_(0.0)
                        net = nn.Sequential(*[MLP(rnn_hdim + state_dim, mlp_hsize, htype), action_mean])
                        self.nets.append(net)

                    self.composer = nn.Sequential(*[MLP(rnn_hdim + state_dim + env_net_dim, cfg.get("composer_dim", [512, 512]) + [num_primitive], htype), nn.Softmax(dim=1)])

                else:
                    self.action_mlp = MLP(rnn_hdim + state_dim + env_feat_dim, mlp_hsize, htype)
                    self.action_fc = nn.Linear(mlp_hsize[-1], self.body_action_dim)

                if self.model_specs.get("use_voxel", False):
                    self.env_mlp = MLP(env_feat_dim, mlp_hsize, htype)

            else:
                if self.model_specs.get("use_voxel", False):
                    root_feat_dim = np.sum(self.is_root_obs == 1)
                    body_feat_dim = np.sum(self.is_root_obs == 0)
                    env_feat_dim = np.sum(self.is_root_obs == 2)

                    self.action_rnn = RNN(root_feat_dim + body_feat_dim, rnn_hdim, "gru")
                    self.action_rnn.set_mode("step")
                    self.env_mlp = MLP(env_feat_dim, mlp_hsize, htype)

                    self.root_mlp = MLP(rnn_hdim + body_feat_dim + root_feat_dim + mlp_hsize[-1], mlp_hsize, htype)
                    self.body_mlp = MLP(rnn_hdim + body_feat_dim + root_feat_dim + mlp_hsize[-1], mlp_hsize, htype)
                    self.root_fc = nn.Linear(mlp_hsize[-1], 6)
                    self.body_fc = nn.Linear(mlp_hsize[-1], 63)
                else:
                    self.action_rnn = RNN(self.state_dim, rnn_hdim, "gru")
                    self.action_rnn.set_mode("step")
                    self.root_mlp = MLP(rnn_hdim + self.state_dim, mlp_hsize, htype)
                    self.body_mlp = MLP(rnn_hdim + self.state_dim, mlp_hsize, htype)
                    self.root_fc = nn.Linear(mlp_hsize[-1], 6)
                    self.body_fc = nn.Linear(mlp_hsize[-1], 63)

            if self.use_tcn:
                if self.model_specs.get("tcn_3dpos", False):
                    self.traj_in_features = traj_in_features = 5
                    if self.model_specs.get("tcn_body2d", False):
                        self.body_in_features = body_in_features = 2
                    else:
                        self.body_in_features = body_in_features = 5
                else:
                    self.traj_in_features = traj_in_features = 2
                    self.body_in_features = body_in_features = 2

                casual = self.model_specs.get("casual_tcn", True)

                self.tcn_body_1f = TemporalModelOptimized1f(num_joints_in=12, in_features=body_in_features, num_joints_out=14, filter_widths=filter_widths, causal=casual)
                self.tcn_traj_1f = TemporalModelOptimized1f(num_joints_in=12, in_features=traj_in_features, num_joints_out=1, filter_widths=filter_widths, causal=casual)
                self.tcn_body_1f.float(), self.tcn_traj_1f.float()

                self.tcn_opt = torch.optim.Adam(list(self.tcn_body_1f.parameters()) + list(self.tcn_traj_1f.parameters()), lr=0.0005)

        else:
            self.action_mlp = MLP(self.state_dim, mlp_hsize, htype)
            self.action_fc = nn.Linear(mlp_hsize[-1], self.body_action_dim)

        self.norm = RunningNorm(self.state_dim)
        self.setup_optimizer()
        print("**********************************************************")
        print(self.model_specs)
        print("**********************************************************")

    def set_sim(self, humor_dict):
        self.sim = humor_dict

    def get_action(self, state):
        if self.use_rnn:
            state_norm = self.norm(state)
            if not self.sept_root_mlp:

                agent_state = state_norm[:, np.logical_or(self.is_root_obs == 0, self.is_root_obs == 1)[0]]
                env_state = state_norm[:, (self.is_root_obs == 2)[0]]
                rnn_out = self.action_rnn(agent_state)
                state_norm = torch.cat((agent_state, rnn_out), dim=1)

                if self.model_specs.get("use_voxel", False):
                    env_net_res = self.env_mlp(env_state)
                    state_norm_env = torch.cat([state_norm, env_net_res], dim=-1)
                else:
                    state_norm_env = state_norm

                if self.use_mcp:
                    x_all = torch.stack([net(state_norm) for net in self.nets], dim=1)
                    weight = self.composer(state_norm_env)
                    action = torch.sum(weight[:, :, None] * x_all, dim=1)
                else:
                    state_norm = self.action_mlp(state_norm)
                    action = self.action_fc(state_norm)
            else:
                if self.model_specs.get("use_voxel", False):
                    root_feat = state_norm[:, (self.is_root_obs == 1)[0]]
                    body_feat = state_norm[:, (self.is_root_obs == 0)[0]]
                    env_feat = state_norm[:, (self.is_root_obs == 2)[0]]

                    rnn_out = self.action_rnn(torch.cat([root_feat, body_feat], dim=1))
                    env_feat = self.env_mlp(env_feat)

                    root_x = torch.cat((env_feat, root_feat, body_feat, rnn_out), dim=1)
                    root_x = self.root_mlp(root_x)
                    root_action = self.root_fc(root_x)

                    body_x = torch.cat((env_feat, root_feat, body_feat, rnn_out), dim=1)
                    body_x = self.body_mlp(body_x)
                    body_action = self.body_fc(body_x)
                    action = torch.cat([root_action, body_action], dim=1)
                else:
                    rnn_out = self.action_rnn(state_norm)

                    root_x = torch.cat((state_norm, rnn_out), dim=1)
                    root_x = self.root_mlp(root_x)
                    root_action = self.root_fc(root_x)

                    body_x = torch.cat((state_norm, rnn_out), dim=1)
                    body_x = self.body_mlp(body_x)
                    body_action = self.body_fc(body_x)
                    action = torch.cat([root_action, body_action], dim=1)

            if self.use_tcn:
                B = state.shape[0]
                kp_2d_feats = state[:, (self.is_root_obs == 3)[0]]  # TCN has its own norm, no need for norm here
                if self.mode == "test":
                    if self.model_specs.get("tcn_3dpos", False):
                        # ZL: may have bugs here. need to check
                        kp_2d_feats = kp_2d_feats.reshape(B, self.num_context, 12, 5).float()
                        kp_2d_feats_local = kp_2d_feats.clone()
                        kp_2d_feats_local[..., 2:] = kp_2d_feats[..., 2:] - kp_2d_feats[..., 7:8, 2:]
                        traj_pred = self.tcn_traj_1f(kp_2d_feats)
                        if self.model_specs.get("tcn_body2d", False):
                            body_pos_pred = self.tcn_body_1f(kp_2d_feats_local[:, :, :, :2])
                        else:
                            body_pos_pred = self.tcn_body_1f(kp_2d_feats_local)
                    else:
                        traj_pred = self.tcn_traj_1f(kp_2d_feats.reshape(B, self.num_context, 12, 2).float())
                        body_pos_pred = self.tcn_body_1f(kp_2d_feats.reshape(B, self.num_context, 12, 2).float())

                else:
                    traj_pred = torch.zeros([B, 1, 3]).to(state)
                    body_pos_pred = torch.zeros([B, 14, 3]).to(state)

                action = torch.cat([action, traj_pred.reshape(B, -1), body_pos_pred.reshape(B, -1)], dim=1)
        else:
            state_norm = self.norm(state)
            # x = state
            state_norm = self.action_mlp(state_norm)
            action = self.action_fc(state_norm)
        return action

    def get_dim(self, data_dict):
        # pose_aa_curr = data_dict[f"pose_aa"][:, 0, :]
        # trans_curr = data_dict[f"trans"][:, 0, :]
        self.is_root_obs = []
        humor_dict = {k: data_dict[k][:, 0:1, :].clone() for k in self.motion_prior.data_names}
        self.set_sim(humor_dict)

        state, _ = self.get_obs(data_dict, 0)
        self.is_root_obs = state
        self.state_dim = state.shape[-1]

        self.action_dim = np.sum(self.motion_prior.output_dim_list)
        self.body_action_dim = np.sum(self.motion_prior.output_dim_list)

        if self.use_tcn:
            self.action_dim += 14 * 3 + 3

        # self.action_dim = 141

        self.context_dim = self.get_context_dim(data_dict)

    def step(self, action):

        next_global_out = self.motion_prior.step_state(self.sim, action)

        B = next_global_out['pose_body'].size(0)

        body_pose_aa = mat2aa(next_global_out['pose_body'].reshape(B * 21, 3, 3)).reshape(B, 63)
        root_aa = mat2aa(next_global_out['root_orient'].reshape(B, 3, 3)).reshape(B, 3)

        pose_aa = torch.cat([root_aa, body_pose_aa, torch.zeros(B, 6).to(root_aa)], dim=1)

        qpos = smpl_to_qpose_torch(pose_aa, self.model, trans=next_global_out['trans'].reshape(B, 3), count_offset=True)
        out_len = qpos.size(0)
        fk_result = self.humanoid.qpos_fk(qpos, to_numpy=False, full_return=False)
        joints = reorder_joints_to_humor(fk_result["wbpos"].clone(), self.model, self.cfg.cc_cfg.robot_cfg.get("model", "smpl"))[:, :66]
        trans_vel, joints_vel, root_orient_vel = estimate_velocities(torch.cat([self.sim['trans'], next_global_out['trans']], dim=1).reshape(B, -1, 3),
                                                                     torch.cat([self.sim['root_orient'], next_global_out['root_orient']], dim=1).reshape(B, -1, 1, 3, 3),
                                                                     torch.cat([self.sim['joints'], joints[:, None, :]], dim=1).reshape(B, -1, 22, 3),
                                                                     FPS,
                                                                     aa_to_mat=False)

        next_global_out['joints'] = joints.reshape(B, -1, 66)
        next_global_out['trans_vel'] = trans_vel[:, 0:1,]
        next_global_out['joints_vel'] = joints_vel[:, 0:1,]
        next_global_out['root_orient_vel'] = root_orient_vel[:, 0:1,]
        return next_global_out

    def get_sim_joints(self, data_dict):
        B, seq_len, _ = data_dict["pose_aa"].shape
        pose_aa_mat = self.sim['pose_body']
        root_orient = self.sim['root_orient']
        trans = self.sim['trans'].reshape(B, 3)
        beta = data_dict['beta'][:, 0:1].reshape(B, -1)
        pose_aa = mat2aa(pose_aa_mat.reshape(B * 21, 3, 3)).reshape(B, 63)
        root_aa = mat2aa(root_orient.reshape(B * 1, 3, 3)).reshape(B, 3)

        pred_body = self.motion_prior.bm_dict['neutral'](pose_body=pose_aa.float(), pose_hand=None, betas=beta.float(), root_orient=root_aa.float(), trans=trans.float())
        return pred_body

    def get_obs(self, data_dict, t):
        # obs = torch.zeros(1, 447)
        # 1: root 0: body 2: scene
        B, T, _ = data_dict['trans'].shape
        obs = []
        obs.append(np.array([1]))  # hq
        obs.append(np.concatenate([[1 if "root" in k else 0] * data_dict[k].reshape(B, T, -1).shape[-1] for k in self.motion_prior.data_names]))

        if self.model_specs.get("use_contact", False):
            obs.append(np.array([0] * 24))

        if self.model_specs.get("use_tcn", False):
            if self.model_specs.get("tcn_3dpos", False):
                obs.append(np.array([3] * (self.num_context * 12 * 5)))
            else:
                obs.append(np.array([3] * (self.num_context * 12 * 2)))

            if self.cfg.model_specs.get("tcn_body", False):
                obs.append(np.array([0] * 14 * 3))

            if self.cfg.model_specs.get("tcn_traj", False):
                obs.append(np.array([1] * 3))  # Translation

            if not self.cfg.model_specs.get("tcn_root_grad", False):
                obs.append(np.array([1] * 9))  # Rotation

        if self.model_specs.get("use_rt", False):
            obs.append(np.array([1] * 3))

        if self.model_specs.get("use_rr", False):
            obs.append(np.array([1] * 9))

        if self.model_specs.get("use_3d_grad", False) or self.model_specs.get("use_3d_grad_adpt", False):
            # obs.append(np.array([1] * 3))
            # obs.append(np.array([1] * 3))
            # obs.append(np.array([0] * 63))
            obs.append(np.array([1] * 3))
            obs.append(np.array([1] * 3))
            obs.append(np.array([1] * 63))

        elif self.model_specs.get("use_3d_grad_sept", False):

            obs.append(np.array([1] * 3))
            obs.append(np.array([1] * 63))

        if self.model_specs.get("use_sdf", False):
            obs.append(np.array([1] * 66))
        elif self.model_specs.get("use_dir_sdf", False):
            obs.append(np.array([2] * 198))

        if self.model_specs.get("use_voxel", False):
            voxel_res = self.model_specs.get("voxel_res", 8)
            obs.append(np.array([2] * (voxel_res**3)))

        obs = np.concatenate(obs)[None,]

        return obs, self.sim

    def sample_valid_cam_x_y(self, cam_params, trans, offset):
        trans = trans.copy()
        K, full_R, full_t = cam_params['K'], cam_params['full_R'], cam_params['full_t']

        lower_left = np.array([[100, 0, 1], [100, 1080, 1], [1820, 0, 1], [1820, 1080, 1]])
        ptr = lower_left.dot(np.linalg.inv(K).T)

        heading_change = quaternion_matrix(quaternion_about_axis(np.random.rand() * np.pi * 2, [0, 0, 1]))[:3, :3]
        new_trans = (np.matmul(trans + offset, heading_change.T) - offset).copy()
        z = np.random.uniform(1.5, 6)
        frame_ptrs = ptr * z
        x, y = np.random.uniform(np.min(frame_ptrs[:, 0]), np.max(frame_ptrs[:, 0])), np.random.uniform(np.min(frame_ptrs[:, 1]), np.max(frame_ptrs[:, 2]))

        new_ptr = (np.array([[x, y, z]]) - full_t).dot(full_R)
        x, y = new_ptr[0, 0], new_ptr[0, 1]
        new_trans[:, :, :2] += np.array([x, y]) - new_trans[:, 0:1, :2]

        trans_cam = ((new_trans + offset).dot(full_R.T) + full_t)
        root_2d = trans_cam.dot(K.T).copy()
        root_2d /= root_2d[:, :, 2:3]
        num_trys = 1

        while np.isnan(root_2d).any() or np.sum(root_2d < 0) > 0 or np.sum(trans_cam[..., 2] < 1) > 0 or np.sum(root_2d[..., 0] > 2200) or np.sum(root_2d[..., 1] > 1200):  # should be at least 1.5 m from the camera and within the camera frame
            heading_change = quaternion_matrix(quaternion_about_axis(np.random.rand() * np.pi * 2, [0, 0, 1]))[:3, :3]
            new_trans = (np.matmul(trans + offset, heading_change.T) - offset).copy()
            z = np.random.uniform(1.5, 6)
            frame_ptrs = ptr * z
            x, y = np.random.uniform(np.min(frame_ptrs[:, 0]), np.max(frame_ptrs[:, 0])), np.random.uniform(np.min(frame_ptrs[:, 1]), np.max(frame_ptrs[:, 2]))
            new_ptr = (np.array([[x, y, z]]) - full_t).dot(full_R)
            x, y = new_ptr[0, 0], new_ptr[0, 1]
            new_trans[:, :, :2] += np.array([x, y]) - new_trans[:, 0:1, :2]  # New trans in the world space
            trans_cam = ((new_trans + offset).dot(full_R.T) + full_t)
            root_2d = trans_cam.dot(K.T)
            root_2d /= root_2d[:, :, 2:3]
            num_trys += 1
            if num_trys > 10:
                print(f"!!!!!!!! stuck at {self.seq_name}")
                self.scene_name = np.random.choice(PROX_SCENE)
                self.load_camera_params(self.scene_name)
                K, full_R, full_t = self.cam_params['K'], self.cam_params['full_R'], self.cam_params['full_t']
        return x, y, heading_change

    def init_states(self, data_dict, random_cam=False):
        data_dict = copy.deepcopy(data_dict)
        B, T, _ = data_dict["pose_aa"].shape  #
        # device = data_dict["pose_aa"].device
        device = self.device

        self.seq_name = seq_name = data_dict['seq_name'][0] if isinstance(data_dict['seq_name'], list) else data_dict['seq_name']

        betas = data_dict['beta'][0]
        if data_dict['gender'][0, 0] == 0:
            gender = "neutral"
        elif data_dict['gender'][0, 0] == 1:
            gender = 'male'
        elif data_dict['gender'][0, 0] == 2:
            gender = "female"
        data_dict['load_scene'] = True
        if random_cam:
            # data_dict['load_scene'] = False
            if betas.shape[-1] == 10:
                betas = torch.cat([betas, torch.zeros(betas.shape[0], 6).to(betas)], dim=1).to(self.device)
                data_dict['beta'] = betas.reshape(B, T, 16)

            pose_aa_empty = data_dict['pose_aa'].reshape(T, -1, 3)[0:1].to(self.device)
            self.motion_prior.bm_dict[gender].to(self.device)
            pre_body = self.motion_prior.bm_dict[gender](pose_body=pose_aa_empty[:, 1:22].float().reshape(1, -1), pose_hand=None, betas=betas[0:1].float(), root_orient=pose_aa_empty[:, 0:1].float().reshape(1, -1))
            offset = pre_body.Jtr[0, 0]

            self.scene_name = np.random.choice(PROX_SCENE)
            self.load_camera_params(self.scene_name)

            if np.random.uniform() < 0.5 or not self.model_specs.get("load_scene", True):
                data_dict['load_scene'] = False  # sometimes, don't load any scene

            curr_trans = data_dict['trans'].cpu().numpy().copy()
            x, y, heading_change = self.sample_valid_cam_x_y(self.cam_params, curr_trans, offset.cpu().numpy())

            if "obj_info" in data_dict:
                heading_change = torch.from_numpy(heading_change).to(device).type(self.dtype)
                obj_info = data_dict['obj_info']
                data_dict['trans'] = torch.matmul(data_dict['trans'] + offset, heading_change.T) - offset  ## Offset strikes again!!!
                new_root = torch.matmul(heading_change, data_dict['root_orient'].reshape(-1, 3, 3))
                data_dict['pose_aa'][:, :, :3] = tR.matrix_to_axis_angle(new_root)
                data_dict['root_orient'] = new_root.reshape(B, T, 9)

                for k, v in obj_info.items():
                    obj_pos_orig = v['obj_pose'][:, :3].copy()
                    v['obj_pose'][:, 3:] = tR.matrix_to_quaternion(torch.matmul(heading_change, tR.quaternion_to_matrix(torch.from_numpy(v['obj_pose'][:, 3:])).to(device))).cpu().numpy()  # Rotation
                    v['obj_pose'][:, :3] = np.dot(v['obj_pose'][:, :3], heading_change.cpu().numpy().T)  # Translation

                    v['obj_pose'][:, 0] += (x - data_dict['trans'][:, 0, 0]).cpu().numpy()
                    v['obj_pose'][:, 1] += (y - data_dict['trans'][:, 0, 1]).cpu().numpy()

                data_dict['trans'][:, :, 0] += (x - data_dict['trans'][:, 0, 0])
                data_dict['trans'][:, :, 1] += (y - data_dict['trans'][:, 0, 1])  # This has to be indented.

            else:
                if data_dict['load_scene']:
                    scene_sdfs, obj_pos = load_simple_scene(self.scene_name)

                    dist = get_sdf(scene_sdfs=scene_sdfs, points=torch.tensor([[x, y, 0.5], [x, y, 0], [x, y, 1], [x, y, 1.5], [x, y, 0.3], [x, y, 0.7]]), topk=1)

                    while not torch.min(dist) > 0.2:

                        x, y, heading_change = self.sample_valid_cam_x_y(self.cam_params, curr_trans, offset.cpu().numpy())
                        dist = get_sdf(scene_sdfs=scene_sdfs, points=torch.tensor([[x, y, 0.5], [x, y, 0], [x, y, 1], [x, y, 1.5], [x, y, 0.3], [x, y, 0.7]]), topk=1)

                heading_change = torch.from_numpy(heading_change).to(device).type(self.dtype)

                new_root = torch.matmul(heading_change, data_dict['root_orient'].reshape(-1, 3, 3))

                data_dict['trans'] = torch.matmul(data_dict['trans'] + offset, heading_change.T) - offset  ## Offset strikes again!!!
                data_dict['pose_aa'][:, :, :3] = tR.matrix_to_axis_angle(new_root)
                data_dict['root_orient'] = new_root.reshape(B, T, 9)

                data_dict['trans'][:, :, 0] += (x - data_dict['trans'][:, 0, 0])
                data_dict['trans'][:, :, 1] += (y - data_dict['trans'][:, 0, 1])

            data_dict['cam'] = self.cam_params
            camera_params_torch = {k: torch.from_numpy(v).float().to(device) if isNpArray(v) else v for k, v in self.cam_params.items()}

            # Always compute new joints since the rotation is now changed....
            pose_mat = torch.cat([data_dict['root_orient'].reshape(B, T, -1), data_dict['pose_body'].reshape(B, T, -1), torch.eye(3).flatten()[None, None,].repeat(B, T, 2).to(device)], dim=2)

            self.human_b.update_model(data_dict['beta'][:, 0], data_dict['gender'][:, 0])
            fk_dict = self.human_b.fk_batch(pose_mat, data_dict['trans'].clone(), convert_to_mat=False)

            gt_joints = reorder_joints_to_humor(fk_dict['wbpos'].reshape(B, T, -1), self.model, model="smpl")[:, :66]
            data_dict['joints'] = gt_joints.reshape(B, T, -1)
            ## ZL: Assuming alwyas single batch here. For different gender, need to update the code and swtich
            trans_vel, joints_vel, root_orient_vel = estimate_velocities(data_dict['trans'], data_dict['root_orient'].reshape(B, T, 1, 3, 3), data_dict['joints'].reshape(B, T, 22, 3), 30, aa_to_mat=False)
            data_dict['trans_vel'] = trans_vel
            data_dict['joints_vel'] = joints_vel
            data_dict['root_orient_vel'] = root_orient_vel

            K, full_R, full_t = camera_params_torch['K'], camera_params_torch['full_R'], camera_params_torch['full_t']

            wbpos = fk_dict['wbpos'].reshape(B, T, 24, 3)[:, :, MUJOCO_2_SMPL].clone().float()
            wbpos_cam = wbpos @ full_R.T + full_t

            wbpos_cam_pic = wbpos_cam @ (K.T)[None].clone()
            z_norm = wbpos_cam_pic[:, :, :, 2:]
            wbpos2d = wbpos_cam_pic / z_norm
            pred_joints2d = wbpos2d[:, :, self.smpl2op_map[self.smpl2op_map < 22]]
            wbpos_cam = wbpos_cam[:, :, self.smpl2op_map[self.smpl2op_map < 22]]

            if self.mode == "train":
                noise_mul = self.model_specs.get('noise_mul', 1.0)
                pred_joints2d[:, :, :, 2:3] = torch.rand(pred_joints2d[:, :, :, 2:3].shape) * noise_mul  # Need to better handle occlusion

                # pred_joints2d[:, :, :, :2] += torch.randn(pred_joints2d[:, :, :, :2].shape).to(self.device)

            data_dict['joints2d'] = smpl_op_to_op(pred_joints2d)
            data_dict['wbpos_cam'] = wbpos_cam
            data_dict['points3d'] = torch.zeros(B, T, 4096, 3)
            data_dict['root_cam_rot'] = tR.matrix_to_axis_angle(torch.matmul(data_dict['root_orient'].float().reshape(-1, 3, 3), full_R.T)).reshape(B, T, 1, 3)
        else:
            if not 'cam' in data_dict:
                print("no camera in data!!!")
                raise Exception()
            self.scene_name = data_dict['cam']['scene_name']
            self.cam_params = data_dict['cam']
            camera_params_torch = {k: torch.from_numpy(v).float().to(device) if isNpArray(v) else v for k, v in self.cam_params.items()}

        if not "wbpos_cam" in data_dict:
            # For loading the prox dataset.
            full_R, full_t = camera_params_torch['full_R'], camera_params_torch['full_t']
            wbpos = data_dict['joints'].reshape(B, T, 22, 3)[:, :, self.smpl2op_map[self.smpl2op_map < 22]]
            wbpos_cam = wbpos.float() @ full_R.T + full_t
            data_dict['wbpos_cam'] = wbpos_cam

            data_dict['root_cam_rot'] = tR.matrix_to_axis_angle(torch.matmul(data_dict['root_orient'].float().reshape(-1, 3, 3), full_R.T)).reshape(B, T, 1, 3)
        humor_dict = {k: data_dict[k][:, 0:1, :].clone() for k in self.motion_prior.data_names}
        self.set_sim(humor_dict)

        data_dict["init_pose_aa"] = data_dict[f"pose_aa"][:, 0, :]
        data_dict["init_trans"] = data_dict[f"trans"][:, 0, :]

        self.reload_humanoid(beta=data_dict['beta'].squeeze(), gender=[0])
        data_dict['scene_name'] = self.scene_name
        return data_dict

    def forward(self, data_dict):
        # pose: 69 dim body pose
        # feat_names = ['trans', "root_orient", "pose_body", "phase"]
        # feat_names = ['trans', "root_orient", "pose_body"]
        feat_names = ['trans', "root_orient", "pose_body", "time"]
        # feat_names = ['trans', "root_orient", "pose_body", 'trans', "root_orient", "pose_body", "time"]

        batch_size, seq_len, _ = data_dict["pose_aa"].shape  #
        res_init = self.init_states(data_dict)
        # input_dict = [data_dict[k][:, :, :].clone().reshape(batch_size, (seq_len ), -1) for k in feat_names]
        input_dict = [data_dict[k][:, :-1, :].clone().reshape(batch_size, (seq_len - 1), -1) for k in feat_names]

        input_vec = torch.cat(input_dict, dim=2)

        if self.mode == "train":
            input_vec = input_vec + torch.randn(input_vec.shape).to(input_vec) * 0.01
        action = self.get_action(input_vec.squeeze())
        humor_dict = {k: data_dict[k][:, :-1, :].reshape(seq_len - 1, 1, -1).clone() for k in self.motion_prior.data_names}
        self.set_sim(humor_dict)
        next_global_out = self.motion_prior.step_state(self.sim, action)
        B = next_global_out['pose_body'].size(0)
        # root_aa_t_1 = mat2aa(self.sim['root_orient'].reshape(-1, 3, 3)).reshape(B, -1, 3)
        # root_aa_t = mat2aa(next_global_out['root_orient'].reshape(-1, 3, 3)).reshape(B, -1, 3)

        body_pose_aa = mat2aa(next_global_out['pose_body'].reshape(B * 21, 3, 3)).reshape(B, 63)
        root_aa = mat2aa(next_global_out['root_orient'].reshape(B, 3, 3)).reshape(B, 3)
        pose_aa = torch.cat([root_aa, body_pose_aa, torch.zeros(B, 6).to(root_aa)], dim=1)

        qpos = smpl_to_qpose_torch(pose_aa, self.model, trans=next_global_out['trans'].reshape(B, 3), count_offset=True)
        out_len = qpos.size(0)
        fk_result = self.humanoid.qpos_fk(qpos, to_numpy=False)
        joints = reorder_joints_to_humor(fk_result["wbpos"].clone(), self.model, self.cfg.cc_cfg.robot_cfg.get("model", "smpl"))[:, :66]

        trans_vel, joints_vel, root_orient_vel = estimate_velocities(next_global_out['trans'].reshape(1, (out_len), 3), next_global_out['root_orient'].reshape(1, (out_len), 3, 3), joints.reshape(1, (out_len), 22, 3), FPS, aa_to_mat=False)
        next_global_out['trans_vel'] = trans_vel
        next_global_out['joints_vel'] = joints_vel
        next_global_out['root_orient_vel'] = root_orient_vel
        next_global_out['joints'] = joints

        feature_res = {k: v.reshape(1, (out_len), 1, -1) for k, v in next_global_out.items()}
        feature_res['action'] = action[None,]

        feature_res = {k: torch.cat([feature_res[k][:, 0:1], v], dim=1) for k, v in feature_res.items()}

        return feature_res

    def motion_prior_loss(self, z, pm, pv):
        log_prob = -torch.log(torch.sqrt(pv)) - math.log(math.sqrt(2 * math.pi)) - ((z - pm)**2 / (2 * pv))
        log_prob = -torch.sum(log_prob, dim=-1)
        return torch.mean(log_prob)

    def load_camera_params(self, scene_name):
        prox_path = self.cfg.data_specs['prox_path']

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

        cam_params = {"K": K, "full_R": full_R, "full_t": full_t, "img_w": 1980, "img_h": 1080, "scene_name": scene_name}
        self.cam_params = cam_params

    def compute_loss_seq(self, feature_pred, data_dict, epoch=0, max_epoch=100):

        # pred_feats = self.motion_prior.dict2vec_input(feature_pred)
        # humor_feats = self.motion_prior.dict2vec_input(data)
        cam_params = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in self.cam_params.items()}
        K, full_R, full_t = cam_params["K"], cam_params["full_R"], cam_params["full_t"]

        B, seq_len, nq = feature_pred["trans"].shape
        weights = self.model_specs.get("weights", {})
        indices, indices_end = data_dict['indices'], data_dict['indices_end']

        # if self.agent.iter >= 50:
        #     rewards = data_dict['rewards']
        #     gamma = 0.01
        #     weighting = -(1 - rewards)**gamma * torch.log(rewards)
        #     weighting = torch.nan_to_num(weighting, posinf=0, neginf=0)
        #     weighting /= torch.max(
        #         torch.nan_to_num(weighting, posinf=0, neginf=0))

        # else:
        #     weighting = 1

        weighting = 1

        ######################################## SMPL Joints ########################################
        if weights.get("loss_2d", 0) > 0 or weights.get("loss_chamfer", 0) > 0:
            gt_points3d = data_dict["points3d"].reshape(B, seq_len, -1, 3)
            pose_aa_mat = feature_pred['pose_body'].reshape(B * seq_len, -1, 3, 3)
            beta = data_dict['beta'].reshape(B * seq_len, -1)
            root_orient = feature_pred['root_orient'].reshape(B * seq_len, -1, 3, 3)
            trans = feature_pred['trans'].reshape(B * seq_len, 3)

            pose_aa = mat2aa(pose_aa_mat.reshape(B * seq_len * 21, 3, 3)).reshape(B * seq_len, 63)
            root_aa = mat2aa(root_orient.reshape(B * seq_len * 1, 3, 3)).reshape(B * seq_len, 3)

            pred_body = self.motion_prior.bm_dict['neutral'](pose_body=pose_aa.float(), pose_hand=None, betas=beta.float(), root_orient=root_aa.float(), trans=trans.float())

        if weights.get("loss_chamfer", 0) > 0:
            pred_verts = pred_body.v.reshape(B, seq_len, -1, 3).double()
            pred_verts = pred_verts @ aR.T + atr
            pred_verts = pred_verts @ R.T + tr
            import ipdb
            ipdb.set_trace()

            loss_chamfer = points3d_loss(gt_points3d, pred_verts, mean=False)
            loss_chamfer = gather_vecs([loss_chamfer], indices)[0].mean()

        if weights.get("loss_2d", 0) > 0:
            pred_joints3d = pred_body.Jtr
            pred_joints3d = pred_joints3d.reshape(B * seq_len, -1, 3).double()
            pred_joints3d = pred_joints3d[:, self.smpl2op_map, :]
            pred_joints3d = pred_joints3d @ full_R.T + full_t
            pred_joints2d = pred_joints3d @ (K.T)
            z = pred_joints2d[:, :, 2:]
            pred_joints2d = pred_joints2d[:, :, :2] / z

            # pred_joints2d = feature_pred['pred_j2d'].reshape(b_size * seq_len, -1, 2)
            joints2d_gt = data_dict['joints2d'].reshape(B * seq_len, -1).double()
            pred_joints2d = pred_joints2d.reshape(B * seq_len, -1)
            joints2d_gt, pred_joints2d = gather_vecs([joints2d_gt, pred_joints2d], indices)
            joints2d_gt, pred_joints2d = joints2d_gt.reshape(-1, 25, 3), pred_joints2d.reshape(-1, 25, 2)

            inliers = joints2d_gt[:, :, 2] > 0.5
            pred_joints2d_inliders = pred_joints2d[inliers]
            joints2d_gt = joints2d_gt[inliers][:, :2]
            loss_2d = torch.norm(pred_joints2d_inliders - joints2d_gt, dim=1).mean()
        ######################################## SMPL Joints ########################################
        if weights.get("prior_loss", 0) > 0:
            B_seq, T_seq, _ = feature_pred['trans'].shape
            past_global_in = {k: feature_pred[k].reshape(B_seq, T_seq, -1)[:, :-1] for k in self.humor_models[0].data_names}
            next_global_in = {k: feature_pred[k].reshape(B_seq, T_seq, -1)[:, 1:] for k in self.humor_models[0].data_names}

            # past_in, next_out = self.motion_prior.canonicalize_input_double(past_global_in, next_global_in, split_input=False, cam_params=cam_params)
            past_in, next_out = self.humor_models[0].canonicalize_input_double(past_global_in, next_global_in, split_input=False)

            past_in, next_out = torch.cat([past_in, past_in[:, -2:-1]], dim=1), torch.cat([next_out, next_out[:, -2:-1]], dim=1)  # padding such that we can gather

            past_in, next_out = gather_vecs([past_in, next_out], indices_end)

            pm, pv = self.humor_models[0].prior(past_in[1:])
            qm, qv = self.humor_models[0].posterior(past_in[1:], next_out[1:])
            prior_loss = kl_normal(qm, qv, pm, pv)
            prior_loss = prior_loss.mean()

        if weights.get("l1_loss", 0) > 0:
            use_names = ['trans', 'root_orient']
            root_feat_pred = torch.cat([feature_pred[k].reshape([B, seq_len, -1]).clone() for k in use_names], dim=2).reshape(B * seq_len, -1)
            root_humor_gt = torch.cat([data_dict[k].reshape([B, seq_len, -1]).clone() for k in use_names], dim=2).reshape(B * seq_len, -1)
            ############################################################################
            use_names = ['pose_body', "joints"]
            body_feat_pred = torch.cat([feature_pred[k].reshape([B, seq_len, -1]).clone() for k in use_names], dim=2).reshape(B * seq_len, -1)

            body_humor_gt = torch.cat([data_dict[k].reshape([B, seq_len, -1]).clone() for k in use_names], dim=2).reshape(B * seq_len, -1)
            #############################################################################
            use_names = ['root_orient_vel', "joints_vel"]
            vel_pred = torch.cat([feature_pred[k].reshape([B, seq_len, -1]).clone() for k in use_names], dim=2).reshape(B * seq_len, -1)

            vel_gt = torch.cat([data_dict[k].reshape([B, seq_len, -1]).clone() for k in use_names], dim=2).reshape(B * seq_len, -1)

            l1_loss = ((root_feat_pred - root_humor_gt).abs()).mean(dim=1) * self.model_specs.get("root_l1_mul", 0.1) + ((body_feat_pred - body_humor_gt).abs()).mean(dim=1) + ((vel_pred - vel_gt).abs()).mean(dim=1) * self.model_specs.get("vel_l1_mul", 0.5)

            # l1_loss = ((root_feat_pred - root_humor_gt).abs()).mean(dim=1) * self.model_specs.get("root_l1_mul", 1) + ((body_feat_pred - body_humor_gt).abs()).mean(dim=1)
            # l1_loss = ((root_feat_pred - root_humor_gt).abs()).mean(dim=1) * self.model_specs.get("root_l1_mul", 1)

            l1_loss = l1_loss[:, None] * weighting
            l1_loss = gather_vecs([l1_loss], indices)[0].mean()

        if weights.get("l1_loss_local", 0) > 0:
            #############################################################################
            use_names = ['pose_body', "joints"]
            joints_pred = feature_pred["joints"].reshape(B, seq_len, 22, 3)
            pose_pred = feature_pred["pose_body"].reshape(B, seq_len, 21, 3, 3)
            joints_tgt = data_dict["joints"].reshape(B, seq_len, 22, 3)
            pose_tgt = data_dict["pose_body"].reshape(B, seq_len, 21, 3, 3)
            diff_joints = torch.abs((joints_pred - joints_pred[..., 0:1, :]) - (joints_tgt - joints_tgt[..., 0:1, :])).reshape(B * seq_len, -1).mean(-1)
            diff_pose = ((pose_pred - pose_tgt).abs()).reshape(B * seq_len, -1).mean(-1)
            l1_loss_local = (diff_joints + diff_pose)

            l1_loss_local = l1_loss_local[:, None] * weighting
            l1_loss_local = gather_vecs([l1_loss_local], indices)[0].mean()

        if weights.get("l1_loss_dyna", 0) > 0:
            use_names = ['trans', 'root_orient', 'pose_body', "joints"]
            sequential_sim = data_dict['sim_humor_dict']

            sequential_sim_vec = torch.cat([sequential_sim[k][:, :].reshape(B_seq * T_seq, -1) for k in use_names], dim=1)

            sequential_out_vec = torch.cat([feature_pred[k][:, :].reshape(B_seq * T_seq, -1) for k in use_names], dim=1)

            l1_loss_dyna = ((sequential_sim_vec - sequential_out_vec)**2).mean(1)[:, None]
            l1_loss_dyna = gather_vecs([l1_loss_dyna], indices)[0].mean()

        total_loss = 0
        loss_dict = {}
        loss_unweighted_dict = {}
        for k, v in weights.items():
            if k in locals():
                loss = eval(k) * v
                total_loss += loss

                loss_dict[k] = loss.detach().cpu().item()
                loss_unweighted_dict[k + "-uw"] = eval(k).detach().cpu().item()
        # if flags.debug:
        #     print(loss_dict, loss_unweighted_dict)
        return total_loss, loss_dict, loss_unweighted_dict

    def compute_loss_init(self, feature_pred, data):
        return torch.tensor([0]).to(self.device), {}, {}

    def compute_loss_lite(self, pred_qpos, gt_qpos):
        w_rp, w_rr, w_p, w_ee = 50, 50, 1, 10
        fk_res_pred = self.fk_model.qpos_fk(pred_qpos)
        fk_res_gt = self.fk_model.qpos_fk(gt_qpos)

        pred_wbpos = fk_res_pred["wbpos"].reshape(pred_qpos.shape[0], -1)
        gt_wbpos = fk_res_gt["wbpos"].reshape(pred_qpos.shape[0], -1)

        r_pos_loss = root_pos_loss(gt_qpos, pred_qpos).mean()
        r_rot_loss = root_orientation_loss(gt_qpos, pred_qpos).mean()
        p_rot_loss = pose_rot_loss(gt_qpos, pred_qpos).mean()  # pose loss
        ee_loss = end_effector_pos_loss(gt_wbpos, pred_wbpos).mean()  # End effector loss

        loss = w_rp * r_pos_loss + w_rr * r_rot_loss + w_p * p_rot_loss + w_ee * ee_loss

        return loss, [i.item() for i in [r_pos_loss, r_rot_loss, p_rot_loss, ee_loss]]

    def compute_metrics(self, feature_pred, data):
        pred_jpos = (feature_pred["joints"].squeeze().reshape((-1, 22, 3)).clone().clone())
        gt_jpos = data["joints"].squeeze().reshape((-1, 22, 3)).clone()
        mpjpe_global = (np.linalg.norm((pred_jpos - gt_jpos).detach().cpu().numpy(), axis=2).mean() * 1000)

        pred_jpos_local = pred_jpos - pred_jpos[:, 0:1, :]
        gt_jpos_local = gt_jpos - gt_jpos[:, 0:1, :]
        mpjpe_local = (np.linalg.norm((pred_jpos_local - gt_jpos_local).detach().cpu().numpy(), axis=2).mean() * 1000)
        acc_err = (compute_error_accel(pred_jpos.detach().cpu().numpy(), gt_jpos.detach().cpu().numpy()).mean() * 1000)
        vel_err = (compute_error_vel(pred_jpos.detach().cpu().numpy(), gt_jpos.detach().cpu().numpy()).mean() * 1000)
        return {
            "mpjpe_local": mpjpe_local,
            "mpjpe_global": mpjpe_global,
            "acc_err": acc_err,
            "vel_err": vel_err,
        }
