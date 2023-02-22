'''
File: /humanoid_kin_v1.py
Created Date: Tuesday June 22nd 2021
Author: Zhengyi Luo
Comment:
-----
Last Modified: Tuesday June 22nd 2021 5:33:25 pm
Modified By: Zhengyi Luo at <zluo2@cs.cmu.edu>
-----
Copyright (c) 2022 Carnegie Mellon University, KLab
-----
'''

from cmath import inf
from multiprocessing.spawn import get_preparation_data
from turtle import heading
import joblib
from numpy import isin
from scipy.linalg import cho_solve, cho_factor
import time
import pickle
from mujoco_py import functions as mjf
import mujoco_py
from gym import spaces
import os
import sys
import os.path as osp

sys.path.append(os.getcwd())

from uhc.khrylib.rl.envs.common import mujoco_env
from uhc.khrylib.utils import *
from uhc.khrylib.utils.transformation import quaternion_from_euler, quaternion_from_euler_batch
from uhc.khrylib.rl.core.policy_gaussian import PolicyGaussian
from uhc.khrylib.rl.core.critic import Value
from uhc.khrylib.models.mlp import MLP
from uhc.models.policy_mcp import PolicyMCP
from uhc.utils.flags import flags
from uhc.envs.humanoid_im import HumanoidEnv

from gym import spaces
from mujoco_py import functions as mjf
import pickle
import time
from scipy.linalg import cho_solve, cho_factor
import joblib
import numpy as np
import matplotlib.pyplot as plt

from uhc.smpllib.numpy_smpl_humanoid import Humanoid
# from uhc.smpllib.smpl_robot import Robot

from uhc.smpllib.smpl_mujoco import smpl_6d_to_qpose, smpl_to_qpose, qpos_to_smpl, smpl_to_qpose_torch
from uhc.utils.torch_geometry_transforms import (angle_axis_to_rotation_matrix as aa2mat, rotation_matrix_to_angle_axis as mat2aa)
import json
import copy

from embodiedpose.models.humor.utils.humor_mujoco import reorder_joints_to_humor, MUJOCO_2_SMPL
from embodiedpose.models.humor.humor_model import HumorModel
from embodiedpose.models.humor.utils.torch import load_state as load_humor_state
from embodiedpose.models.humor.body_model.utils import smpl_to_openpose
from embodiedpose.smpllib.scene_robot import SceneRobot
from embodiedpose.models.humor.utils.velocities import estimate_velocities
from embodiedpose.models.uhm_model import UHMModel
from scipy.spatial.transform import Rotation as sRot
import uhc.utils.pytorch3d_transforms as tR
import autograd.numpy as anp
from autograd import elementwise_grad as egrad

from autograd.misc import const_graph

from uhc.smpllib.np_smpl_humanoid_batch import Humanoid_Batch
import collections
from uhc.utils.math_utils import normalize_screen_coordinates, op_to_root_orient, smpl_op_to_op
from uhc.utils.torch_ext import isNpArray
from uhc.smpllib.smpl_parser import (
    SMPL_EE_NAMES,
    SMPL_BONE_ORDER_NAMES,
    SMPLH_BONE_ORDER_NAMES,
)


def show_voxel(voxel_feat, name=None):
    num_grid = int(np.cbrt(voxel_feat.shape[0]))
    voxel_feat = voxel_feat.reshape(num_grid, num_grid, num_grid)
    x, y, z = np.indices((num_grid, num_grid, num_grid))
    colors = np.empty(voxel_feat.shape, dtype=object)
    colors[voxel_feat] = 'red'
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxel_feat, facecolors=colors, edgecolor='k')
    ax.view_init(16, 75)
    if name is None:
        plt.show()
    else:
        plt.savefig(name)
    plt.close()


class HumanoidKinEnvRes(HumanoidEnv):
    # Wrapper class that wraps around Copycat agent

    def __init__(self, kin_cfg, init_context, cc_iter=-1, mode="train", agent=None):
        self.cc_cfg = cc_cfg = kin_cfg.cc_cfg
        self.kin_cfg = kin_cfg
        self.target = {}
        self.prev_humor_state = {}
        self.cur_humor_state = {}
        self.is_root_obs = None
        self.agent = agent
        # self.simulate = False
        self.simulate = True
        self.voxel_thresh = 0.1
        self.next_frame_idx = 250
        self.op_thresh = 0.1
        # self.n_ransac = 100

        # env specific
        self.use_quat = cc_cfg.robot_cfg.get("ball", False)
        cc_cfg.robot_cfg['span'] = kin_cfg.model_specs.get("voxel_span", 1.8)

        self.smpl_robot_orig = SceneRobot(cc_cfg.robot_cfg, data_dir=osp.join(cc_cfg.base_dir, "data/smpl"))
        self.hb = Humanoid_Batch(data_dir=osp.join(cc_cfg.base_dir, "data/smpl"))
        self.smpl_robot = SceneRobot(
            cc_cfg.robot_cfg,
            data_dir=osp.join(cc_cfg.base_dir, "data/smpl"),
            masterfoot=cc_cfg.masterfoot,
        )
        self.xml_str = self.smpl_robot.export_xml_string().decode("utf-8")
        ''' Load Humor Model '''

        self.motion_prior = UHMModel(in_rot_rep="mat", out_rot_rep=kin_cfg.model_specs.get("out_rot_rep", "aa"), latent_size=24, model_data_config="smpl+joints", steps_in=1, use_gn=False)
        if self.kin_cfg.model_specs.get("use_rvel", False):
            self.motion_prior.data_names.append("root_orient_vel")
            self.motion_prior.input_dim_list += [3]

        if self.kin_cfg.model_specs.get("use_bvel", False):
            self.motion_prior.data_names.append("joints_vel")
            self.motion_prior.input_dim_list += [66]

        for param in self.motion_prior.parameters():
            param.requires_grad = False

        self.agg_data_names = self.motion_prior.data_names + ['points3d', "joints2d", "wbpos_cam", "beta"]

        if self.kin_cfg.model_specs.get("use_tcn", False):
            tcn_arch = self.kin_cfg.model_specs.get("tcn_arch", "3,3,3")
            filter_widths = [int(x) for x in tcn_arch.split(',')]
            self.num_context = int(np.prod(filter_widths))
            self.j2d_seq_feat = collections.deque([0] * self.num_context, self.num_context)

        self.body_grad = np.zeros(63)

        self.bm = bm = self.motion_prior.bm_dict['neutral']
        self.smpl2op_map = smpl_to_openpose(bm.model_type, use_hands=False, use_face=False, use_face_contour=False, openpose_format='coco25')
        self.smpl_2op_submap = self.smpl2op_map[self.smpl2op_map < 22]
        # if cfg.masterfoot:
        #     mujoco_env.MujocoEnv.__init__(self, cfg.mujoco_model_file)
        # else:
        #     mujoco_env.MujocoEnv.__init__(self, self.xml_str, 15)
        mujoco_env.MujocoEnv.__init__(self, self.xml_str, 15)
        self.prev_qpos = self.data.qpos.copy()

        self.setup_constants(cc_cfg, cc_cfg.data_specs, mode=mode, no_root=False)
        self.neutral_path = self.kin_cfg.data_specs['neutral_path']
        self.neutral_data = joblib.load(self.neutral_path)
        self.load_context(init_context)
        self.set_action_spaces()
        self.set_obs_spaces()
        self.weight = mujoco_py.functions.mj_getTotalmass(self.model)
        ''' Load CC Controller '''
        self.state_dim = state_dim = self.get_cc_obs().shape[0]
        cc_action_dim = self.action_dim
        if cc_cfg.actor_type == "gauss":
            self.cc_policy = PolicyGaussian(cc_cfg, action_dim=cc_action_dim, state_dim=state_dim)
        elif cc_cfg.actor_type == "mcp":
            self.cc_policy = PolicyMCP(cc_cfg, action_dim=cc_action_dim, state_dim=state_dim)

        self.cc_value_net = Value(MLP(state_dim, cc_cfg.value_hsize, cc_cfg.value_htype))
        if cc_iter != -1:
            cp_path = '%s/iter_%04d.p' % (cc_cfg.model_dir, cc_iter)
        else:
            import ipdb; ipdb.set_trace()
            cc_iter = np.max([int(i.split("_")[-1].split(".")[0]) for i in os.listdir(cc_cfg.model_dir)])
            cp_path = '%s/iter_%04d.p' % (cc_cfg.model_dir, cc_iter)
        print(('loading model from checkpoint: %s' % cp_path))
        model_cp = pickle.load(open(cp_path, "rb"))
        self.cc_running_state = model_cp['running_state']
        self.cc_policy.load_state_dict(model_cp['policy_dict'])
        self.cc_value_net.load_state_dict(model_cp['value_dict'])

        # Contact modelling
        body_id_list = self.model.geom_bodyid.tolist()
        self.contact_geoms = [body_id_list.index(self.model._body_name2id[body]) for body in SMPL_BONE_ORDER_NAMES]

    def reset_robot(self):
        beta = self.context_dict["beta"].copy()
        gender = self.context_dict["gender"].copy()
        scene_name = self.context_dict['cam']['scene_name']

        if "obj_info" in self.context_dict:
            obj_info = self.context_dict['obj_info']
            self.smpl_robot.load_from_skeleton(torch.from_numpy(beta[0:1, :]).float(), gender=gender, obj_info=obj_info)
        else:
            if not self.context_dict.get("load_scene", True):
                scene_name = None

            self.smpl_robot.load_from_skeleton(torch.from_numpy(beta[0:1, :]).float(), gender=gender, scene_and_key=scene_name)

        xml_str = self.smpl_robot.export_xml_string().decode("utf-8")

        self.reload_sim_model(xml_str)
        self.weight = self.smpl_robot.weight

        self.hb.update_model(torch.from_numpy(beta[0:1, :16]), torch.tensor(gender[0:1]))
        self.hb.update_projection(self.camera_params, self.smpl2op_map, MUJOCO_2_SMPL)
        self.proj_2d_loss = egrad(self.hb.proj_2d_loss)
        self.proj_2d_body_loss = egrad(self.hb.proj_2d_body_loss)
        self.proj_2d_root_loss = egrad(self.hb.proj_2d_root_loss)
        self.proj_2d_line_loss = egrad(self.hb.proj_2d_line_loss)
        return xml_str

    def load_context(self, data_dict):
        self.context_dict = {k: v.squeeze().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}

        self.camera_params = data_dict['cam']
        self.camera_params_torch = {k: torch.from_numpy(v).double() if isNpArray(v) else v for k, v in self.camera_params.items()}

        self.reset_robot()
        self.humanoid.update_model(self.model)

        self.context_dict['len'] = self.context_dict['pose_aa'].shape[0] - 1

        gt_qpos = smpl_to_qpose(self.context_dict['pose_aa'], self.model, trans=self.context_dict['trans'], count_offset=True)
        init_qpos = smpl_to_qpose(self.context_dict['init_pose_aa'][None,], self.model, trans=self.context_dict['init_trans'][None,], count_offset=True)
        self.context_dict["qpos"] = gt_qpos

        self.target = self.humanoid.qpos_fk(torch.from_numpy(init_qpos))
        self.prev_humor_state = {k: data_dict[k][:, 0:1, :].clone() for k in self.motion_prior.data_names}
        self.cur_humor_state = self.prev_humor_state
        self.gt_targets = self.humanoid.qpos_fk(torch.from_numpy(gt_qpos))
        self.target.update({k: data_dict[k][:, 0:1, :].clone() for k in self.motion_prior.data_names})  # Initializing target

        if self.kin_cfg.model_specs.get("use_tcn", False):
            world_body_pos = self.target['wbpos'].reshape(24, 3)[MUJOCO_2_SMPL][self.smpl_2op_submap]

            world_trans = world_body_pos[..., 7:8:, :]
            self.pred_tcn = {
                'world_body_pos': world_body_pos - world_trans,
                'world_trans': world_trans,
            }

            casual = self.kin_cfg.model_specs.get("casual_tcn", True)
            full_R, full_t = self.camera_params["full_R"], self.camera_params['full_t']
            if casual:
                joints2d = self.context_dict["joints2d"][0:1].copy()
                joints2d[joints2d[..., 2] < self.op_thresh] = 0
                joints2d[..., :2] = normalize_screen_coordinates(joints2d[..., :2], self.camera_params['img_w'], self.camera_params['img_h'])
                joints2d = np.pad(joints2d, ((self.num_context - 1, 0), (0, 0), (0, 0)), mode="edge")

            else:
                joints2d = self.context_dict["joints2d"][:(self.num_context // 2 + 1)].copy()
                joints2d[joints2d[..., 2] < self.op_thresh] = 0
                joints2d[..., :2] = normalize_screen_coordinates(joints2d[..., :2], self.camera_params['img_w'], self.camera_params['img_h'])
                joints2d = np.pad(joints2d, ((self.num_context // 2, self.num_context // 2 + 1 - joints2d.shape[0]), (0, 0), (0, 0)), mode="edge")

            if self.kin_cfg.model_specs.get("tcn_3dpos", False):
                world_body_pos = self.target['wbpos'].reshape(24, 3)[MUJOCO_2_SMPL][self.smpl_2op_submap]
                world_body_pos = smpl_op_to_op(world_body_pos)
                cam_body_pos = world_body_pos @ full_R.T + full_t
                j2d3dfeat = np.concatenate([joints2d[..., :2], np.repeat(cam_body_pos[None,], self.num_context, axis=0)], axis=-1)

                [self.j2d_seq_feat.append(j3dfeat) for j3dfeat in j2d3dfeat]
                self.pred_tcn['cam_body_pos'] = cam_body_pos
            else:
                [self.j2d_seq_feat.append(j2dfeat) for j2dfeat in joints2d[..., :2]]

    def set_model_params(self):
        if self.cc_cfg.action_type == 'torque' and hasattr(self.cc_cfg, 'j_stiff'):
            self.model.jnt_stiffness[1:] = self.cc_cfg.j_stiff
            self.model.dof_damping[6:] = self.cc_cfg.j_damp

    def get_obs(self):
        ar_obs = self.get_ar_obs_v1()
        return ar_obs

    def get_cc_obs(self):
        return super().get_obs()

    def get_ar_obs_v1(self):
        t = self.cur_t
        obs = []
        compute_root_obs = False
        if self.is_root_obs is None:
            self.is_root_obs = []
            compute_root_obs = True

        curr_qpos = self.data.qpos[:self.qpos_lim].copy()
        curr_qvel = self.data.qvel[:self.qvel_lim].copy()
        self.prev_humor_state = copy.deepcopy(self.cur_humor_state)
        self.cur_humor_state = humor_dict = self.get_humor_dict_obs_from_sim()

        curr_root_quat = self.remove_base_rot(curr_qpos[3:7])

        full_R, full_t = self.camera_params_torch['full_R'], self.camera_params_torch['full_t']

        target_global_dict = {k: torch.from_numpy(self.context_dict[k][(t + 1):(t + 2)].reshape(humor_dict[k].shape)) for k in self.motion_prior.data_names}

        humor_local_dict, next_target_local_dict, info_dict = self.motion_prior.canonicalize_input_double(humor_dict, target_global_dict, split_input=False, return_info=True)

        # print(torch.matmul(humor_dict['trans'], full_R.T) + full_t)
        heading_rot = info_dict['world2aligned_rot'].numpy()

        curr_body_obs = np.concatenate([humor_local_dict[k].flatten().numpy() for k in self.motion_prior.data_names])

        hq = get_heading_new(curr_qpos[3:7])
        hq = 0
        obs.append(np.array([hq]))

        obs.append(curr_body_obs)
        if compute_root_obs:
            self.is_root_obs.append(np.array([1]))
            self.is_root_obs.append(np.concatenate([[1 if "root" in k else 0] * humor_local_dict[k].flatten().numpy().shape[-1] for k in self.motion_prior.data_names]))

        if self.kin_cfg.model_specs.get("use_tcn", False):
            casual = self.kin_cfg.model_specs.get("casual_tcn", True)
            if casual:
                joints2d_gt = self.context_dict['joints2d'][self.cur_t + 1].copy()
                joints2d_gt[..., :2] = normalize_screen_coordinates(joints2d_gt[..., :2], self.camera_params['img_w'], self.camera_params['img_h'])
                joints2d_gt[joints2d_gt[..., 2] < self.op_thresh] = 0
            else:
                t = self.cur_t + 1
                pad_num = self.num_context // 2 + 1
                joints2d_gt = self.context_dict['joints2d'][t:(t + pad_num)].copy()
                if joints2d_gt.shape[0] < pad_num:
                    joints2d_gt = np.pad(joints2d_gt, ([0, pad_num - joints2d_gt.shape[0]], [0, 0], [0, 0]), mode="edge")

                joints2d_gt[..., :2] = normalize_screen_coordinates(joints2d_gt[..., :2], self.camera_params['img_w'], self.camera_params['img_h'])
                joints2d_gt[joints2d_gt[..., 2] < self.op_thresh] = 0

            if self.kin_cfg.model_specs.get("tcn_3dpos", False):
                # cam_pred_tcn_3d = humor_dict['cam_pred_tcn_3d']
                # j2d3dfeat = np.concatenate([joints2d_gt[..., :2], cam_pred_tcn_3d.numpy().squeeze()], axis = 1)
                cam_pred_3d = humor_dict['cam_pred_3d']
                cam_pred_3d = smpl_op_to_op(cam_pred_3d)

                if casual:
                    j2d3dfeat = np.concatenate([joints2d_gt[..., :2], cam_pred_3d.squeeze()], axis=1)
                    self.j2d_seq_feat.append(j2d3dfeat)  # push next step obs into state
                else:
                    j2d3dfeat = np.concatenate([joints2d_gt[..., :2], np.repeat(cam_pred_3d.squeeze(1), self.num_context // 2 + 1, axis=0)], axis=-1)

                    [self.j2d_seq_feat.pop() for _ in range(self.num_context // 2)]
                    [self.j2d_seq_feat.append(feat) for feat in j2d3dfeat]

            else:
                if casual:
                    self.j2d_seq_feat.append(joints2d_gt[:, :2])  # push next step obs into state
                else:
                    [self.j2d_seq_feat.pop() for _ in range(self.num_context // 2)]

                    [self.j2d_seq_feat.append(feat) for feat in joints2d_gt[..., :2]]

            j2d_seq = np.array(self.j2d_seq_feat).flatten()
            obs.append(j2d_seq)

            if compute_root_obs:
                self.is_root_obs.append(np.array([3] * j2d_seq.shape[0]))

            tcn_root_grad = self.kin_cfg.model_specs.get("tcn_root_grad", False)  # use tcn directly on the projection gradient

            world_body_pos, world_trans = self.pred_tcn['world_body_pos'], self.pred_tcn['world_trans']

            curr_body_jts = humor_dict['joints'].reshape(22, 3)[self.smpl_2op_submap].numpy()
            curr_body_jts -= curr_body_jts[..., 7:8, :]
            world_body_pos -= world_body_pos[..., 7:8, :]

            body_diff = transform_vec_batch_new(world_body_pos - curr_body_jts, curr_root_quat).T.flatten()

            if self.kin_cfg.model_specs.get("tcn_body", False):
                obs.append(body_diff)

            curr_trans = self.target['wbpos'][:, :3]  # this is in world coord
            trans_diff = np.matmul(world_trans - curr_trans, heading_rot[0].T).flatten()
            trans_diff[2] = world_trans[:, 2]  # Mimicking the target trans feat.
            if self.kin_cfg.model_specs.get("tcn_traj", False):
                obs.append(trans_diff)

            if not tcn_root_grad:
                pred_root_mat = op_to_root_orient(world_body_pos[None,])
                root_rot_diff = np.matmul(heading_rot, pred_root_mat).flatten()
                obs.append(root_rot_diff)

            if self.kin_cfg.model_specs.get("tcn_body", False):
                if compute_root_obs:
                    self.is_root_obs.append(np.array([0] * body_diff.shape[0]))

            if self.kin_cfg.model_specs.get("tcn_traj", False):
                if compute_root_obs:
                    self.is_root_obs.append(np.array([1] * trans_diff.shape[0]))

            if not tcn_root_grad:
                if compute_root_obs:
                    self.is_root_obs.append(np.array([1] * root_rot_diff.shape[0]))

        if self.kin_cfg.model_specs.get("use_rt", True):
            trans_target_local = next_target_local_dict['trans'].flatten().numpy()
            obs.append(trans_target_local)
            if compute_root_obs:
                self.is_root_obs.append(np.array([1] * trans_target_local.shape[0]))

        if self.kin_cfg.model_specs.get("use_rr", False):
            root_rot_diff = next_target_local_dict['root_orient'].flatten().numpy()
            obs.append(root_rot_diff)
            if compute_root_obs:
                self.is_root_obs.append(np.array([1] * root_rot_diff.shape[0]))

        if self.kin_cfg.model_specs.get("use_3d_grad", False):
            normalize = self.kin_cfg.model_specs.get("normalize_3d_grad", True)
            proj2dgrad = humor_dict['proj2dgrad'].squeeze().numpy().copy()
            proj2dgrad = np.nan_to_num(proj2dgrad, nan=0, posinf=0, neginf=0)
            proj2dgrad = np.clip(proj2dgrad, -200, 200)

            if normalize:
                body_mul = root_mul = 1
            else:
                grad_mul = self.kin_cfg.model_specs.get("grad_mul", 10)
                body_mul = (10 * grad_mul)
                root_mul = (100 * grad_mul)

            trans_grad = (np.matmul(heading_rot, proj2dgrad[:3]) / root_mul).squeeze()
            root_grad = (sRot.from_matrix(heading_rot) * sRot.from_rotvec(proj2dgrad[3:6] / body_mul)).as_rotvec().squeeze()
            body_grad = proj2dgrad[6:69] / body_mul

            obs.append(trans_grad)
            if compute_root_obs:
                self.is_root_obs.append(np.array([1] * trans_grad.shape[0]))
            obs.append(root_grad)
            if compute_root_obs:
                self.is_root_obs.append(np.array([1] * root_grad.shape[0]))
            obs.append(body_grad)
            if compute_root_obs:
                self.is_root_obs.append(np.array([1] * body_grad.shape[0]))

        elif self.kin_cfg.model_specs.get("use_3d_grad_adpt", False):
            no_grad_body = self.kin_cfg.model_specs.get("no_grad_body", False)
            proj2dgrad = humor_dict['proj2dgrad'].squeeze().numpy().copy()
            proj2dgrad = np.nan_to_num(proj2dgrad, nan=0, posinf=0, neginf=0)
            proj2dgrad = np.clip(proj2dgrad, -200, 200)

            trans_grad = (np.matmul(heading_rot, proj2dgrad[:3])).squeeze()
            root_grad = (sRot.from_matrix(heading_rot) * sRot.from_rotvec(proj2dgrad[3:6])).as_rotvec().squeeze()
            body_grad = proj2dgrad[6:69]

            if no_grad_body:
                # Ablation, zero body grad. Just TCN
                body_grad = np.zeros_like(body_grad)

            obs.append(trans_grad)
            if compute_root_obs:
                self.is_root_obs.append(np.array([1] * trans_grad.shape[0]))
            obs.append(root_grad)
            if compute_root_obs:
                self.is_root_obs.append(np.array([1] * root_grad.shape[0]))
            obs.append(body_grad)
            if compute_root_obs:
                self.is_root_obs.append(np.array([1] * body_grad.shape[0]))

        if self.kin_cfg.model_specs.get("use_sdf", False):
            sdf_vals = self.smpl_robot.get_sdf_np(self.cur_humor_state['joints'].reshape(-1, 3), topk=3)
            obs.append(sdf_vals.numpy().flatten())

            if compute_root_obs:
                self.is_root_obs.append(np.array([2] * sdf_vals.shape[0]))

        elif self.kin_cfg.model_specs.get("use_dir_sdf", False):
            sdf_vals, sdf_dirs = self.smpl_robot.get_sdf_np(self.cur_humor_state['joints'].reshape(-1, 3), topk=3, return_grad=True)
            sdf_dirs = np.matmul(sdf_dirs, heading_rot[0].T)  # needs to be local dir coord
            sdf_feat = (sdf_vals[:, :, None] * sdf_dirs).numpy().flatten()
            obs.append(sdf_feat)
            if compute_root_obs:
                self.is_root_obs.append(np.array([2] * sdf_feat.shape[0]))

        if self.kin_cfg.model_specs.get("use_voxel", False):
            voxel_res = self.kin_cfg.model_specs.get("voxel_res", 8)

            voxel_feat = self.smpl_robot.query_voxel(self.cur_humor_state['trans'].reshape(-1, 3), self.cur_humor_state['root_orient'].reshape(3, 3), res=voxel_res).flatten()

            inside, outside = voxel_feat <= 0, voxel_feat >= self.voxel_thresh
            middle = np.logical_and(~inside, ~outside)

            voxel_feat[inside], voxel_feat[outside] = 1, 0
            voxel_feat[middle] = (self.voxel_thresh - voxel_feat[middle]) / self.voxel_thresh

            # voxel_feat[:] = 0

            if compute_root_obs:
                self.is_root_obs.append(np.array([2] * voxel_feat.shape[0]))
            obs.append(voxel_feat)

        if self.kin_cfg.model_specs.get("use_contact", False):
            contact_feat = np.zeros(24)
            for contact in self.data.contact[:self.data.ncon]:
                g1, g2 = contact.geom1, contact.geom2
                if g1 in self.contact_geoms and not g2 in self.contact_geoms:
                    contact_feat[g1 - 1] = 1
                if g2 in self.contact_geoms and not g1 in self.contact_geoms:
                    contact_feat[g2 - 1] = 1

            if compute_root_obs:
                self.is_root_obs.append(np.array([0] * contact_feat.shape[0]))
            obs.append(contact_feat)

        # voxel_feat_show = self.smpl_robot.query_voxel(
        #     self.cur_humor_state['trans'].reshape(-1, 3),
        #     self.cur_humor_state['root_orient'].reshape(3, 3),
        #     res=16).flatten()
        # os.makedirs(osp.join("temp", self.context_dict['seq_name']), self.cfg.id, exist_ok=True)
        # show_voxel(voxel_feat_show <= 0.05, name = osp.join("temp", self.context_dict['seq_name'], self.cfg.id, f"voxel_{self.cur_t:05d}.png"))
        # show_voxel(voxel_feat_show <= 0.05, name = None)

        obs = np.concatenate(obs)
        if compute_root_obs:
            self.is_root_obs = np.concatenate(self.is_root_obs)
            assert (self.is_root_obs.shape == obs.shape)

        return obs

    def step_ar(self, action, dt=1 / 30):
        cfg = self.kin_cfg
        next_global_out = self.motion_prior.step_state(self.cur_humor_state, torch.from_numpy(action[None, :69]))  # change this to number of joints
        body_pose_aa = mat2aa(next_global_out['pose_body'].reshape(21, 3, 3)).reshape(1, 63)
        root_aa = mat2aa(next_global_out['root_orient'].reshape(1, 3, 3)).reshape(1, 3)

        pose_aa = torch.cat([root_aa, body_pose_aa, torch.zeros(1, 6).to(root_aa)], dim=1)
        qpos = smpl_to_qpose_torch(pose_aa, self.model, trans=next_global_out['trans'].reshape(1, 3), count_offset=True)
        if self.mode == "train" and self.agent.iter < self.agent.num_supervised and self.agent.iter >= 0:
            # Dagger
            qpos = torch.from_numpy(self.gt_targets['qpos'][(self.cur_t):(self.cur_t + 1)])
            fk_res = self.humanoid.qpos_fk(qpos)
        else:
            fk_res = self.humanoid.qpos_fk(qpos)

        self.target = fk_res
        self.target.update(next_global_out)

        if self.kin_cfg.model_specs.get("use_tcn", False):
            full_R, full_t = self.camera_params['full_R'], self.camera_params['full_t']
            kp_feats = action[69:].copy()
            cam_trans = kp_feats[None, :3]
            cam_body_pos = kp_feats[3:].reshape(14, 3)

            self.pred_tcn['world_trans'] = (cam_trans - full_t).dot(full_R)  # camera to world transformation
            self.pred_tcn['world_body_pos'] = cam_body_pos.dot(full_R)
            self.pred_tcn['cam_body_pos'] = cam_trans + cam_body_pos

    def get_humanoid_pose_aa_trans(self, qpos=None):
        if qpos is None:
            qpos = self.data.qpos.copy()[None]
        pose_aa, trans = qpos_to_smpl(qpos, self.model, self.cc_cfg.robot_cfg.get("model", "smpl"))

        return pose_aa, trans

    def get_humor_dict_obs_from_sim(self):
        # Compute humor obs based on current and previous simulation state.

        qpos = self.data.qpos.copy()[None]
        # qpos = self.get_expert_qpos()[None] # No simulate

        prev_qpos = self.prev_qpos[None]

        # Calculating the velocity difference from simulation
        qpos_stack = np.concatenate([prev_qpos, qpos])
        pose_aa, trans = self.get_humanoid_pose_aa_trans(qpos_stack)
        fk_result = self.humanoid.qpos_fk(torch.from_numpy(qpos_stack), to_numpy=False, full_return=False)
        trans_batch = torch.from_numpy(trans[None])

        joints = fk_result["wbpos"].reshape(-1, 24, 3)[:, MUJOCO_2_SMPL].reshape(-1, 72)[:, :66]
        pose_aa_mat = aa2mat(torch.from_numpy(pose_aa.reshape(-1, 3))).reshape(1, 2, 24, 4, 4)[..., :3, :3]

        humor_out = {}
        trans_vel, joints_vel, root_orient_vel = estimate_velocities(trans_batch, pose_aa_mat[:, :, 0], joints[None], 30, aa_to_mat=False)

        humor_out['trans_vel'] = trans_vel[:, 0:1, :]
        humor_out['joints_vel'] = joints_vel[:, 0:1, :]
        humor_out['root_orient_vel'] = root_orient_vel[:, 0:1, :]

        humor_out['joints'] = joints[None, 1:2]
        humor_out['pose_body'] = pose_aa_mat[:, 1:2, 1:22]  # contains current qpos

        humor_out['root_orient'] = pose_aa_mat[:, 1:2, 0]
        humor_out['trans'] = trans_batch[:, 1:2]

        ######################## Compute 2D Keypoint projection and 3D keypoint ######################
        grad_frame_num = self.kin_cfg.model_specs.get("grad_frame_num", 1)

        t = self.cur_t + 1
        joints2d_gt = self.context_dict['joints2d'][t:(t + grad_frame_num)].copy()

        if joints2d_gt.shape[0] < grad_frame_num:
            joints2d_gt = np.pad(joints2d_gt, ([0, grad_frame_num - joints2d_gt.shape[0]], [0, 0], [0, 0]), mode="edge")

        inliers = joints2d_gt[..., 2] > self.op_thresh
        self.hb.update_tgt_joints(joints2d_gt[..., :2], inliers)

        input_vec = np.concatenate([humor_out['trans'].numpy(), pose_aa[1:2].reshape(1, -1, 72)], axis=2)
        pred_2d, cam_pred_3d = self.hb.proj2d(fk_result["wbpos"][1:2].reshape(24, 3).numpy(), return_cam_3d=True)

        humor_out["pred_joints2d"] = torch.from_numpy(pred_2d[None,])
        humor_out["cam_pred_3d"] = torch.from_numpy(cam_pred_3d[None,])

        if self.kin_cfg.model_specs.get("use_tcn", False) and self.kin_cfg.model_specs.get("tcn_3dpos", False):
            cam_pred_tcn_3d = self.pred_tcn['cam_body_pos'][None,]
            humor_out["cam_pred_tcn_3d"] = torch.from_numpy(cam_pred_tcn_3d[None,])

        order = self.kin_cfg.model_specs.get("use_3d_grad_ord", 1)
        normalize = self.kin_cfg.model_specs.get("normalize_grad", False)

        depth = np.mean(cam_pred_3d[..., 2])
        if self.kin_cfg.model_specs.get("use_3d_grad", False):
            num_adpt_grad = 1
            grad_step = self.kin_cfg.model_specs.get("grad_step", 5)

            pose_grad, input_vec_new, curr_loss = self.multi_step_grad(input_vec, order=order, num_adpt_grad=num_adpt_grad, normalize=normalize, step_size=grad_step)

            multi = depth / 10
            pose_grad[:6] *= multi

            humor_out["proj2dgrad"] = pose_grad

        elif self.kin_cfg.model_specs.get("use_3d_grad_line", False):
            proj2dgrad = self.proj_2d_line_loss(input_vec)
            humor_out["proj2dgrad"] = -torch.from_numpy(proj2dgrad)

        elif self.kin_cfg.model_specs.get("use_3d_grad_adpt", False):
            num_adpt_grad = self.kin_cfg.model_specs.get("use_3d_grad_adpt_num", 5)
            grad_step = self.kin_cfg.model_specs.get("grad_step", 5)

            pose_grad, input_vec_new, curr_loss = self.multi_step_grad(input_vec, order=order, num_adpt_grad=num_adpt_grad, normalize=normalize, step_size=grad_step)

            multi = depth / 10
            pose_grad[:6] *= multi

            humor_out["proj2dgrad"] = pose_grad

        return humor_out

    def geo_trans(self, input_vec_new):
        delta_t = np.zeros(3)
        geo_tran_cap = self.kin_cfg.model_specs.get("geo_trans_cap", 0.1)
        try:
            inliners = self.hb.inliers
            if np.sum(inliners) >= 3:
                wbpos = self.hb.fk_batch_grad(input_vec_new)
                cam2d, cam3d = self.hb.proj2d(wbpos, True)
                cam3d = smpl_op_to_op(cam3d)
                j2ds = self.hb.gt_2d_joints[0].copy()
                K = self.camera_params['K']
                fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                A1 = np.tile([fu, 0], 12)
                A2 = np.tile([0, fv], 12)
                A3 = np.tile([cu, cv], 12) - j2ds.flatten()

                b_1 = j2ds[:, 0] * cam3d[0, :, 2] - fu * cam3d[0, :, 0] - cu * cam3d[0, :, 2]
                b_2 = j2ds[:, 1] * cam3d[0, :, 2] - fv * cam3d[0, :, 1] - cv * cam3d[0, :, 2]
                b = np.hstack([b_1[:, None], b_2[:, None]]).flatten()[:, None]
                A = np.hstack([A1[:, None], A2[:, None], A3[:, None]])
                A = A[np.tile(inliners, 2).squeeze()]
                b = b[np.tile(inliners, 2).squeeze()]
                u, sigma, vt = np.linalg.svd(A)

                Sigma_pinv = np.zeros_like(A).T
                Sigma_pinv[:3, :3] = np.diag(1 / sigma)
                delta_t = vt.T.dot(Sigma_pinv).dot(u.T).dot(b)
                delta_t = self.hb.full_R.T @ delta_t[:, 0]

                if np.linalg.norm(delta_t) > geo_tran_cap:
                    delta_t = delta_t / np.linalg.norm(delta_t) * geo_tran_cap
            else:
                delta_t = np.zeros(3)

        except Exception as e:
            print("error in svd and pose", e)
        return delta_t

    # def geo_trans(self, input_vec_new):
    #     best_delta_t = np.zeros((3, 1))
    #     geo_tran_cap = self.kin_cfg.model_specs.get("geo_trans_cap", 0.01)
    #     wbpos = self.hb.fk_batch_grad(input_vec_new)
    #     cam2d, cam3d = self.hb.proj2d(wbpos, True)
    #     cam3d = smpl_op_to_op(cam3d)

    #     # j2ds = self.hb.gt_2d_joints[0].copy()
    #     # inliners = self.hb.inliers

    #     grad_frame_num = self.kin_cfg.model_specs.get("grad_frame_num", 1)
    #     t = self.cur_t + 1
    #     joints2d_gt = self.context_dict['joints2d'][t:(t +
    #                                                    grad_frame_num)].copy()
    #     if joints2d_gt.shape[0] < grad_frame_num:
    #         joints2d_gt = np.pad(
    #             joints2d_gt,
    #             ([0, grad_frame_num - joints2d_gt.shape[0]], [0, 0], [0, 0]),
    #             mode="edge")

    #     j2ds = joints2d_gt[..., :2].copy()[0]
    #     # inliners = joints2d_gt[..., 2].copy() > 0.5

    #     best_err = np.inf
    #     for _ in range(self.n_ransac):
    #         samples = np.arange(j2ds.shape[0])
    #         samples = np.random.choice(samples, 3, replace=False)
    #         inds = np.zeros((1, j2ds.shape[0]), dtype=bool)
    #         inds[:, samples] = 1
    #         K = self.camera_params['K']
    #         fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    #         A1 = np.tile([fu, 0], 12)
    #         A2 = np.tile([0, fv], 12)
    #         A3 = np.tile([cu, cv], 12) - j2ds.flatten()

    #         b_1 = j2ds[:, 0] * cam3d[0, :, 2] - fu * cam3d[
    #             0, :, 0] - cu * cam3d[0, :, 2]
    #         b_2 = j2ds[:, 1] * cam3d[0, :, 2] - fv * cam3d[
    #             0, :, 1] - cv * cam3d[0, :, 2]
    #         b_orig = np.hstack([b_1[:, None], b_2[:, None]]).flatten()[:, None]
    #         A_orig = np.hstack([A1[:, None], A2[:, None], A3[:, None]])
    #         A = A_orig[np.tile(inds, 2).squeeze()].copy()
    #         b = b_orig[np.tile(inds, 2).squeeze()].copy()
    #         try:
    #             u, sigma, vt = np.linalg.svd(A)
    #             Sigma_pinv = np.zeros_like(A).T
    #             Sigma_pinv[:3, :3] = np.diag(1 / sigma)
    #             delta_t = vt.T.dot(Sigma_pinv).dot(u.T).dot(b)
    #             err = np.linalg.norm(A_orig.dot(delta_t) - b_orig, 2)
    #             if err < best_err:
    #                 best_err = err
    #                 best_delta_t = delta_t
    #         except Exception:
    #             continue

    #     delta_t = self.hb.full_R.T @ best_delta_t[:, 0]

    #     meter_cap = geo_tran_cap
    #     if np.linalg.norm(delta_t) > meter_cap:
    #         delta_t = delta_t / np.linalg.norm(delta_t) * meter_cap
    #     return delta_t

    def multi_step_grad(self, input_vec, num_adpt_grad=5, normalize=False, order=2, step_size=5):
        geo_trans = self.kin_cfg.model_specs.get("geo_trans", False)
        tcn_root_grad = self.kin_cfg.model_specs.get("tcn_root_grad", False)
        input_vec_new = input_vec.copy()
        prev_loss = orig_loss = self.hb.proj_2d_loss(input_vec_new, ord=order, normalize=normalize)

        if tcn_root_grad:
            world_body_pos, world_trans = self.pred_tcn['world_body_pos'], self.pred_tcn['world_trans']
            pred_root_vec = sRot.from_matrix(op_to_root_orient(world_body_pos[None,])).as_rotvec()  # tcn's root
            input_vec_new[..., 3:6] = pred_root_vec

        if order == 1:
            step_size = 0.00001
            step_size_a = step_size * np.clip(prev_loss, 0, 5)
        else:
            if normalize:
                step_size_a = step_size / 1.02
            else:
                step_size_a = 0.000005
        for iteration in range(num_adpt_grad):
            if self.kin_cfg.model_specs.get("use_3d_grad_sept", False):
                proj2dgrad_body = self.proj_2d_body_loss(input_vec_new, ord=order, normalize=normalize)
                proj2dgrad = self.proj_2d_loss(input_vec_new, ord=order, normalize=normalize)
                proj2dgrad[..., 3:] = proj2dgrad_body[..., 3:]
                proj2dgrad = np.nan_to_num(proj2dgrad, posinf=0, neginf=0)  # This is essentail, otherwise nan will get more
            else:
                proj2dgrad = self.proj_2d_loss(input_vec_new, ord=order, normalize=normalize)
                proj2dgrad = np.nan_to_num(proj2dgrad, posinf=0, neginf=0)  # This is essentail, otherwise nan will get more

            # import ipdb
            # ipdb.set_trace()
            # wbpos = self.hb.fk_batch_grad(input_vec_new); pred_joints2d = self.hb.proj2d(wbpos); joblib.dump(pred_joints2d, "a.pkl"); joblib.dump(self.hb.gt_2d_joints, "b.pkl")

            input_vec_new = input_vec_new - proj2dgrad * step_size_a

            if geo_trans:
                delta_t = self.geo_trans(input_vec_new)
                delta_t = np.concatenate([delta_t, np.zeros(72)])
                input_vec_new += delta_t

            curr_loss = self.hb.proj_2d_loss(input_vec_new, ord=order, normalize=normalize)

            if curr_loss > prev_loss:
                step_size_a *= 0.5
            prev_loss = curr_loss

        if self.hb.proj_2d_loss(input_vec_new, ord=order, normalize=normalize) > orig_loss:
            pose_grad = torch.zeros(proj2dgrad.shape)
        else:
            pose_grad = torch.from_numpy(input_vec_new - input_vec)
        return pose_grad, input_vec_new, curr_loss

    def load_camera_params(self):
        if "scene_name" in self.context_dict:
            scene_key = self.context_dict['scene_name']
        else:
            scene_key = self.context_dict['seq_name'][:-9]

        prox_path = self.kin_cfg.data_specs['prox_path']

        with open(f'{prox_path}/calibration/Color.json', 'r') as f:
            cameraInfo = json.load(f)
            K = np.array(cameraInfo['camera_mtx']).astype(np.float32)

        with open(f'{prox_path}/cam2world/{scene_key}.json', 'r') as f:
            camera_pose = np.array(json.load(f)).astype(np.float32)
            R = camera_pose[:3, :3]
            tr = camera_pose[:3, 3]
            R = R.T
            tr = -np.matmul(R, tr)

        with open(f'{prox_path}/alignment/{scene_key}.npz', 'rb') as f:
            aRt = np.load(f)
            aR = aRt['R']
            atr = aRt['t']
            aR = aR.T
            atr = -np.matmul(aR, atr)

        full_R = R.dot(aR)
        full_t = R.dot(atr) + tr
        self.camera_params = {"K": K, "R": R, "tr": tr, "aR": aR, "atr": atr, "full_R": full_R, "full_t": full_t}
        self.camera_params_torch = {k: torch.from_numpy(v).double() for k, v in self.camera_params.items()}

    def step(self, a, kin_override=False):
        fail = False
        cfg = self.kin_cfg
        cc_cfg = self.cc_cfg

        self.prev_qpos = self.get_humanoid_qpos()
        # self.prev_qpos = self.get_expert_qpos() ## No simulate

        self.prev_qvel = self.get_humanoid_qvel()
        self.prev_bquat = self.bquat.copy()
        self.prev_hpos = self.get_head().copy()
        self.step_ar(a.copy())

        # if flags.debug:
        # self.target = self.humanoid.qpos_fk(torch.from_numpy(self.context_dict['qpos'][self.cur_t:self.cur_t + 1])) # GT
        # self.target = self.smpl_humanoid.qpos_fk(self.ar_context['ar_qpos'][self.cur_t + 1]) # Debug
        # self.target = self.humanoid.qpos_fk(torch.from_numpy(self.gt_targets['qpos'][self.cur_t:self.cur_t + 1])) # GT
        # self.target = self.humanoid.qpos_fk(torch.from_numpy(self.gt_targets['qpos'][self.cur_t:self.cur_t + 1])) # Use gt

        cc_obs = self.get_cc_obs()
        cc_obs = self.cc_running_state(cc_obs, update=False)
        cc_a = self.cc_policy.select_action(torch.from_numpy(cc_obs)[None,], mean_action=True)[0].numpy()  # CC step

        if flags.debug:
            self.do_simulation(cc_a, self.frame_skip)
            # self.data.qpos[:self.qpos_lim] = self.gt_targets['qpos'][self.cur_t + 1]  # debug
            # self.sim.forward()  # debug
        else:
            if kin_override:
                self.data.qpos[:self.qpos_lim] = self.gt_targets['qpos'][self.cur_t + 1]  # debug
                self.sim.forward()  # debug
            else:
                if self.simulate:
                    try:
                        self.do_simulation(cc_a, self.frame_skip)
                    except Exception as e:
                        print("Exception in do_simulation", e, self.cur_t)
                        fail = True
                else:
                    self.data.qpos[:self.qpos_lim] = self.get_expert_qpos()  # debug
                    self.sim.forward()  # debug

        # if self.cur_t == 0 and self.agent.global_start_fr == 0:
        #     # ZL: Stablizing the first frame jump
        #     self.data.qpos[:self.qpos_lim] = self.get_expert_qpos()  # debug
        #     self.data.qvel[:] = 0
        #     self.sim.forward()  # debug

        self.cur_t += 1

        self.bquat = self.get_body_quat()
        # get obs
        reward = 1.0

        if cfg.env_term_body == 'body':
            body_diff = self.calc_body_diff()
            if self.mode == "train":
                body_gt_diff = self.calc_body_gt_diff()
                fail = fail or (body_diff > 2.5 or body_gt_diff > 3.5)
            else:
                fail = fail or body_diff > 7

            # fail = body_diff > 10
            # print(fail, self.cur_t)
            # fail = False
        else:
            raise NotImplemented()

        # if flags.debug:
        #     fail = False

        end = (self.cur_t >= cc_cfg.env_episode_len) or (self.cur_t + self.start_ind >= self.context_dict['len'])
        done = fail or end

        # if done:
        # print(f"Fail: {fail} | End: {end}", self.cur_t, body_diff)

        percent = self.cur_t / self.context_dict['len']
        if not done:
            obs = self.get_obs()  # can no longer compute obs when done....
        else:
            obs = np.zeros(self.obs_dim)

        return obs, reward, done, {'fail': fail, 'end': end, "percent": percent}

    def set_mode(self, mode):
        self.mode = mode

    def ar_fail_safe(self):
        self.data.qpos[:self.qpos_lim] = self.context_dict['ar_qpos'][self.cur_t + 1]
        # self.data.qpos[:self.qpos_lim] = self.get_target_qpos()
        self.data.qvel[:self.qvel_lim] = self.context_dict['ar_qvel'][self.cur_t + 1]
        self.sim.forward()

    def reset_model(self, qpos=None, qvel=None):
        cfg = self.kin_cfg
        ind = 0
        self.start_ind = 0
        if qpos is None:
            init_pose_aa = self.context_dict['init_pose_aa']
            init_trans = self.context_dict['init_trans']
            init_qpos = smpl_to_qpose(torch.from_numpy(init_pose_aa[None,]), self.model, torch.from_numpy(init_trans[None,]), count_offset=True).squeeze()
            init_vel = np.zeros(self.qvel_lim)
        else:
            init_qpos = qpos
            init_vel = qvel
        self.set_state(init_qpos, init_vel)
        self.prev_qpos = self.get_humanoid_qpos()
        return self.get_obs()

    def viewer_setup(self, mode):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.lookat[:2] = self.get_humanoid_qpos()[:2]
        # if mode not in self.set_cam_first:
        #     self.viewer.video_fps = 33
        #     self.viewer.frame_skip = self.frame_skip
        #     self.viewer.cam.distance = self.model.stat.extent * 1.2
        #     self.viewer.cam.elevation = -20
        #     self.viewer.cam.azimuth = 45
        #     self.set_cam_first.add(mode)

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array" or mode == "depth_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self._viewers[mode] = self.viewer
        self.viewer_setup("rgb")

        full_R, full_t = self.camera_params['full_R'], self.camera_params['full_t']
        distance = np.linalg.norm(full_t)
        x_axis = full_R.T[:, 0]
        pos_3d = -full_R.T.dot(full_t)
        rotation = sRot.from_matrix(full_R).as_euler("XYZ", degrees=True)
        self.viewer.cam.distance = 2  # + 3 to have better viewing
        self.viewer.cam.lookat[:] = pos_3d + x_axis
        self.viewer.cam.azimuth = 90 - rotation[2]
        self.viewer.cam.elevation = -8

        return self.viewer

    def match_heading_and_pos(self, qpos_1, qpos_2):
        posxy_1 = qpos_1[:2]
        qpos_1_quat = self.remove_base_rot(qpos_1[3:7])
        qpos_2_quat = self.remove_base_rot(qpos_2[3:7])
        heading_1 = get_heading_q(qpos_1_quat)
        qpos_2[3:7] = de_heading(qpos_2[3:7])
        qpos_2[3:7] = quaternion_multiply(heading_1, qpos_2[3:7])
        qpos_2[:2] = posxy_1
        return qpos_2

    def get_expert_qpos(self, delta_t=0):
        expert_qpos = self.target['qpos'].copy().squeeze()
        return expert_qpos

    def get_target_kin_pose(self, delta_t=0):
        return self.get_expert_qpos()[7:]

    def get_expert_joint_pos(self, delta_t=0):
        # world joint position
        wbpos = self.target['wbpos'].squeeze()
        return wbpos

    def get_expert_com_pos(self, delta_t=0):
        # body joint position
        body_com = self.target['body_com'].squeeze()
        return body_com

    def get_expert_bquat(self, delta_t=0):
        bquat = self.target['bquat'].squeeze()
        return bquat

    def get_expert_wbquat(self, delta_t=0):
        wbquat = self.target['wbquat'].squeeze()
        return wbquat

    def get_expert_shape_and_gender(self):
        cfg = self.cc_cfg

        shape = self.context_dict['beta'][0].squeeze()
        if shape.shape[0] == 10:
            shape = np.concatenate([shape, np.zeros(6)])

        gender = self.context_dict['gender'][0].squeeze()
        obs = []
        if cfg.get("has_pca", True):
            obs.append(shape)

        obs.append([gender])

        if cfg.get("has_weight", False):
            obs.append([self.weight])

        if cfg.get("has_bone_length", False):
            obs.append(self.smpl_robot.bone_length)

        return np.concatenate(obs)

    def calc_body_diff(self):
        cur_wbpos = self.get_wbody_pos().reshape(-1, 3)
        e_wbpos = self.get_expert_joint_pos().reshape(-1, 3)
        diff = cur_wbpos - e_wbpos
        diff *= self.jpos_diffw
        jpos_dist = np.linalg.norm(diff, axis=1).sum()
        return jpos_dist

    def calc_body_ar_diff(self):
        cur_wbpos = self.get_wbody_pos().reshape(-1, 3)
        # e_wbpos = self.get_target_joint_pos().reshape(-1, 3)
        e_wbpos = self.context_dict['ar_wbpos'][self.cur_t + 1].reshape(-1, 3)
        diff = cur_wbpos - e_wbpos
        diff *= self.jpos_diffw
        jpos_dist = np.linalg.norm(diff, axis=1).sum()
        return jpos_dist

    def calc_body_gt_diff(self):
        cur_wbpos = self.get_wbody_pos().reshape(-1, 3)
        e_wbpos = self.gt_targets['wbpos'][self.cur_t].reshape(-1, 3)
        diff = cur_wbpos - e_wbpos
        diff *= self.jpos_diffw
        jpos_dist = np.linalg.norm(diff, axis=1).sum()
        return jpos_dist

    def get_expert_attr(self, attr, ind):
        return self.context_dict[attr][ind].copy()


if __name__ == "__main__":
    pass
