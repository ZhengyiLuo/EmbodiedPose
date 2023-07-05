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

from uhc.smpllib.smpl_mujoco import smpl_6d_to_qpose, smpl_to_qpose, qpos_to_smpl, smpl_to_qpose_torch, smpl_to_qpose_multi
from uhc.utils.torch_geometry_transforms import (
    angle_axis_to_rotation_matrix as aa2mat, rotation_matrix_to_angle_axis as
    mat2aa)
import json
from uhc.utils.transformation import (
    quaternion_from_euler_batch,
    quaternion_multiply_batch,
    quat_mul_vec,
    quat_mul_vec_batch,
    quaternion_from_euler,
    quaternion_inverse_batch,
)

from embodiedpose.models.humor.utils.humor_mujoco import reorder_joints_to_humor, MUJOCO_2_SMPL
from embodiedpose.models.humor.humor_model import HumorModel
from embodiedpose.models.humor.utils.torch import load_state as load_humor_state
from embodiedpose.models.humor.body_model.utils import smpl_to_openpose
from embodiedpose.models.humor.utils.velocities import estimate_velocities
from embodiedpose.models.uhm_model import UHMModel
from scipy.spatial.transform import Rotation as sRot
import copycat.utils.pytorch3d_transforms as tR
import autograd.numpy as anp
from autograd import elementwise_grad as egrad
from uhc.smpllib.torch_smpl_humanoid import Humanoid
from uhc.smpllib.np_smpl_humanoid_batch import Humanoid_Batch
from embodiedpose.smpllib.scene_robot import SceneRobot
from embodiedpose.smpllib.multi_robot import Multi_Robot
from uhc.smpllib.smpl_parser import (
    SMPL_EE_NAMES,
    SMPL_BONE_ORDER_NAMES,
    SMPLH_BONE_ORDER_NAMES,
)
from uhc.smpllib.smpl_mujoco import SMPLConverter


class HumanoidKinEnvMulti(HumanoidEnv):
    # Wrapper class that wraps around Copycat agent

    def __init__(self,
                 kin_cfg,
                 init_context,
                 cc_iter=-1,
                 mode="train",
                 agent=None):
        self.cc_cfg = cc_cfg = kin_cfg.cc_cfg
        self.kin_cfg = kin_cfg
        self.target = {}
        self.prev_humor_state = {}
        self.cur_humor_state = {}
        self.agent = agent

        # env specific
        self.use_quat = cc_cfg.robot_cfg.get("ball", False)
        self.smpl_robot_orig = SceneRobot(cc_cfg.robot_cfg,
                                          data_dir=osp.join(
                                              cc_cfg.base_dir, "data/smpl"))
        self.hb = Humanoid_Batch(
            data_dir=osp.join(cc_cfg.base_dir, "data/smpl"))

        self.smpl_robot = Multi_Robot(
            cc_cfg.robot_cfg,
            data_dir=osp.join(cc_cfg.base_dir, "data/smpl"),
        )
        self.num_people = self.smpl_robot.num_people

        self.xml_str = self.smpl_robot.export_xml_string().decode("utf-8")
        ''' Load Humor Model '''

        self.motion_prior = UHMModel(in_rot_rep="mat",
                                     out_rot_rep=kin_cfg.model_specs.get(
                                         "out_rot_rep", "aa"),
                                     latent_size=24,
                                     model_data_config="smpl+joints",
                                     steps_in=1,
                                     use_gn=False)

        for param in self.motion_prior.parameters():
            param.requires_grad = False

        self.agg_data_names = self.motion_prior.data_names + [
            'points3d', "joints2d", "beta"
        ]

        if self.kin_cfg.model_specs.get("use_rvel", False):
            self.motion_prior.data_names.append("root_orient_vel")
            self.motion_prior.input_dim_list += [3]

        if self.kin_cfg.model_specs.get("use_bvel", False):
            self.motion_prior.data_names.append("joints_vel")
            self.motion_prior.input_dim_list += [66]

        self.bm = bm = self.motion_prior.bm_dict['neutral']
        self.smpl2op_map = smpl_to_openpose(bm.model_type,
                                            use_hands=False,
                                            use_face=False,
                                            use_face_contour=False,
                                            openpose_format='coco25')

        # if cfg.masterfoot:
        #     mujoco_env.MujocoEnv.__init__(self, cfg.mujoco_model_file)
        # else:
        #     mujoco_env.MujocoEnv.__init__(self, self.xml_str, 15)
        mujoco_env.MujocoEnv.__init__(self, self.xml_str, 15)
        self.prev_qpos = self.data.qpos.copy()

        self.setup_constants(cc_cfg,
                             cc_cfg.data_specs,
                             mode=mode,
                             no_root=False)
        self.neutral_path = self.kin_cfg.data_specs['neutral_path']
        self.neutral_data = joblib.load(self.neutral_path)
        self.load_context(init_context)
        self.set_action_spaces()
        self.set_obs_spaces()
        self.weight = mujoco_py.functions.mj_getTotalmass(self.model)
        ''' Load CC Controller '''
        self.state_dim = state_dim = self.get_cc_obs()[0].shape[0]
        cc_action_dim = int(self.action_dim / self.num_people)
        if cc_cfg.actor_type == "gauss":
            self.cc_policy = PolicyGaussian(cc_cfg,
                                            action_dim=cc_action_dim,
                                            state_dim=state_dim)
        elif cc_cfg.actor_type == "mcp":
            self.cc_policy = PolicyMCP(cc_cfg,
                                       action_dim=cc_action_dim,
                                       state_dim=state_dim)

        self.cc_value_net = Value(
            MLP(state_dim, cc_cfg.value_hsize, cc_cfg.value_htype))
        print(cc_cfg.model_dir)
        if cc_iter != -1:
            cp_path = '%s/iter_%04d.p' % (cc_cfg.model_dir, cc_iter)
        else:
            cc_iter = np.max([
                int(i.split("_")[-1].split(".")[0])
                for i in os.listdir(cc_cfg.model_dir)
            ])
            cp_path = '%s/iter_%04d.p' % (cc_cfg.model_dir, cc_iter)
        print(('loading model from checkpoint: %s' % cp_path))
        model_cp = pickle.load(open(cp_path, "rb"))
        self.cc_running_state = model_cp['running_state']
        self.cc_policy.load_state_dict(model_cp['policy_dict'])
        self.cc_value_net.load_state_dict(model_cp['value_dict'])

    def load_models(self):
        self.converter = SMPLConverter(
            self.smpl_model,
            self.smpl_model,  # Converterrrr
            smpl_model=self.cc_cfg.robot_cfg.get("model", "smpl"),
        )

        self.sim_iter = 15
        self.qpos_lim = self.converter.get_new_qpos_lim()
        self.qvel_lim = self.converter.get_new_qvel_lim()
        self.body_lim = self.converter.get_new_body_lim()
        self.jpos_diffw = self.converter.get_new_diff_weight()[:, None]
        self.body_diffw = self.converter.get_new_diff_weight()[1:]
        self.body_qposaddr = get_body_qposaddr(self.model)
        self.mujoco_body_order = [k[:-3] for k in self.body_qposaddr.keys()
                                  ][:24]  # Hacky

        self.jkd = self.converter.get_new_jkd() * self.cc_cfg.get("pd_mul", 1)
        self.jkp = self.converter.get_new_jkp() * self.cc_cfg.get("pd_mul", 1)

        self.a_scale = self.converter.get_new_a_scale()

        self.torque_lim = self.converter.get_new_torque_limit(
        ) * self.cc_cfg.get("tq_mul", 1)

        self.set_action_spaces()

    def set_action_spaces(self):
        cfg = self.cc_cfg
        self.vf_dim = 0
        self.meta_pd_dim = 0
        self.ndof = self.model.actuator_ctrlrange.shape[0]
        body_id_list = self.model.geom_bodyid.tolist()
        if cfg.residual_force:
            if cfg.residual_force_mode == "implicit":
                self.vf_dim = 6
            else:

                self.vf_bodies, self.vf_geoms, self.body_vf_dim, self.vf_dim, self.meta_pd_dim= [],[], 0, 0, 0
                for idx in range(self.num_people):
                    if cfg.residual_force_bodies == "all":
                        self.vf_bodies += [
                            i + f"_{idx:02d}" for i in SMPL_BONE_ORDER_NAMES
                        ]
                    else:
                        raise NotImplementedError

                    self.vf_geoms += [
                        body_id_list.index(self.model._body_name2id[body])
                        for body in self.vf_bodies
                    ]

                    if cfg.meta_pd:
                        self.meta_pd_dim += 2 * 15
                    elif cfg.meta_pd_joint:
                        self.meta_pd_dim += 2 * self.jkp.shape[0]
                self.body_vf_dim = 6 + cfg.residual_force_torque * 3
                self.vf_dim += (self.body_vf_dim * len(self.vf_bodies) *
                                cfg.get("residual_force_bodies_num", 1))

        self.action_dim = self.ndof + self.vf_dim + self.meta_pd_dim
        self.action_space = spaces.Box(
            low=-np.ones(self.action_dim),
            high=np.ones(self.action_dim),
            dtype=np.float32,
        )

    def rfc_explicit(self, vf):
        qfrc = np.zeros(self.data.qfrc_applied.shape)
        num_each_body = self.cc_cfg.get("residual_force_bodies_num", 1)

        residual_contact_projection = self.cc_cfg.get(
            "residual_contact_projection", False)
        for i, body in enumerate(self.vf_bodies):
            body_id = self.model._body_name2id[body]
            for idx in range(num_each_body):
                contact_point = vf[(i * num_each_body + idx) *
                                   self.body_vf_dim:(i * num_each_body + idx) *
                                   self.body_vf_dim + 3]

                force = (vf[(i * num_each_body + idx) * self.body_vf_dim + 3:
                            (i * num_each_body + idx) * self.body_vf_dim + 6] *
                         self.cc_cfg.residual_force_scale)
                torque = (
                    vf[(i * num_each_body + idx) * self.body_vf_dim +
                       6:(i * num_each_body + idx) * self.body_vf_dim + 9] *
                    self.cc_cfg.residual_force_scale
                    if self.cc_cfg.residual_force_torque else np.zeros(3))

                contact_point = self.pos_body2world(body, contact_point)

                force = self.vec_body2world(body, force)
                torque = self.vec_body2world(body, torque)

                # print(np.linalg.norm(force), np.linalg.norm(torque))
                mjf.mj_applyFT(
                    self.model,
                    self.data,
                    force,
                    torque,
                    contact_point,
                    body_id,
                    qfrc,
                )

        self.data.qfrc_applied[:] = qfrc

    def load_context(self, data_dict):
        self.context_dict = [{
            k: v.squeeze().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in data_item.items()
        } for data_item in data_dict]

        self.reset_robot()
        self.targets = []
        self.gt_targets = []
        for p in range(self.num_people):
            self.context_dict[p][
                'len'] = self.context_dict[p]['pose_aa'].shape[0] - 2
            gt_qpos = smpl_to_qpose_multi(
                self.context_dict[p]['pose_aa'],
                offset=self.model.body_pos[p * 24 + 1],
                mujoco_body_order=self.mujoco_body_order,
                trans=self.context_dict[p]['trans'],
                count_offset=True)

            init_qpos = smpl_to_qpose_multi(
                self.context_dict[p]['init_pose_aa'][None, ],
                offset=self.model.body_pos[p * 24 + 1],
                mujoco_body_order=self.mujoco_body_order,
                trans=self.context_dict[p]['init_trans'][None, ],
                count_offset=True)

            self.context_dict[p]["qpos"] = gt_qpos

            self.targets.append(self.humanoids[p].qpos_fk(
                torch.from_numpy(init_qpos)))
            self.gt_targets.append(self.humanoids[p].qpos_fk(
                torch.from_numpy(gt_qpos)))
            # self.prev_humor_state = {
            #     k: data_dict[k][:, 0:1, :].clone()
            #     for k in self.motion_prior.data_names
            # }
            # self.cur_humor_state = self.prev_humor_state

        # self.target.update({
        #     k: data_dict[k][:, 0:1, :].clone()
        #     for k in self.motion_prior.data_names
        # })  # Initializing target

        # self.load_camera_params()

        # self.humanoid.update_model(self.model)

    def reset_robot(self):
        self.num_people = len(self.context_dict)
        self.models = []
        self.humanoids = []
        self.grad2dlosses = []
        betas, genders = [], []
        for p in range(self.num_people):
            beta = self.context_dict[p]["beta"].copy()
            gender = self.context_dict[p]["gender"].copy()
            betas.append(beta[0:1, :]), genders.append(gender[0])

            self.smpl_robot_orig.load_from_skeleton(
                torch.from_numpy(beta[0:1, :]).float())
            h_model = mujoco_py.load_model_from_xml(
                self.smpl_robot_orig.export_xml_string().decode("utf-8"))
            self.models.append(h_model)
            self.humanoids.append(Humanoid(model=h_model))

            self.weight = self.smpl_robot.weight

            self.hb.update_model(torch.from_numpy(beta[0:1, :16]),
                                 torch.tensor(gender[0:1]))
            camera_params = {
                "full_R": np.eye(3),
                "full_t": np.zeros(3),
                "K": np.eye(3),
                "img_w": 1980, 
                "img_h": 1080
            }
            self.hb.update_projection(camera_params, self.smpl2op_map, MUJOCO_2_SMPL)
            self.grad2dloss = egrad(self.hb.proj_2d_loss)
            self.grad2dlosses.append(egrad(self.hb.proj_2d_loss))

        self.smpl_robot.load_from_skeleton(torch.from_numpy(
            np.concatenate(betas)).float(),
                                           gender=genders)
        xml_str = self.smpl_robot.export_xml_string().decode("utf-8")
        self.reload_sim_model(xml_str)
        self.set_action_spaces()

        self.torque_lim_all = np.tile(self.torque_lim, self.num_people)
        self.person_dof = int(self.ndof / self.num_people)
        self.person_vf_dim = int(self.vf_dim / self.num_people)
        self.person_meta_pd_dim = int(self.meta_pd_dim / self.num_people)
        self.person_ctrl_dim = self.person_dof + self.person_vf_dim + self.person_meta_pd_dim
        return xml_str

    def set_model_params(self):
        if self.cc_cfg.action_type == 'torque' and hasattr(
                self.cc_cfg, 'j_stiff'):
            self.model.jnt_stiffness[1:] = self.cc_cfg.j_stiff
            self.model.dof_damping[6:] = self.cc_cfg.j_damp

    def get_obs(self):
        ar_obs = self.get_ar_obs_v1()
        return ar_obs

    def get_cc_obs(self):
        return super().get_obs()

    def get_full_obs_v2(self, delta_t=0):
        data = self.data
        p_obs = []
        for p in range(self.num_people):
            obs = []
            qpos = data.qpos[(self.qpos_lim * p):(self.qpos_lim *
                                                  (p + 1))].copy()
            qvel = data.qvel[(self.qvel_lim * p):(self.qvel_lim *
                                                  (p + 1))].copy()

            # transform velocity
            qvel[:3] = transform_vec(
                qvel[:3], qpos[3:7],
                self.cc_cfg.obs_coord).ravel()  # body angular velocity

            curr_root_quat = self.remove_base_rot(qpos[3:7])
            hq = get_heading_q(curr_root_quat)
            obs.append(hq)  # obs: heading (4,)
            # ZL : why use heading????. Should remove this...

            ######## target body pose #########
            target_body_qpos = self.get_expert_qpos(
                delta_t=1 + delta_t, idx=p).copy()  # target body pose (1, 76)
            target_quat = self.get_expert_wbquat(delta_t=1 + delta_t,
                                                 idx=p).reshape(-1, 4).copy()
            target_jpos = self.get_expert_joint_pos(delta_t=1 + delta_t,
                                                    idx=p).copy()
            curr_jpos = self.data.body_xpos[(1 + 24 * p):(24 * (p + 1) +
                                                          1)].copy()
            cur_quat = self.data.body_xquat[(1 + 24 * p):(24 * (p + 1) +
                                                          1)].copy()

            ################ Body pose and z ################
            target_root_quat = self.remove_base_rot(target_body_qpos[3:7])
            qpos[3:7] = de_heading(curr_root_quat)  # deheading the root
            diff_qpos = target_body_qpos.copy()
            diff_qpos[2] -= qpos[2]
            diff_qpos[7:] -= qpos[7:]
            diff_qpos[3:7] = quaternion_multiply(
                target_root_quat, quaternion_inverse(curr_root_quat))

            obs.append(
                target_body_qpos[2:])  # obs: target z + body pose (1, 74)
            obs.append(qpos[2:])  # obs: target z +  body pose (1, 74)
            obs.append(diff_qpos[2:])  # obs:  difference z + body pose (1, 74)

            ################ vels ################
            # vel
            qvel[:3] = transform_vec(
                qvel[:3], curr_root_quat, self.cc_cfg.obs_coord
            ).ravel(
            )  # ZL: I think this one has some issues. You are doing this twice.
            if self.cc_cfg.obs_vel == "root":
                obs.append(qvel[:6])
            elif self.cc_cfg.obs_vel == "full":
                obs.append(qvel)  # full qvel, 75

            ################ relative heading and root position ################
            rel_h = get_heading(target_root_quat) - get_heading(curr_root_quat)
            if rel_h > np.pi:
                rel_h -= 2 * np.pi
            if rel_h < -np.pi:
                rel_h += 2 * np.pi
            # obs: heading difference in angles (1, 1)
            obs.append(np.array([rel_h]))

            rel_pos = target_root_quat[:3] - qpos[:
                                                  3]  # ZL: this is wrong. Makes no sense. Should be target_root_pos. Should be fixed.
            rel_pos = transform_vec(rel_pos, curr_root_quat,
                                    self.cc_cfg.obs_coord).ravel()
            obs.append(rel_pos[:2])  # obs: relative x, y difference (1, 2)

            ################ target/difference joint positions ################

            # translate to body frame (zero-out root)
            r_jpos = curr_jpos - qpos[None, :3]
            r_jpos = transform_vec_batch(
                r_jpos, curr_root_quat,
                self.cc_cfg.obs_coord)  # body frame position
            # obs: target body frame joint position (1, 72)
            obs.append(r_jpos.ravel())
            diff_jpos = target_jpos.reshape(-1, 3) - curr_jpos
            diff_jpos = transform_vec_batch(diff_jpos, curr_root_quat,
                                            self.cc_cfg.obs_coord)
            obs.append(diff_jpos.ravel(
            ))  # obs: current diff body frame joint position  (1, 72)

            ################ target/relative global joint quaternions ################
            if cur_quat[0, 0] == 0:
                cur_quat = target_quat.copy()

            r_quat = cur_quat.copy()
            hq_invert = quaternion_inverse(hq)
            hq_invert_batch = np.repeat(
                hq_invert[None, ],
                r_quat.shape[0],
                axis=0,
            )

            obs.append(
                quaternion_multiply_batch(hq_invert_batch, r_quat).ravel()
            )  # obs: current target body quaternion (1, 96) # this contains redundant information
            obs.append(
                quaternion_multiply_batch(quaternion_inverse_batch(cur_quat),
                                          target_quat).ravel()
            )  # obs: current target body quaternion (1, 96)

            if self.cc_cfg.has_shape and self.cc_cfg.get(
                    "has_shape_obs", True):
                obs.append(self.get_expert_shape_and_gender(idx=p))
            p_obs.append(np.concatenate(obs).copy())

        return p_obs

    def get_ar_obs_v1(self):
        qpos = self.data.qpos.copy()
        cfg = self.kin_cfg
        t = self.cur_t
        obs = []
        # curr_qpos = self.data.qpos[:self.qpos_lim].copy()
        # curr_qvel = self.data.qvel[:self.qvel_lim].copy()

        # self.prev_humor_state = copy.deepcopy(self.cur_humor_state)
        # self.cur_humor_state = humor_dict = self.get_humor_dict_obs_from_sim()

        # target_global_dict = {
        #     k: torch.from_numpy(self.context_dict[k][t:(t + 1)].reshape(
        #         humor_dict[k].shape))
        #     for k in self.motion_prior.data_names
        # }

        # humor_local_dict, next_target_local_dict, info_dict = self.motion_prior.canonicalize_input_double(
        #     humor_dict,
        #     target_global_dict,
        #     split_input=False,
        #     return_info=True)

        # curr_body_obs = np.concatenate([
        #     humor_local_dict[k].flatten().numpy()
        #     for k in self.motion_prior.data_names
        # ])

        # hq = get_heading_new(curr_qpos[3:7])  # heading
        # # camera_space_mat = torch.matmul(self.camera_params_torch['full_R'], tR.quaternion_to_matrix(torch.from_numpy(curr_qpos[3:7])))
        # # camera_space_quat = tR.matrix_to_quaternion(camera_space_mat)
        # # hq, y, r = get_pyr(camera_space_quat.numpy()) # camera space heading

        # obs.append(np.array(
        #     [hq]))  # camera space heading. ZL: may need a second look

        # if self.kin_cfg.model_specs.get("use_full_root", False):
        #     obs.append(np.array(
        #         [y, r]))  # camera space heading. ZL: may need a second look

        # obs.append(curr_body_obs)
        # trans_target_local = next_target_local_dict['trans'].flatten().numpy()
        # # trans_target_local[:] = 0
        # if self.kin_cfg.model_specs.get("use_rt", True):
        #     obs.append(trans_target_local)

        # if self.kin_cfg.model_specs.get("use_rr", False):
        #     obs.append(next_target_local_dict['root_orient'].flatten().numpy())

        # if self.kin_cfg.model_specs.get("use_2d", True):
        #     pred_joints2d = humor_dict['pred_joints2d'].squeeze().numpy().copy(
        #     )
        #     pred_joints2d[:, 1] = pred_joints2d[:, 1] / 1080 - 0.5
        #     pred_joints2d[:, 0] = pred_joints2d[:, 0] / 1920 - 0.5

        #     joints2d_gt = self.context_dict['joints2d'][
        #         t + 1].copy()  # This should be t + 1??
        #     joints2d_gt[:, 0] = joints2d_gt[:, 0] / 1920 - 0.5
        #     joints2d_gt[:, 1] = joints2d_gt[:, 1] / 1080 - 0.5

        #     if self.kin_cfg.model_specs.get("use_rel2d", True):
        #         pred_2d_root = pred_joints2d[8, :].copy()  # has to copy!!
        #         pred_joints2d -= pred_2d_root
        #         joints2d_gt[:, :2] -= pred_2d_root

        #     delta_2d = pred_joints2d - joints2d_gt[:, :2]
        #     outliers = joints2d_gt[:,
        #                            2] < 0.3  ### Need a better way to incoporate 2d keypoint confidence
        #     joints2d_gt[outliers, :] = 0
        #     delta_2d[outliers, :] = 0

        #     obs.append(joints2d_gt.flatten())
        #     obs.append(pred_joints2d.flatten())
        #     obs.append(delta_2d.flatten())

        # if self.kin_cfg.model_specs.get("use_3d_grad", False):
        #     proj2dgrad = humor_dict['proj2dgrad'].squeeze().numpy().copy()
        #     proj2dgrad = np.nan_to_num(proj2dgrad, nan=0, posinf=0, neginf=0)
        #     proj2dgrad = np.clip(proj2dgrad, -200, 200)
        #     grad_mul = self.kin_cfg.model_specs.get("grad_mul", 10)
        #     heading_rot = info_dict['world2aligned_rot'].numpy()
        #     trans_grad = (np.matmul(heading_rot, proj2dgrad[:3]) /
        #                   (100 * grad_mul)).squeeze()
        #     root_grad = (sRot.from_matrix(heading_rot) * sRot.from_rotvec(
        #         proj2dgrad[3:6] / (10 * grad_mul))).as_rotvec().squeeze()
        #     body_grad = proj2dgrad[6:69] / (10 * grad_mul)

        #     obs.append(trans_grad)
        #     obs.append(root_grad)
        #     obs.append(body_grad)

        # if self.kin_cfg.model_specs.get("use_sdf", False):
        #     sdf_vals = self.smpl_robot.get_sdf_np(
        #         self.cur_humor_state['joints'].reshape(-1, 3), topk=3)
        #     obs.append(sdf_vals.numpy().flatten())
        # elif self.kin_cfg.model_specs.get("use_dir_sdf", False):
        #     sdf_vals, sdf_dirs = self.smpl_robot.get_sdf_np(
        #         self.cur_humor_state['joints'].reshape(-1, 3),
        #         topk=3,
        #         return_grad=True)

        #     sdf_feat = sdf_vals[:, :, None] * sdf_dirs
        #     obs.append(sdf_feat.numpy().flatten())

        # obs = np.concatenate(obs)
        obs = np.zeros(616)

        return obs

    def step_ar(self, action, dt=1 / 30):
        cfg = self.kin_cfg

        # next_global_out = self.motion_prior.step_state(
        #     self.cur_humor_state, torch.from_numpy(action[None, ]))

        # body_pose_aa = mat2aa(next_global_out['pose_body'].reshape(21, 3,
        #                                                            3)).reshape(
        #                                                                1, 63)
        # root_aa = mat2aa(next_global_out['root_orient'].reshape(1, 3,
        #                                                         3)).reshape(
        #                                                             1, 3)
        # pose_aa = torch.cat(
        #     [root_aa, body_pose_aa,
        #      torch.zeros(1, 6).to(root_aa)], dim=1)
        # qpos = smpl_to_qpose_torch(pose_aa,
        #                            self.model,
        #                            trans=next_global_out['trans'].reshape(
        #                                1, 3),
        #                            count_offset=True)
        # if self.mode == "train" and self.agent.iter < self.agent.num_supervised and self.agent.iter >= 0:
        #     # Dagger
        #     qpos = torch.from_numpy(
        #         self.gt_targets['qpos'][self.cur_t:self.cur_t + 1])
        #     fk_res = self.humanoid.qpos_fk(qpos)
        # else:
        #     fk_res = self.humanoid.qpos_fk(qpos)

        # self.target = fk_res
        # self.target.update(next_global_out)
        self.targets = []
        for p in range(self.num_people):
            qpos = torch.from_numpy(
                self.gt_targets[p]['qpos'][(self.cur_t + 1):(self.cur_t + 2)])
            fk_res = self.humanoids[p].qpos_fk(qpos)
            self.targets.append(fk_res)

    def get_humanoid_pose_aa_trans(self, qpos=None):
        if qpos is None:
            qpos = self.data.qpos.copy()[None]
        pose_aa, trans = qpos_to_smpl(
            qpos, self.model, self.cc_cfg.robot_cfg.get("model", "smpl"))

        return pose_aa, trans

    def get_humor_dict_obs_from_sim(self):
        # Compute humor obs based on current and previous simulation state.
        qpos = self.data.qpos.copy()[None]
        prev_qpos = self.prev_qpos[None]

        # Calculating the velocity difference from simulation
        qpos_stack = np.concatenate([prev_qpos, qpos])
        pose_aa, trans = self.get_humanoid_pose_aa_trans(qpos_stack)
        fk_result = self.humanoid.qpos_fk(torch.from_numpy(qpos_stack),
                                          to_numpy=False)
        trans_batch = torch.from_numpy(trans[None])

        joints = fk_result["wbpos"].reshape(-1, 24, 3)[:,
                                                       MUJOCO_2_SMPL].reshape(
                                                           -1, 72)[:, :66]
        pose_aa_mat = aa2mat(torch.from_numpy(pose_aa.reshape(-1, 3))).reshape(
            1, 2, 24, 4, 4)[..., :3, :3]

        humor_out = {}
        trans_vel, joints_vel, root_orient_vel = estimate_velocities(
            trans_batch,
            pose_aa_mat[:, :, 0],
            joints[None],
            30,
            aa_to_mat=False)

        humor_out['trans_vel'] = trans_vel[:, 0:1, :]
        humor_out['joints_vel'] = joints_vel[:, 0:1, :]
        humor_out['root_orient_vel'] = root_orient_vel[:, 0:1, :]

        humor_out['joints'] = joints[None, 1:2]
        humor_out['pose_body'] = pose_aa_mat[:, 1:2,
                                             1:22]  # contains current qpos

        humor_out['root_orient'] = pose_aa_mat[:, 1:2, 0]
        humor_out['trans'] = trans_batch[:, 1:2]

        ######################## Compute 2D Keypoint projection and 3D keypoint ######################
        joints2d_gt = self.context_dict['joints2d'][self.cur_t + 1].copy()
        inliers = joints2d_gt[:,
                              2] > 0.3  ### Need a better way to incoporate 2d keypoint confidence
        self.hb.update_tgt_joints(joints2d_gt[:, :2], inliers)

        input_vec = np.concatenate(
            [humor_out['trans'].numpy(), pose_aa[1:2].reshape(1, -1, 72)],
            axis=2)

        pred_2d = self.hb.proj2d(fk_result["wbpos"][1:2].reshape(24,
                                                                 3).numpy())
        proj2dgrad = self.grad2dloss(input_vec)
        # with np.errstate(divide='raise'):
        #     try:
        #         proj2dgrad = self.grad2dloss(input_vec)
        #     except FloatingPointError:
        #         import ipdb; ipdb.set_trace()

        # bm, smpl2op_map, aR, atr, R, tr, K = self.bm, self.smpl2op_map, self.camera_params_torch[
        #     'aR'], self.camera_params_torch['atr'], self.camera_params_torch[
        #         'R'], self.camera_params_torch['tr'], self.camera_params_torch[
        #             'K']

        # pred_body = self.bm(
        #     pose_body=torch.from_numpy(pose_aa[1:2, 1:22].reshape(1,
        #                                                           -1)).float(),
        #     pose_hand=None,
        #     betas=torch.from_numpy(self.context_dict['beta'][0:1]).float(),
        #     root_orient=torch.from_numpy(pose_aa[1:2,
        #                                          0:1].reshape(1, -1)).float(),
        #     trans=torch.from_numpy(trans[1:2]).float())

        # pred_vertices = pred_body.v.double()
        # pred_vertices = pred_vertices.reshape(1, -1, 3)
        # pred_vertices = pred_vertices @ aR.T + atr

        # pred_vertices = pred_vertices @ R.T + tr

        # pred_joints3d = pred_body.Jtr.double()
        # pred_joints3d = pred_joints3d.reshape(1, -1, 3)
        # pred_joints3d = pred_joints3d[:, smpl2op_map, :]
        # pred_joints3d = pred_joints3d @ aR.T + atr
        # pred_joints3d = pred_joints3d @ R.T + tr
        # pred_joints2d = pred_joints3d @ (K.T)[None]
        # z = pred_joints2d[:, :, 2:]
        # pred_joints2d = pred_joints2d[:, :, :2] / z

        # humor_out["pred_joints2d"] = pred_joints2d[None, ]
        # humor_out["pred_vertices"] = pred_vertices[None, ]
        # pred_joints2d[:, self.smpl2op_map < 22]
        # import ipdb; ipdb.set_trace()
        # diff = pred_body.Jtr.double()[0, :22, :] - fk_result["wbpos"].reshape(-1, 24, 3)[0, MUJOCO_2_SMPL][:22]

        humor_out["pred_joints2d"] = torch.from_numpy(pred_2d[None, ])
        humor_out["proj2dgrad"] = torch.from_numpy(proj2dgrad)

        return humor_out

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
        self.camera_params = {
            "K": K,
            "R": R,
            "tr": tr,
            "aR": aR,
            "atr": atr,
            "full_R": full_R,
            "full_t": full_t
        }
        self.camera_params_torch = {
            k: torch.from_numpy(v).double()
            for k, v in self.camera_params.items()
        }

    def step(self, a, kin_override=False):
        cfg = self.kin_cfg
        cc_cfg = self.cc_cfg
        # record prev state
        # self.prev_qpos = self.get_humanoid_qpos()
        # self.prev_qvel = self.get_humanoid_qvel()
        # self.prev_bquat = self.bquat.copy()
        # self.prev_hpos = self.get_head().copy()

        self.step_ar(a.copy())

        cc_obs = self.get_cc_obs()

        cc_as = []
        for p in range(self.num_people):
            cc_ob = cc_obs[p]
            cc_ob = self.cc_running_state(cc_ob, update=False)
            cc_a = self.cc_policy.select_action(
                torch.from_numpy(cc_ob)[None, ],
                mean_action=True)[0].numpy()  # CC step
            cc_as.append(cc_a)

        cc_as = np.concatenate(cc_as)

        if flags.debug:
            # self.do_simulation(cc_a, self.frame_skip)
            self.data.qpos[:] = np.concatenate([
                self.gt_targets[p]['qpos'][self.cur_t + 1]
                for p in range(self.num_people)
            ])

            self.sim.forward()  # debug
        else:
            # self.data.qpos[:self.qpos_lim] = self.gt_targets[0]['qpos'][self.cur_t + 1]  # debug
            # self.data.qpos[self.qpos_lim:self.qpos_lim * 2] = self.gt_targets[1]['qpos'][self.cur_t + 1]  # debug
            # self.sim.forward()  # debug

            # self.do_simulation(cc_as, self.frame_skip)
            try:
                self.do_simulation(cc_as, self.frame_skip)
            except Exception as e:
                print("Exception in do_simulation", e, self.cur_t)
                fail = True

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
            body_diff = np.mean(
                [self.calc_body_diff(idx=p) for p in range(self.num_people)])
            if self.mode == "train":
                body_gt_diff = np.mean([
                    self.calc_body_gt_diff(idx=p)
                    for p in range(self.num_people)
                ])

                fail = body_diff > 5 or body_gt_diff > (
                    10 if (self.agent.iter < self.agent.num_warmup
                           and self.agent.iter != -1) else 10)
            else:
                fail = body_diff > 50

            # fail = body_diff > 10
            # print(fail, self.cur_t)
            # fail = False
        else:
            raise NotImplemented()
        # if flags.debug:
        # fail = False

        smallest_seq_len = np.min(
            [self.context_dict[p]['len'] for p in range(self.num_people)])
        end = (self.cur_t >= cc_cfg.env_episode_len) or (
            self.cur_t + self.start_ind >= smallest_seq_len)
        done = fail or end

        percent = self.cur_t / smallest_seq_len
        obs = self.get_obs()
        return obs, reward, done, {
            'fail': fail,
            'end': end,
            "percent": percent
        }

    def get_humanoid_qpos(self, idx=0):
        return self.data.qpos.copy()[(self.qpos_lim * idx):(self.qpos_lim *
                                                            (idx + 1))]

    def get_humanoid_qvel(self, idx=0):
        return self.data.qvel.copy()[(self.qvel_lim * idx):(self.qvel_lim *
                                                            (idx + 1))]

    def compute_torque(self, ctrl, i_iter=0):
        cfg = self.cc_cfg
        dt = self.model.opt.timestep

        ctrl_joint = ctrl
        person_dof, person_vf_dim, person_meta_pd_dim, person_ctrl_dim = self.person_dof, self.person_vf_dim, self.person_meta_pd_dim, self.person_ctrl_dim
        torques_acc = []
        for p in range(self.num_people):
            qpos = self.get_humanoid_qpos(idx=p).copy()
            qvel = self.get_humanoid_qvel(idx=p).copy()

            curr_ctrl_joint = ctrl_joint[(p * person_ctrl_dim):(
                p * person_ctrl_dim + person_dof)]

            if self.cc_cfg.action_v == 1:
                base_pos = self.get_expert_kin_pose(
                    delta_t=1, idx=p
                )  # should use the target pose instead of the current pose
                while np.any(base_pos - qpos[7:] > np.pi):
                    base_pos[base_pos - qpos[7:] > np.pi] -= 2 * np.pi
                while np.any(base_pos - qpos[7:] < -np.pi):
                    base_pos[base_pos - qpos[7:] < -np.pi] += 2 * np.pi
            elif self.cc_cfg.action_v == 0:
                base_pos = cfg.a_ref

            target_pos = base_pos + curr_ctrl_joint

            k_p = np.zeros(qvel.shape[0])
            k_d = np.zeros(qvel.shape[0])

            if cfg.meta_pd:
                meta_pds = ctrl[(person_ctrl_dim * p +
                                 person_vf_dim):(person_ctrl_dim * p +
                                                 person_vf_dim +
                                                 person_meta_pd_dim)]
                curr_jkp = self.jkp.copy() * np.clip(
                    (meta_pds[i_iter] + 1), 0, 10)
                curr_jkd = self.jkd.copy() * np.clip(
                    (meta_pds[i_iter + self.sim_iter] + 1), 0, 10)

            elif cfg.meta_pd_joint:
                raise NotImplementedError
            else:
                curr_jkp = self.jkp.copy()
                curr_jkd = self.jkd.copy()

            k_p[6:] = curr_jkp
            k_d[6:] = curr_jkd
            qpos_err = np.concatenate(
                (np.zeros(6), qpos[7:] + qvel[6:] * dt - target_pos))
            qvel_err = qvel

            q_accel = self.compute_desired_accel(qpos_err, qvel_err, k_p, k_d,
                                                 p)
            qvel_err += q_accel * dt
            curr_torque = -curr_jkp * qpos_err[6:] - curr_jkd * qvel_err[6:]
            torques_acc.append(curr_torque)
        torques_acc = np.concatenate(torques_acc)
        return torques_acc

    def compute_desired_accel(self, qpos_err, qvel_err, k_p, k_d, p):
        dt = self.model.opt.timestep
        nv = self.model.nv

        M = np.zeros(nv * nv)
        mjf.mj_fullM(self.model, M, self.data.qM)
        M.resize(nv, nv)

        M = M[(self.qvel_lim * p):(self.qvel_lim * (p + 1)),
              (self.qvel_lim * p):(self.qvel_lim * (p + 1))]
        C = self.data.qfrc_bias.copy()[(self.qvel_lim * p):(self.qvel_lim *
                                                            (p + 1))]
        K_p = np.diag(k_p)
        K_d = np.diag(k_d)
        q_accel = cho_solve(
            cho_factor(M + K_d * dt, overwrite_a=True, check_finite=False),
            -C[:, None] - K_p.dot(qpos_err[:, None]) -
            K_d.dot(qvel_err[:, None]),
            overwrite_b=True,
            check_finite=False,
        )
        return q_accel.squeeze()

    def do_simulation(self, action, n_frames):
        t0 = time.time()
        cfg = self.cc_cfg
        ctrl = action
        person_dof, person_vf_dim, person_meta_pd_dim, person_ctrl_dim = self.person_dof, self.person_vf_dim, self.person_meta_pd_dim, self.person_ctrl_dim
        # import ipdb; ipdb.set_trace()

        # meta_pds = ctrl[(self.ndof + self.vf_dim):(self.ndof + self.vf_dim +
        #                                            self.meta_pd_dim)]
        # print(np.max(meta_pds), np.min(meta_pds))
        self.curr_torque = []
        for i in range(n_frames):
            if cfg.action_type == "position":
                torque = self.compute_torque(ctrl, i_iter=i)
            elif cfg.action_type == "torque":
                torque = ctrl * self.a_scale * 100
            torque = np.clip(torque, -self.torque_lim_all, self.torque_lim_all)

            self.curr_torque.append(torque)
            self.data.ctrl[:] = torque
            """ Residual Force Control (RFC) """
            if cfg.residual_force:
                vfs = []
                for p in range(self.num_people):
                    vf = ctrl[(person_ctrl_dim * p +
                               person_dof):(person_ctrl_dim * p + person_dof +
                                            person_vf_dim)].copy()
                    vfs.append(vf)

                vfs = np.concatenate(vfs)

                if cfg.residual_force_mode == "implicit":
                    self.rfc_implicit(vf)
                else:
                    self.rfc_explicit(vfs)

            self.sim.step()
            # try:
            #     self.sim.step()
            # except Exception as e:
            #     # if flags.debug:
            #     #     import ipdb
            #     #     ipdb.set_trace()
            #     print("Exception in do_simulation step:", e)
            # pass

            # self.render()

        if self.viewer is not None:
            self.viewer.sim_time = time.time() - t0

    def set_mode(self, mode):
        self.mode = mode

    def ar_fail_safe(self):
        self.data.qpos[:self.qpos_lim] = self.context_dict['ar_qpos'][
            self.cur_t + 1]
        # self.data.qpos[:self.qpos_lim] = self.get_target_qpos()
        self.data.qvel[:self.qvel_lim] = self.context_dict['ar_qvel'][
            self.cur_t + 1]
        self.sim.forward()

    def reset_model(self, qpos=None, qvel=None):
        cfg = self.kin_cfg
        ind = 0
        self.start_ind = 0
        qposes = []
        for p in range(self.num_people):
            init_pose_aa = self.context_dict[p]['init_pose_aa']
            init_trans = self.context_dict[p]['init_trans']
            init_qpos = smpl_to_qpose_multi(
                init_pose_aa[None, ],
                offset=self.model.body_pos[p * 24 + 1],
                mujoco_body_order=self.mujoco_body_order,
                trans=init_trans[None, ],
                count_offset=True)
            qposes.append(init_qpos.flatten())

        self.set_state(np.concatenate(qposes), np.zeros_like(self.data.qvel))

        self.prev_qpos = self.get_humanoid_qpos()
        return self.get_obs()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array" or mode == "depth_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self._viewers[mode] = self.viewer
        self.viewer_setup("rgb")

        # full_R, full_t = self.camera_params['full_R'], self.camera_params[
        #     'full_t']
        # distance = np.linalg.norm(full_t)
        # x_axis = full_R.T[:, 0]
        # pos_3d = -full_R.T.dot(full_t)
        # rotation = sRot.from_matrix(full_R).as_euler("XYZ", degrees=True)
        # self.viewer.cam.distance = 2  # + 3 to have better viewing
        # self.viewer.cam.lookat[:] = pos_3d + x_axis
        # self.viewer.cam.azimuth = 90 - rotation[2]
        # self.viewer.cam.elevation = -8

        return self.viewer

    def viewer_setup(self, mode):
        self.viewer.cam.trackbodyid = 1
        # self.viewer.cam.lookat[:2] = self.get_humanoid_qpos()[:2]
        if mode not in self.set_cam_first:
            self.viewer.video_fps = 33
            self.viewer.frame_skip = self.frame_skip
            self.viewer.cam.distance = self.model.stat.extent * 1.2
            self.viewer.cam.elevation = -20
            self.viewer.cam.azimuth = 45
            self.set_cam_first.add(mode)

    def get_expert_qpos(self, delta_t=0, idx=0):
        expert_qpos = self.targets[idx]['qpos'].copy().squeeze()
        return expert_qpos

    def get_target_kin_pose(self, delta_t=0, idx=0):
        return self.get_expert_qpos(idx=idx)[7:]

    def get_expert_joint_pos(self, delta_t=0, idx=0):
        # world joint position
        wbpos = self.targets[idx]['wbpos'].squeeze()
        return wbpos

    def get_expert_com_pos(self, delta_t=0, idx=0):
        # body joint position
        body_com = self.targets[idx]['body_com'].squeeze()
        return body_com

    def get_expert_bquat(self, delta_t=0, idx=0):
        bquat = self.targets[idx]['bquat'].squeeze()
        return bquat

    def get_expert_wbquat(self, delta_t=0, idx=0):
        wbquat = self.targets[idx]['wbquat'].squeeze()
        return wbquat

    def get_expert_shape_and_gender(self, idx=0):
        cfg = self.cc_cfg

        shape = self.context_dict[idx]['beta'][0].squeeze()
        if shape.shape[0] == 10:
            shape = np.concatenate([shape, np.zeros(6)])

        gender = self.context_dict[idx]['gender'][0].squeeze()
        obs = []
        if cfg.get("has_pca", True):
            obs.append(shape)

        obs.append([gender])

        if cfg.get("has_weight", False):
            obs.append([self.weight])

        if cfg.get("has_bone_length", False):
            obs.append(self.smpl_robot.bone_length)

        return np.concatenate(obs)

    def calc_body_diff(self, idx=0):
        cur_wbpos = self.get_wbody_pos(idx=idx).reshape(-1, 3)
        e_wbpos = self.get_expert_joint_pos(idx=idx).reshape(-1, 3)
        diff = cur_wbpos - e_wbpos
        diff *= self.jpos_diffw
        jpos_dist = np.linalg.norm(diff, axis=1).sum()
        return jpos_dist

    def calc_body_ar_diff(self, idx=0):
        cur_wbpos = self.get_wbody_pos().reshape(-1, 3)
        # e_wbpos = self.get_target_joint_pos().reshape(-1, 3)
        e_wbpos = self.context_dict['ar_wbpos'][self.cur_t + 1].reshape(-1, 3)
        diff = cur_wbpos - e_wbpos
        diff *= self.jpos_diffw
        jpos_dist = np.linalg.norm(diff, axis=1).sum()
        return jpos_dist

    def calc_body_gt_diff(self, idx=0):
        cur_wbpos = self.get_wbody_pos(idx=idx).reshape(-1, 3)
        e_wbpos = self.gt_targets[idx]['wbpos'][self.cur_t].reshape(-1, 3)
        diff = cur_wbpos - e_wbpos
        diff *= self.jpos_diffw
        jpos_dist = np.linalg.norm(diff, axis=1).sum()
        return jpos_dist

    def get_wbody_pos(self, selectList=None, idx=0):
        body_pos = []
        if selectList is None:
            # body_names = self.model.body_names[1:] # ignore plane
            return self.data.body_xpos[(1 +
                                        24 * idx):(1 + 24 *
                                                   (idx + 1))].copy().ravel()
        else:
            body_names = selectList
        for body in body_names:
            bone_idx = self.model._body_name2id[body]
            bone_vec = self.data.body_xpos[bone_idx]
            body_pos.append(bone_vec)
        return np.concatenate(body_pos)

    def get_expert_attr(self, attr, ind, idx=0):
        return self.context_dict[idx][attr][ind].copy()

    def get_body_quat(self):
        qpos = self.get_humanoid_qpos()
        body_quat = []
        for body in self.model.body_names[1:self.body_lim]:
            if body.startswith("Pelvis"):
                start, end = self.body_qposaddr[body]
                body_quat.append(qpos[start:start + 4])
            else:
                start, end = self.body_qposaddr[body]
                euler = np.zeros(3)
                euler[:end - start] = qpos[start:end]
                quat = quaternion_from_euler(euler[0], euler[1], euler[2],
                                             "rzyx")
                body_quat.append(quat)
        body_quat = np.concatenate(body_quat)

        return body_quat

    def get_expert_kin_pose(self, delta_t=0, idx=0):
        return self.get_expert_qpos(delta_t=delta_t, idx=idx)[7:]


if __name__ == "__main__":
    pass
