'''
File: /kin_policy_humor.py
Created Date: Wednesday February 16th 2022
Author: Zhengyi Luo
Comment:
-----
Last Modified: Wednesday February 16th 2022 9:53:25 am
Modified By: Zhengyi Luo at <zluo2@cs.cmu.edu>
-----
Copyright (c) 2022 Carnegie Mellon University, KLab
-----
'''

import torch.nn as nn
import torch
import pickle

from tqdm import tqdm

from uhc.khrylib.rl.core.distributions import DiagGaussian
from uhc.utils.math_utils import *
from uhc.utils.torch_ext import gather_vecs

from uhc.utils.flags import flags
import copy
from uhc.utils.torch_ext import get_scheduler

from uhc.models.kin_policy import KinPolicy
from embodiedpose.models import model_dict
from uhc.utils.torch_geometry_transforms import (angle_axis_to_rotation_matrix as aa2mat, rotation_matrix_to_angle_axis as mat2aa)
from embodiedpose.models.humor.utils.velocities import estimate_velocities
from uhc.utils.torch_ext import isNpArray, dict_to_torch
import gc


class KinPolicyHumorRes(KinPolicy):

    def __init__(self, cfg, data_sample, device, dtype, agent, mode="train"):
        super(KinPolicy, self).__init__()
        self.cfg = cfg
        self.policy_specs = cfg.get("policy_specs", {})
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.agent = agent
        self.type = 'gaussian'
        fix_std = cfg.policy_specs['fix_std']
        log_std = np.array(cfg.policy_specs['log_std'])

        data_sample = {k: v.to(self.device).type(self.dtype) if isinstance(v, torch.Tensor) else v for k, v in data_sample.items()}

        self.kin_net = model_dict[cfg.model_name](cfg, data_sample=data_sample, device=device, dtype=dtype, mode=mode, agent=agent)
        self.setup_optimizers()

        self.to(device)
        # self.obs_lim = self.kin_net.get_obs(data_sample, 0)[0].shape[1]
        self.state_dim = state_dim = self.kin_net.state_dim
        self.action_dim = action_dim = self.kin_net.action_dim
        self.use_tcn = self.kin_net.use_tcn
        self.mode = mode
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std, requires_grad=not fix_std)

    def forward(self, all_state, gather=True):
        if self.mode == "test" or not self.kin_net.use_rnn:
            action_mean = self.get_action(all_state)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
        elif self.mode == 'train':
            device, dtype = all_state.device, all_state.dtype
            s_ctx = torch.zeros((self.num_episode * self.max_episode_len, self.state_dim), device=device)
            if gather:
                all_state_vec = self.scatter_vecs([all_state])[0]
            else:
                all_state_vec = all_state

            s_ctx = all_state_vec.view(-1, self.max_episode_len, self.state_dim).transpose(0, 1).contiguous()

            self.kin_net.action_rnn.initialize(s_ctx.shape[1])

            action_mean_acc = []
            for i in range(s_ctx.shape[0]):
                curr_state = s_ctx[i]
                action_ar = self.get_action(curr_state)
                action_mean_acc.append(action_ar)
            action_mean_acc = torch.stack(action_mean_acc).transpose(0, 1).contiguous().view(-1, action_mean_acc[0].shape[-1])

            if gather:
                s_gather_indices = torch.LongTensor(np.tile(self.indices[:, None], (1, action_mean_acc.shape[-1]))).to(self.device)
                action_mean = torch.gather(action_mean_acc, 0, s_gather_indices)
            else:
                action_mean = action_mean_acc

            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)

        return DiagGaussian(action_mean, action_std), action_mean, action_std

    def init_context(self, data_dict, random_cam=False):

        self.action_log_std[:] = torch.tensor(self.cfg.policy_specs['log_std']).to(self.device)
        data_dict = dict_to_torch(data_dict, dtype=self.dtype, device=self.device)

        ar_context = self.kin_net.init_states(data_dict, random_cam=random_cam)

        self.reset()

        if self.kin_net.use_rnn:
            self.reset_rnn(1)

        return ar_context

    def scatter_vecs(self, vecs_list):
        res = []
        for vec in vecs_list:
            assert (len(vec.shape) == 2)
            s_gather_indices = torch.LongTensor(np.tile(self.indices[:, None], (1, vec.shape[-1]))).to(self.device)
            s_vec = torch.zeros((self.num_episode * self.max_episode_len, vec.shape[-1]), device=self.device)
            s_vec.scatter_(0, s_gather_indices, vec.clone())
            res.append(s_vec)
        return res

    def update_supervised(self, all_state, humor_target_vec, sim_humor_vec, seq_data, num_epoch=20):
        # import ipdb
        # ipdb.set_trace()
        num_samples = all_state.shape[0]
        rewards = seq_data[2]
        all_state_s, humor_target_vec_s, sim_humor_vec_s, rewards_s = self.scatter_vecs([all_state, humor_target_vec, sim_humor_vec, rewards[:, None]])

        sim_humor_dict = self.kin_net.motion_prior.split_input(sim_humor_vec_s, convert_rots=False)
        humor_target_dict = self.kin_net.motion_prior.split_input(humor_target_vec_s[:, :sim_humor_vec_s.shape[1]], convert_rots=False)

        beta = humor_target_vec_s[:, -16:]
        points = humor_target_vec_s[:, sim_humor_vec_s.shape[1]:-16].reshape(humor_target_vec_s.shape[0], -1, 3)
        humor_target_dict['points3d'] = points[:, :4096, :]
        humor_target_dict['joints2d'] = points[:, 4096:(4096 + 12), :].reshape(-1, 12, 3)
        humor_target_dict['wbpos_cam'] = points[:, (4096 + 12):, :].reshape(-1, 14, 3)

        humor_target_dict['beta'] = beta
        humor_target_dict['rewards'] = rewards_s

        humor_target_dict['indices_end'] = self.indices_end
        humor_target_dict['indices'] = self.indices
        humor_target_dict['max_episode_len'] = self.max_episode_len
        humor_target_dict['indices'] = self.indices
        humor_target_dict['sim_humor_dict'] = sim_humor_dict

        pbar = tqdm(range(num_epoch))
        for epoch in pbar:
            ### Prior Loss
            self.kin_net.set_sim(sim_humor_dict)
            _, action_mean, _ = self.forward(all_state_s, gather=False)

            next_global_out = self.kin_net.step(action_mean[:, :69])
            next_global_out['trans'] = next_global_out['trans'].view(-1, self.max_episode_len, 3)
            next_global_out['root_orient'] = next_global_out['root_orient'].view(-1, self.max_episode_len, 3, 3)
            next_global_out['pose_body'] = next_global_out['pose_body'].view(-1, self.max_episode_len, 21, 3, 3)
            next_global_out['joints'] = next_global_out['joints'].view(-1, self.max_episode_len, 22, 3)

            trans_vel, joints_vel, root_orient_vel = estimate_velocities(next_global_out['trans'], next_global_out['root_orient'], next_global_out['joints'], data_fps=30, aa_to_mat=False)
            next_global_out['trans_vel'] = trans_vel
            next_global_out['joints_vel'] = joints_vel
            next_global_out['root_orient_vel'] = root_orient_vel
            B = next_global_out['trans'].shape[0]
            next_global_out = {k: v.reshape(B, self.max_episode_len, -1) for k, v in next_global_out.items()}

            total_loss, loss_dict, loss_unweighted_dict = self.kin_net.compute_loss_seq(next_global_out, humor_target_dict, epoch=epoch, max_epoch=num_epoch)

            self.optimizer.zero_grad()
            total_loss.backward()  # Testing GT
            self.optimizer.step()  # Testing GT

            if self.use_tcn:
                gt_bpos = humor_target_dict['wbpos_cam'].reshape(B, self.max_episode_len, 14, 3).clone()
                gt_bpos = gather_vecs([gt_bpos.reshape(-1, 14 * 3)], self.indices)[0].reshape(-1, 14, 3)
                gt_traj = gt_bpos[..., 7:8, :]

                gt_body_pos = gt_bpos - gt_traj
                tcn_input = all_state[:, (self.kin_net.is_root_obs == 3)[0]].reshape(-1, self.kin_net.num_context, 12, self.kin_net.traj_in_features).float()

                root_mpjpe_acc, body_mpjpe_acc = [], []
                for tcn_input_idx in torch.split(torch.randperm(tcn_input.shape[0]), 1024, 0):
                    tcn_input_chunk = tcn_input[tcn_input_idx]
                    gt_body_pos_chunk = gt_body_pos[tcn_input_idx]
                    gt_traj_chunk = gt_traj[tcn_input_idx]

                    if self.kin_net.model_specs.get("tcn_3dpos", False):
                        kp_2d_feats_local = tcn_input_chunk.clone()
                        kp_2d_feats_local[..., 2:] = tcn_input_chunk[..., 2:] - tcn_input_chunk[..., 7:8, 2:]
                        if self.kin_net.model_specs.get("tcn_body2d", False):
                            pred_body_pos = self.kin_net.tcn_body_1f(kp_2d_feats_local[..., :2])
                        else:
                            pred_body_pos = self.kin_net.tcn_body_1f(kp_2d_feats_local)
                        pred_traj = self.kin_net.tcn_traj_1f(tcn_input_chunk)
                    else:
                        pred_body_pos = self.kin_net.tcn_body_1f(tcn_input_chunk)
                        pred_traj = self.kin_net.tcn_traj_1f(tcn_input_chunk)

                    root_mpjpe = torch.norm(gt_traj_chunk.reshape(-1, 3) - pred_traj.reshape(-1, 3), dim=1)

                    body_mpjpe = torch.norm(gt_body_pos_chunk.reshape(-1, 14, 3) - pred_body_pos.reshape(-1, 14, 3), dim=2).mean(-1)

                    tcn_loss = (root_mpjpe + body_mpjpe).mean()

                    self.kin_net.tcn_opt.zero_grad()
                    tcn_loss.backward()  # Testing GT
                    self.kin_net.tcn_opt.step()  # Testing GT
                    root_mpjpe_acc.append(root_mpjpe.mean().detach().cpu().item())
                    body_mpjpe_acc.append(body_mpjpe.mean().detach().cpu().item())

                loss_dict['root_mpjpe'] = np.mean(root_mpjpe_acc)
                loss_dict['body_mpjpe'] = np.mean(body_mpjpe_acc)

            pbar.set_description_str(f"Per-step loss: {total_loss.cpu().detach().numpy():.3f} [{ {k: f'{v:.3f}'  for k, v in loss_dict.items()}}] lr: {self.scheduler.get_last_lr()[0]:.5f} | num_s {num_samples} | max_ep {self.max_episode_len}")
            gc.collect()
            torch.cuda.empty_cache()
            del total_loss, loss_dict, loss_unweighted_dict, next_global_out, self.kin_net.sim,

    def train_full_supervised(self, cfg, dataset, device, dtype, num_epoch=20, eval_freq=5, scheduled_sampling=0, save_func=None):
        pbar = tqdm(range(num_epoch))
        self.kin_net.set_schedule_sampling(scheduled_sampling)
        t_min = 30
        t_max = self.cfg.fr_num

        for epoch in pbar:
            fr_num = int((t_max - t_min) / num_epoch * epoch + t_min)
            # fr_num = self.cfg.fr_num
            # print(fr_num)
            # train_loader = dataset.sampling_loader(num_samples=cfg.num_samples, batch_size=cfg.batch_size, num_workers=10, fr_num=self.cfg.fr_num)
            train_loader = dataset.sampling_loader(num_samples=cfg.num_samples, batch_size=cfg.batch_size, num_workers=10, fr_num=fr_num)
            self.kin_net.per_epoch_update(epoch)
            self.kin_net.training_epoch(train_loader, epoch, num_epoch)
            if (epoch + 1) % eval_freq == 0:
                self.kin_net.eval_model(dataset)
                if not save_func is None:
                    save_func(0)

        self.kin_net.eval_model(dataset)