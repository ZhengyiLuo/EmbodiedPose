'''
File: /agent_scene.py
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

from unittest import loader
import joblib
import os.path as osp
import pdb
import sys
import glob
from multiprocessing import Pool
from tqdm import tqdm
import pickle
from collections import defaultdict
import multiprocessing
import math
import time
import os
import torch
import wandb

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(os.getcwd())

from copycat.khrylib.utils import to_device, create_logger, ZFilter
from copycat.khrylib.models.mlp import MLP
from copycat.khrylib.rl.core import estimate_advantages
from copycat.khrylib.utils.torch import *
from copycat.khrylib.utils.memory import Memory
from copycat.khrylib.rl.core import LoggerRL
from copycat.khrylib.rl.core.critic import Value
from copycat.khrylib.utils import get_eta_str
from copycat.utils.flags import flags
from copycat.khrylib.utils.logger import create_logger

from copycat.envs import env_dict
from copycat.models.kin_policy import KinPolicy
from copycat.agents.agent_uhm import AgentUHM

from sceneplus.data_loaders import data_dict
from sceneplus.envs import env_dict
from sceneplus.models import policy_dict
from sceneplus.core.reward_function import reward_func
from sceneplus.core.trajbatch_humor import TrajBatchHumor
from copycat.utils.torch_ext import isNpArray
from copycat.smpllib.smpl_eval import compute_metrics


class AgentMulti(AgentUHM):
    def __init__(self, cfg, dtype, device, mode="train", checkpoint_epoch=0):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.mode = mode

        self.global_start_fr = 0
        self.iter = checkpoint_epoch
        self.num_warmup = 2000
        self.num_supervised = 10
        self.fr_num = self.cfg.fr_num
        self.setup_vars()
        self.setup_data_loader()
        self.setup_policy()
        self.setup_env()
        self.setup_value()
        self.setup_optimizer()
        self.setup_logging()
        self.setup_reward()
        self.seed(cfg.seed)
        self.print_config()
        if checkpoint_epoch > 0:
            self.load_checkpoint(checkpoint_epoch)
        elif checkpoint_epoch == -1:
            self.load_curr()
        # self.load_curr()

        self.freq_dict = defaultdict(list)

        super(AgentUHM, self).__init__(
            env=self.env,
            dtype=dtype,
            device=device,
            # running_state=ZFilter((self.env.state_dim, ), clip=5),
            running_state=None,
            custom_reward=self.expert_reward,
            mean_action=cfg.render and not cfg.show_noise,
            render=cfg.render,
            num_threads=cfg.num_threads,
            data_loader=self.data_loader,
            policy_net=self.policy_net,
            value_net=self.value_net,
            optimizer_policy=self.optimizer_policy,
            optimizer_value=self.optimizer_value,
            opt_num_epochs=cfg.policy_specs['num_optim_epoch'],
            gamma=cfg.policy_specs['gamma'],
            tau=cfg.policy_specs['tau'],
            clip_epsilon=cfg.policy_specs['clip_epsilon'],
            policy_grad_clip=[(self.policy_net.parameters(), 40)],
            end_reward=cfg.policy_specs['end_reward'],
            use_mini_batch=False,
            mini_batch_size=0)

        self.fit_single = self.cfg.get("fit_single", True)
        # import ipdb; ipdb.set_trace()
        # if self.iter == 0 and self.mode == "train":
        # self.train_init()
        # self.train_init()

    def setup_vars(self):
        super().setup_vars()

        self.traj_batch = TrajBatchHumor

    def setup_reward(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.expert_reward = expert_reward = reward_func[
            cfg.policy_specs['reward_id']]

    def setup_env(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        """load CC model"""

        with torch.no_grad():
            data_sample = self.data_loader.sample_seq(
                fr_num=20, fr_start=self.global_start_fr)
            data_sample = [{
                k: torch.from_numpy(v).to(device).clone().type(dtype)
                if isNpArray(v) else v
                for k, v in data_item.items()
            } for data_item in data_sample]

            context_sample = self.policy_net.init_context(data_sample,
                                                          random_cam=True)
        self.cc_cfg = cfg.cc_cfg
        self.env = env_dict[self.cfg.env_name](cfg,
                                               init_context=context_sample,
                                               cc_iter=cfg.policy_specs.get(
                                                   'cc_iter', -1),
                                               mode="train",
                                               agent=self)
        self.env.seed(cfg.seed)

    def setup_policy(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        data_sample = self.data_loader.sample_seq(
            fr_num=20, fr_start=self.global_start_fr)

        data_sample = [{
            k: torch.from_numpy(v[None, ]).clone().type(dtype)
            if isNpArray(v) else v
            for k, v in multi_sample.items()
        } for multi_sample in data_sample]

        self.policy_net = policy_net = policy_dict[cfg.policy_name](
            cfg,
            data_sample,
            device=device,
            dtype=dtype,
            mode=self.mode,
            agent=self)
        to_device(device, self.policy_net)

    def setup_value(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        state_dim = self.policy_net.state_dim
        action_dim = self.env.action_space.shape[0]
        self.value_net = Value(
            MLP(state_dim, self.cc_cfg.value_hsize, self.cc_cfg.value_htype))
        to_device(device, self.value_net)

    def setup_data_loader(self):
        cfg = self.cfg
        train_files_path = cfg.data_specs.get("train_files_path", [])
        test_files_path = cfg.data_specs.get("test_files_path", [])
        self.train_data_loaders, self.test_data_loaders, self.data_loader = [], [], None

        if len(train_files_path) > 0:
            for train_file, dataset_name in train_files_path:

                data_loader = data_dict[dataset_name](cfg, [train_file],
                                                      multiproess=False)
                self.train_data_loaders.append(data_loader)

        if len(test_files_path) > 0:
            for test_file, dataset_name in test_files_path:
                data_loader = data_dict[dataset_name](cfg, [test_file],
                                                      multiproess=False)
                self.test_data_loaders.append(data_loader)

        self.data_loader = np.random.choice(self.train_data_loaders)

    def eval_seq(self, take_keys, loader):
        curr_env = self.env

        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                res = defaultdict(list)
                self.policy_net.set_mode('test')
                curr_env.set_mode('test')
                context_sample = loader.get_sample_from_keys(
                    take_keys=take_keys, full_sample=True, return_batch=True)

                curr_env.load_context(
                    self.policy_net.init_context(context_sample))
                state = curr_env.reset()

                if self.running_state is not None:
                    state = self.running_state(state)
                for t in range(10000):

                    res['gt'].append(
                        np.concatenate([
                            curr_env.context_dict[p]['qpos'][self.env.cur_t]
                            for p in range(self.env.num_people)
                        ]))
                    res['target'].append(
                        np.concatenate([
                            curr_env.targets[p]['qpos']
                            for p in range(self.env.num_people)
                        ]))
                    res['pred'].append(
                        np.concatenate([
                            curr_env.get_humanoid_qpos(idx=p)
                            for p in range(self.env.num_people)
                        ]))
                    res["gt_jpos"].append(
                        np.concatenate([
                            self.env.gt_targets[p]['wbpos'][
                                self.env.cur_t].copy()
                            for p in range(self.env.num_people)
                        ]))
                    res["pred_jpos"].append(
                        np.concatenate([
                            self.env.get_wbody_pos(idx=p).copy()
                            for p in range(self.env.num_people)
                        ]))

                    state_var = tensor(state).unsqueeze(0).double()
                    trans_out = self.trans_policy(state_var)
                    action = self.policy_net.select_action(
                        trans_out, mean_action=True)[0].numpy()
                    action = int(
                        action
                    ) if self.policy_net.type == 'discrete' else action.astype(
                        np.float64)
                    next_state, env_reward, done, info = curr_env.step(action)

                    # c_reward, c_info = self.custom_reward(curr_env, state, action, info)
                    # res['reward'].append(c_reward)
                    if self.cfg.render:
                        curr_env.render()
                    if self.running_state is not None:
                        next_state = self.running_state(next_state)

                    if done:
                        res = {k: np.vstack(v) for k, v in res.items()}
                        # print(info['percent'], context_dict['ar_qpos'].shape[1], loader.curr_key, np.mean(res['reward']))
                        res['percent'] = info['percent']
                        res['fail_safe'] = False
                        res.update(compute_metrics(res, self.env.converter))
                        return res
                    state = next_state

    def sample_worker(self, pid, queue, min_batch_size):
        self.seed_worker(pid)
        memory = Memory()
        logger = self.logger_cls()
        self.policy_net.set_mode('test')
        self.env.set_mode('train')
        freq_dict = defaultdict(list)

        while logger.num_steps < min_batch_size:

            context_sample = self.data_loader.sample_seq(
                fr_num=self.cfg.fr_num)
            # should not try to fix the height during training!!!
            ar_context = self.policy_net.init_context(context_sample,
                                                      random_cam=True)
            self.env.load_context(ar_context)
            state = self.env.reset()
            self.policy_net.reset()

            if self.running_state is not None:
                state = self.running_state(state)
            logger.start_episode(self.env)
            self.pre_episode()

            for t in range(10000):
                state_var = tensor(state).unsqueeze(0).double()
                trans_out = self.trans_policy(state_var)
                # mean_action = self.mean_action or self.env.np_random.binomial(
                #     1, 1 - self.noise_rate)

                mean_action = True
                action = self.policy_net.select_action(trans_out,
                                                       mean_action)[0].numpy()

                action = int(
                    action
                ) if self.policy_net.type == 'discrete' else action.astype(
                    np.float64)
                #################### ZL: Jank Code.... ####################
                # humor_target = np.concatenate([
                #     self.env.context_dict[k][self.env.cur_t + 1].flatten()
                #     for k in self.env.agg_data_names
                # ])
                # sim_humor_state = np.concatenate([
                #     self.env.cur_humor_state[k].numpy().flatten()
                #     for k in self.env.motion_prior.data_names
                # ])
                humor_target = {}
                sim_humor_state = {}

                #################### ZL: Jank Code.... ####################

                next_state, env_reward, done, info = self.env.step(action)

                if self.running_state is not None:
                    next_state = self.running_state(next_state)
                # use custom or env reward
                self.custom_reward = None
                if self.custom_reward is not None:
                    c_reward, c_info = self.custom_reward(
                        self.env, state, action, info)
                    reward = c_reward
                else:
                    c_reward, c_info = 0.0, np.array([0.0])
                    reward = env_reward

                # if flags.debug:
                #     np.set_printoptions(precision=4, suppress=1)
                #     print(c_reward, c_info)

                # add end reward
                if self.end_reward and info.get('end', False):
                    reward += self.env.end_reward
                # logging
                logger.step(self.env, env_reward, c_reward, c_info, info)

                mask = 0 if done else 1
                exp = 1 - mean_action
                self.push_memory(memory, state, action, mask, next_state,
                                 reward, exp, humor_target, sim_humor_state)

                if pid == 0 and self.render:
                    for _ in range(10):
                        self.env.render()

                if done:
                    freq_dict[self.data_loader.curr_key].append(
                        [info['percent'], self.data_loader.fr_start])
                    # print(self.data_loader.fr_start, self.data_loader.curr_key, info['percent'], self.env.cur_t)
                    break

                state = next_state

            logger.end_episode(self.env)
        logger.end_sampling()

        if queue is not None:
            queue.put([pid, memory, logger, freq_dict])
        else:
            return memory, logger, freq_dict

    def push_memory(self, memory, state, action, mask, next_state, reward, exp,
                    humor_target, sim_humor_state):
        v_meta = np.array([
            self.data_loader.curr_take_ind, self.data_loader.fr_start,
            self.data_loader.fr_num
        ])
        memory.push(state, action, mask, next_state, reward, exp, v_meta,
                    humor_target, sim_humor_state)

    def optimize_policy(self, epoch):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.iter = epoch
        t0 = time.time()
        self.pre_epoch_update(epoch)

        if self.iter >= self.num_warmup:
            # Dagger
            self.cfg.lr = 1.e-5
            self.cfg.model_specs['weights']['l1_loss'] = 0
            self.cfg.model_specs['weights'][
                'prior_loss'] = 0.0001 if self.cfg.model_specs.get(
                    "use_prior", False) else 0
            # self.cfg.model_specs['weights']['prior_loss'] = 0
            self.cfg.model_specs['weights']['loss_2d'] = 0.005
            self.cfg.model_specs['weights']['loss_chamfer'] = 50
            self.cfg.policy_specs["num_step_update"] = 0
            self.opt_num_epochs = self.cfg.policy_specs["num_optim_epoch"] = 10
            cfg.policy_specs['min_batch_size'] = 5000
            # cfg.policy_specs['min_batch_size'] = 100
            self.cfg.policy_specs["rl_update"] = True
            self.cfg.policy_specs["step_update"] = False
            self.cfg.fr_num = 300
        else:
            self.cfg.lr = 5.e-4
            self.cfg.model_specs['weights']['l1_loss'] = 5
            self.cfg.model_specs['weights'][
                'prior_loss'] = 0.0001 if self.cfg.model_specs.get(
                    "use_prior", False) else 0
            self.cfg.model_specs['weights']['loss_2d'] = 0
            self.cfg.model_specs['weights']['loss_chamfer'] = 0
            self.cfg.policy_specs["num_step_update"] = 10
            self.cfg.policy_specs["rl_update"] = False
            # cfg.policy_specs['min_batch_size'] = 8000
            cfg.policy_specs['min_batch_size'] = 5000

            self.cfg.fr_num = 500 if self.iter < self.num_supervised else max(
                int(min(self.iter / (self.num_warmup / 2), 1) *
                    self.fr_num), self.cfg.data_specs.get("t_min", 30))

            # cfg.policy_specs['min_batch_size'] = 50
        if self.cfg.lr != self.policy_net.optimizer.param_groups[0]['lr']:
            self.policy_net.setup_optimizers()

        batch, log = self.sample(cfg.policy_specs['min_batch_size'])

        if cfg.policy_specs['end_reward']:
            self.env.end_reward = log.avg_c_reward * cfg.policy_specs[
                'gamma'] / (1 - cfg.policy_specs['gamma'])
        """update networks"""
        t1 = time.time()
        self.update_params(batch)
        t2 = time.time()
        info = {
            'log': log,
            'T_sample': t1 - t0,
            'T_update': t2 - t1,
            'T_total': t2 - t0
        }

        if (self.iter + 1) % cfg.save_n_epochs == 0:
            self.save_checkpoint(self.iter)
            log_eval = self.eval_policy("test")
            info['log_eval'] = log_eval

        if (self.iter + 1) % 5 == 0:
            self.save_curr()

        self.log_train(info)
        joblib.dump(self.freq_dict, osp.join(cfg.result_dir, "freq_dict.pt"))

    def update_params(self, batch):

        t0 = time.time()
        to_train(*self.update_modules)
        states = torch.from_numpy(batch.states).to(self.dtype).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.dtype).to(
            self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype).to(
            self.device)
        masks = torch.from_numpy(batch.masks).to(self.dtype).to(self.device)
        exps = torch.from_numpy(batch.exps).to(self.dtype).to(self.device)
        v_metas = torch.from_numpy(batch.v_metas).to(self.dtype).to(
            self.device)
        humor_target = torch.from_numpy(batch.humor_target).to(self.dtype).to(
            self.device)
        sim_humor_state = torch.from_numpy(batch.sim_humor_state).to(
            self.dtype).to(self.device)

        with to_test(*self.update_modules):
            with torch.no_grad():
                values = self.value_net(
                    self.trans_value(states[:, :self.policy_net.state_dim]))

        seq_data = (masks, v_metas, rewards)
        self.policy_net.set_mode('train')
        self.policy_net.recrete_eps(seq_data)
        """get advantage estimation from the trajectories"""
        print("==================================================>")

        if self.cfg.policy_specs.get("rl_update", False):
            print("RL:")
            advantages, returns = estimate_advantages(rewards, masks, values,
                                                      self.gamma, self.tau)
            self.update_policy(states, actions, returns, advantages, exps)

        if self.cfg.policy_specs.get(
                "init_update", False) or self.cfg.policy_specs.get(
                    "step_update", False) or self.cfg.policy_specs.get(
                        "full_update", False):
            print("Supervised:")

        # if self.cfg.policy_specs.get("init_update", False):
        #     self.policy_net.update_init_supervised(self.cfg, self.data_loader, device=self.device, dtype=self.dtype, num_epoch=int(self.cfg.policy_specs.get("num_init_update", 5)))

        if self.cfg.policy_specs.get("step_update", False):

            self.policy_net.update_supervised(
                states,
                humor_target,
                sim_humor_state,
                seq_data,
                num_epoch=int(self.cfg.policy_specs.get("num_step_update",
                                                        10)))

        # if self.cfg.policy_specs.get("full_update", False):
        #     self.policy_net.train_full_supervised(self.cfg, self.data_loader, device=self.device, dtype=self.dtype, num_epoch=1, scheduled_sampling=0.3)

        # self.policy_net.step_lr()

        return time.time() - t0
