import math
from matplotlib.pyplot import flag
import torch
import numpy as np

from uhc.utils.flags import flags
from uhc.khrylib.utils import (get_angvel_fd, multi_quat_norm, multi_quat_diff, quat_mul_vec, get_qvel_fd_new, de_heading)

from embodiedpose.utils.math_utils import multi_quat_norm_v2, get_angvel_fd, multi_quat_diff
from uhc.smpllib.smpl_mujoco import qpos_to_smpl
from embodiedpose.models.humor.numpy_humor_loss import motion_prior_loss, points3d_loss
from embodiedpose.models.humor.torch_humor_loss import kl_normal


def dynamic_supervision_v1(env, state, action, info):
    # V1 uses GT
    # V1 now does not regulate the action using GT, and only has act_v
    cfg = env.kin_cfg
    ws = cfg.policy_specs['reward_weights']
    w_hp, w_hq, w_hv, w_p, w_jp, w_rp, w_rq, w_act_p, w_act_v = ws.get('w_hp', 1.0), ws.get('w_hq', 1.0),\
         ws.get('w_hv', 0.05), ws.get('w_p', 1.0), ws.get('w_jp', 1.0), ws.get('w_rp', 1.0), ws.get('w_rq', 1.0), ws.get('w_act_p', 1.0), ws.get('w_act_v', 1.0)
    k_hp, k_hq, k_hv, k_p, k_jp, k_rp, k_rq, k_act_p, k_act_v = ws.get('k_hp', 1.0), ws.get('k_hq', 1.0), ws.get('k_hv', 1.0 ), \
        ws.get('k_p', 1.0), ws.get('k_jp', 0.1), ws.get('k_rp', 0.1), ws.get('k_rq', 0.1), ws.get('k_act_p', 0.1), ws.get('k_act_v', 0.1)
    v_ord = ws.get('v_ord', 2)

    ind = env.cur_t

    cur_bquat = env.get_body_quat()
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)
    tgt_bquat, tgt_wbpos = env.target['bquat'], env.target['wbpos']

    pose_quat_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat.flatten(), tgt_bquat.flatten())).mean()
    pose_pos_diff = np.linalg.norm(cur_wbpos - tgt_wbpos, axis=1).mean()

    p_reward = math.exp(-k_p * (pose_quat_diff**2))
    jp_reward = math.exp(-k_jp * (pose_pos_diff**2))

    # Comparing with GT
    gt_bquat = env.ar_context['bquat'][ind].flatten()
    gt_prev_bquat = env.ar_context['bquat'][ind - 1].flatten()
    prev_bquat = env.prev_bquat

    pose_gt_diff = multi_quat_norm_v2(multi_quat_diff(gt_bquat, cur_bquat)).mean()

    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    tgt_bangvel = get_angvel_fd(gt_prev_bquat, gt_bquat, env.dt)
    vel_dist = np.linalg.norm(cur_bangvel - tgt_bangvel, ord=v_ord)
    act_v_reward = math.exp(-k_act_v * (vel_dist**2))

    # rp_dist = np.linalg.norm(tgt_qpos[:3] - act_qpos[:3])
    # rq_dist = multi_quat_norm_v2(multi_quat_diff(tgt_qpos[3:7], act_qpos[3:7])).mean()
    # rq_reward = math.exp(-k_rq * (rq_dist ** 2))
    # rp_reward = math.exp(-k_rp * (rp_dist ** 2))
    gt_p_reward = math.exp(-k_act_p * pose_gt_diff)

    reward = w_p * p_reward + w_jp * jp_reward + w_act_p * gt_p_reward + w_act_v * act_v_reward

    # if flags.debug:
    #     import pdb; pdb.set_trace()
    #     np.set_printoptions(precision=4, suppress=1)
    #     print(reward, np.array([p_reward, jp_reward, gt_p_reward, act_v_reward]))

    return reward, np.array([p_reward, jp_reward, gt_p_reward, act_v_reward])


def dynamic_supervision_v2(env, state, action, info):
    # V2 uses no GT
    # velocity loss is from AR-Net , reguralize the actions by running the model kinematically
    # This thing makes 0 sense rn
    # cfg = env.cfg
    # ws = cfg.policy_specs['reward_weights']
    # w_hp, w_hq, w_hv, w_p, w_jp, w_rp, w_rq, w_act_v, w_act_p = ws.get('w_hp', 1.0), ws.get('w_hq', 1.0),\
    #      ws.get('w_hv', 0.05), ws.get('w_p', 1.0), ws.get('w_jp', 1.0), ws.get('w_rp', 1.0), ws.get('w_rq', 1.0), ws.get('w_act_v', 1.0),  ws.get('w_act_p', 1.0)
    # k_hp, k_hq, k_hv, k_p, k_jp, k_rp, k_rq, k_act_v, k_act_p = ws.get('k_hp', 1.0), ws.get('k_hq', 1.0), ws.get('k_hv', 1.0 ), \
    #     ws.get('k_p', 1.0), ws.get('k_jp', 0.1), ws.get('k_rp', 0.1), ws.get('k_rq', 0.1), ws.get('k_act_v', 0.1), ws.get('k_act_p', 0.1)
    # v_ord = ws.get('v_ord', 2)

    # ind = env.cur_t
    # # Head losses
    # tgt_hpos = env.ar_context['head_pose'][ind]
    # tgt_hvel = env.ar_context['head_vels'][ind]

    # cur_hpos = env.get_head().copy()
    # prev_hpos = env.prev_hpos.copy()

    # hp_dist = np.linalg.norm(cur_hpos[:3] - tgt_hpos[:3])
    # hp_reward = math.exp(-k_hp * (hp_dist ** 2))

    # # head orientation reward
    # hq_dist = multi_quat_norm_v2(multi_quat_diff(cur_hpos[3:], tgt_hpos[3:])).mean()
    # hq_reward = math.exp(-k_hq * (hq_dist ** 2))

    # # head velocity reward
    # # hpvel = (cur_hpos[:3] - prev_hpos[:3]) / env.dt
    # # hqvel = get_angvel_fd(prev_hpos[3:], cur_hpos[3:], env.dt)
    # # hpvel_dist = np.linalg.norm(hpvel - tgt_hvel[:3])
    # # hqvel_dist = np.linalg.norm(hqvel - tgt_hvel[3:])
    # # hv_reward = math.exp(-hpvel_dist - k_hv * hqvel_dist)
    # hv_reward = 0

    # cur_bquat = env.get_body_quat()
    # cur_wbpos = env.get_wbody_pos().reshape(-1, 3)
    # tgt_bquat, tgt_wbpos = env.target['bquat'], env.target['wbpos']

    # pose_quat_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat.flatten(), tgt_bquat.flatten())).mean()
    # pose_pos_diff = np.linalg.norm(cur_wbpos - tgt_wbpos, axis = 1).mean()

    # p_reward = math.exp(-k_p * (pose_quat_diff ** 2))
    # jp_reward = math.exp(-k_jp * (pose_pos_diff ** 2))

    # ## ARNet Action supervision
    # act_qpos = env.target['qpos']
    # tgt_qpos = env.ar_context['ar_qpos'][ind]

    # act_bquat = env.target['bquat'].flatten()
    # tgt_bquat = env.ar_context['ar_bquat'][ind].flatten()
    # tgt_prev_bquat = env.ar_context['ar_bquat'][ind - 1].flatten()
    # prev_bquat = env.prev_bquat

    # rp_dist = np.linalg.norm(tgt_qpos[:3] - act_qpos[:3])
    # rq_dist = multi_quat_norm_v2(multi_quat_diff(tgt_qpos[3:7], act_qpos[3:7])).mean()
    # pose_action_diff = multi_quat_norm_v2(multi_quat_diff(tgt_bquat, act_bquat)).mean()

    # cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    # tgt_bangvel = get_angvel_fd(tgt_prev_bquat, tgt_bquat, env.dt)
    # vel_dist = np.linalg.norm(cur_bangvel - tgt_bangvel, ord=v_ord)
    # act_v_reward = math.exp(-k_act_v * (vel_dist ** 2))

    # rq_reward = math.exp(-k_rq * (rq_dist ** 2))
    # rp_reward = math.exp(-k_rp * (rp_dist ** 2))
    # act_p_reward = math.exp(-k_act_p * (pose_action_diff))
    # # rq_reward = 0
    # # rp_reward = 0
    # # act_p_reward = 0

    # reward = w_hp * hp_reward + w_hq * hq_reward + w_hv * hv_reward + w_p * p_reward + \
    #     w_jp * jp_reward + w_rp * rp_reward + w_rq * rq_reward  + w_act_v * act_v_reward + w_act_p * act_p_reward
    # print(reward)
    # if flags.debug:
    #     import pdb; pdb.set_trace()
    #     np.set_printoptions(precision=4, suppress=1)
    #     print(np.array([hp_reward, hq_reward, hv_reward, p_reward, jp_reward, rp_reward, rq_reward, act_v_reward, act_p_reward]))

    return reward, np.array([hp_reward, hq_reward, hv_reward, p_reward, jp_reward, rp_reward, rq_reward, act_v_reward, act_p_reward])


def dynamic_supervision_v3(env, state, action, info):
    # V3 is V2 mutiplicative
    # This is wrong, very wrong. This does not work since you should compare the simulated with the estimated!!!!!!
    cfg = env.cfg
    ws = cfg.policy_specs['reward_weights']
    # w_hp, w_hq, w_p, w_jp, w_rp, w_rq, w_act_p, w_act_v = ws.get('w_hp', 1.0), ws.get('w_hq', 1.0),\
    #     ws.get('w_p', 1.0), ws.get('w_jp', 1.0), ws.get('w_rp', 1.0), ws.get('w_rq', 1.0), ws.get('w_act_p', 1.0), ws.get('w_act_v', 1.0)
    k_hp, k_hq,  k_p, k_jp, k_rp, k_rq, k_act_p, k_act_v = ws.get('k_hp', 1.0), ws.get('k_hq', 1.0),   \
        ws.get('k_p', 1.0), ws.get('k_jp', 0.1), ws.get('k_rp', 0.1), ws.get('k_rq', 0.1), ws.get('k_act_p', 0.1), ws.get('k_act_v', 0.1)
    v_ord = ws.get('v_ord', 2)

    ind = env.cur_t
    # Head losses
    tgt_hpos = env.ar_context['head_pose'][ind]
    tgt_hvel = env.ar_context['head_vels'][ind]

    cur_hpos = env.get_head().copy()
    prev_hpos = env.prev_hpos.copy()

    hp_dist = np.linalg.norm(cur_hpos[:3] - tgt_hpos[:3])
    hp_reward = math.exp(-k_hp * (hp_dist**2))

    # head orientation reward
    hq_dist = multi_quat_norm_v2(multi_quat_diff(cur_hpos[3:], tgt_hpos[3:])).mean()
    hq_reward = math.exp(-k_hq * (hq_dist**2))

    cur_bquat = env.get_body_quat()

    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)
    tgt_bquat, tgt_wbpos = env.target['bquat'], env.target['wbpos']

    pose_quat_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat, tgt_bquat.flatten())).mean()
    pose_pos_diff = np.linalg.norm(cur_wbpos - tgt_wbpos, axis=1).mean()

    p_reward = math.exp(-k_p * (pose_quat_diff**2))
    jp_reward = math.exp(-k_jp * (pose_pos_diff**2))

    ## ARNet Action supervision
    act_qpos = env.target['qpos']
    tgt_qpos = env.ar_context['ar_qpos'][ind]

    act_bquat = env.target['bquat'].flatten()
    tgt_bquat = env.ar_context['ar_bquat'][ind].flatten()
    tgt_prev_bquat = env.ar_context['ar_bquat'][ind - 1].flatten()
    prev_bquat = env.prev_bquat

    rp_dist = np.linalg.norm(tgt_qpos[:3] - act_qpos[:3])
    rq_dist = multi_quat_norm_v2(multi_quat_diff(tgt_qpos[3:7], act_qpos[3:7])).mean()
    pose_action_diff = multi_quat_norm_v2(multi_quat_diff(tgt_bquat, act_bquat)).mean()

    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    tgt_bangvel = get_angvel_fd(tgt_prev_bquat, tgt_bquat, env.dt)
    vel_dist = np.linalg.norm(cur_bangvel - tgt_bangvel, ord=v_ord)
    act_v_reward = math.exp(-k_act_v * (vel_dist**2))
    # act_v_reward = 0

    rq_reward = math.exp(-k_rq * (rq_dist**2))
    rp_reward = math.exp(-k_rp * (rp_dist**2))
    act_p_reward = math.exp(-k_act_p * (pose_action_diff))

    # import pdb; pdb.set_trace()

    # reward = hp_reward * hq_reward  *  p_reward * jp_reward * rp_reward  * rq_reward  * act_p_reward * act_v_reward
    reward = hp_reward * hq_reward * p_reward * jp_reward * rp_reward * rq_reward * act_p_reward
    # if flags.debug:
    # np.set_printoptions(precision=4, suppress=1)
    # print(reward, np.array([hp_reward, hq_reward, p_reward, jp_reward, rp_reward, rq_reward, act_p_reward, act_v_reward]))

    return reward, np.array([hp_reward, hq_reward, p_reward, jp_reward, rp_reward, rq_reward, act_p_reward, act_v_reward])


def dynamic_supervision_v4(env, state, action, info):
    # V4 does not have the action terms (does not regularize the action)
    cfg = env.cfg
    ws = cfg.policy_specs['reward_weights']
    w_hp, w_hq, w_hv, w_p, w_jp, w_rp, w_rq, w_act_p = ws.get('w_hp', 1.0), ws.get('w_hq', 1.0),\
         ws.get('w_hv', 0.05), ws.get('w_p', 1.0), ws.get('w_jp', 1.0), ws.get('w_rp', 1.0), ws.get('w_rq', 1.0), ws.get('w_act_p', 1.0)
    k_hp, k_hq, k_hv, k_p, k_jp, k_rp, k_rq, k_act_p = ws.get('k_hp', 1.0), ws.get('k_hq', 1.0), ws.get('k_hv', 1.0 ), \
        ws.get('k_p', 1.0), ws.get('k_jp', 0.1), ws.get('k_rp', 0.1), ws.get('k_rq', 0.1), ws.get('k_act_p', 0.1)

    ind = env.cur_t
    # Head losses
    tgt_hpose = env.ar_context['head_pose'][ind]
    # tgt_hvel = env.ar_context['head_vels'][ind]

    cur_hpose = env.get_head().copy()
    prev_hpos = env.prev_hpos.copy()

    hpvel = (cur_hpose[:3] - prev_hpos[:3]) / env.dt
    # hqvel = get_angvel_fd(prev_hpos[3:], cur_hpose[3:], env.dt)

    hp_dist = np.linalg.norm(cur_hpose[:3] - tgt_hpose[:3])
    hp_reward = math.exp(-k_hp * (hp_dist**2))

    # head orientation reward
    hq_dist = multi_quat_norm_v2(multi_quat_diff(cur_hpose[3:], tgt_hpose[3:])).mean()
    hq_reward = math.exp(-k_hq * (hq_dist**2))

    # head velocity reward
    # hpvel_dist = np.linalg.norm(hpvel - tgt_hvel[:3])
    # hqvel_dist = np.linalg.norm(hqvel - tgt_hvel[3:])
    # hv_reward = math.exp(-hpvel_dist - k_hv * hqvel_dist)
    hv_reward = 0

    cur_bquat = env.get_body_quat()
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)
    tgt_bquat, tgt_wbpos = env.target['bquat'], env.target['wbpos']

    pose_quat_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat.flatten(), tgt_bquat.flatten())).mean()
    pose_pos_diff = np.linalg.norm(cur_wbpos - tgt_wbpos, axis=1).mean()

    p_reward = math.exp(-k_p * (pose_quat_diff**2))
    jp_reward = math.exp(-k_jp * (pose_pos_diff**2))

    reward = w_hp * hp_reward + w_hq * hq_reward + w_hv * hv_reward + w_p * p_reward + w_jp * jp_reward

    # if flags.debug:
    # np.set_printoptions(precision=4, suppress=1)
    # print(np.array([hp_reward, hq_reward, hv_reward, p_reward, jp_reward, rp_reward, rq_reward, act_p_reward]))

    return reward, np.array([hp_reward, hq_reward, hv_reward, p_reward, jp_reward])


def dynamic_supervision_v5(env, state, action, info):
    # V5 is V4 with multiplicative reward
    cfg = env.cfg
    ws = cfg.policy_specs['reward_weights']
    w_hp, w_hq, w_hv, w_p, w_jp, w_rp, w_rq, w_act_p = ws.get('w_hp', 1.0), ws.get('w_hq', 1.0),\
         ws.get('w_hv', 0.05), ws.get('w_p', 1.0), ws.get('w_jp', 1.0), ws.get('w_rp', 1.0), ws.get('w_rq', 1.0), ws.get('w_act_p', 1.0)
    k_hp, k_hq, k_hv, k_p, k_jp, k_rp, k_rq, k_act_p = ws.get('k_hp', 1.0), ws.get('k_hq', 1.0), ws.get('k_hv', 1.0 ), \
        ws.get('k_p', 1.0), ws.get('k_jp', 0.1), ws.get('k_rp', 0.1), ws.get('k_rq', 0.1), ws.get('k_act_p', 0.1)

    ind = env.cur_t
    # Head losses
    tgt_hpose = env.ar_context['head_pose'][ind]
    # tgt_hvel = env.ar_context['head_vels'][ind]

    cur_hpose = env.get_head().copy()
    prev_hpos = env.prev_hpos.copy()

    hpvel = (cur_hpose[:3] - prev_hpos[:3]) / env.dt
    # hqvel = get_angvel_fd(prev_hpos[3:], cur_hpose[3:], env.dt)

    hp_dist = np.linalg.norm(cur_hpose[:3] - tgt_hpose[:3])
    hp_reward = math.exp(-k_hp * (hp_dist**2))

    # head orientation reward
    hq_dist = multi_quat_norm_v2(multi_quat_diff(cur_hpose[3:], tgt_hpose[3:])).mean()
    hq_reward = math.exp(-k_hq * (hq_dist**2))

    # head velocity reward
    # hpvel_dist = np.linalg.norm(hpvel - tgt_hvel[:3])
    # hqvel_dist = np.linalg.norm(hqvel - tgt_hvel[3:])
    # hv_reward = math.exp(-hpvel_dist - k_hv * hqvel_dist)
    hv_reward = 0

    cur_bquat = env.get_body_quat()
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)
    tgt_bquat, tgt_wbpos = env.target['bquat'], env.target['wbpos']

    pose_quat_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat.flatten(), tgt_bquat.flatten())).mean()
    pose_pos_diff = np.linalg.norm(cur_wbpos - tgt_wbpos, axis=1).mean()

    p_reward = math.exp(-k_p * (pose_quat_diff**2))
    jp_reward = math.exp(-k_jp * (pose_pos_diff**2))

    reward = hp_reward * hq_reward * p_reward * jp_reward

    # if flags.debug:
    # np.set_printoptions(precision=4, suppress=1)
    # print(np.array([hp_reward, hq_reward, hv_reward, p_reward, jp_reward, rp_reward, rq_reward, act_p_reward]))

    return reward, np.array([hp_reward, hq_reward, hv_reward, p_reward, jp_reward])


def dynamic_supervision_v6(env, state, action, info):
    # no head reward anymore
    cfg = env.cfg
    ws = cfg.policy_specs['reward_weights']
    w_hp, w_hq, w_hv, w_p, w_jp, w_rp, w_rq, w_act_p, w_act_v = ws.get('w_hp', 1.0), ws.get('w_hq', 1.0),\
         ws.get('w_hv', 0.05), ws.get('w_p', 1.0), ws.get('w_jp', 1.0), ws.get('w_rp', 1.0), ws.get('w_rq', 1.0), ws.get('w_act_p', 1.0), ws.get('w_act_v', 1.0)
    k_hp, k_hq, k_hv, k_p, k_jp, k_rp, k_rq, k_act_p, k_act_v = ws.get('k_hp', 1.0), ws.get('k_hq', 1.0), ws.get('k_hv', 1.0 ), \
        ws.get('k_p', 1.0), ws.get('k_jp', 0.1), ws.get('k_rp', 0.1), ws.get('k_rq', 0.1), ws.get('k_act_p', 0.1), ws.get('k_act_v', 0.1)
    v_ord = ws.get('v_ord', 2)

    ind = env.cur_t

    # Head losses
    tgt_hpose = env.ar_context['head_pose'][ind]

    cur_hpose = env.get_head().copy()
    prev_hpos = env.prev_hpos.copy()

    hp_dist = np.linalg.norm(cur_hpose[:3] - tgt_hpose[:3])
    hp_reward = math.exp(-k_hp * (hp_dist**2))

    # head orientation reward
    hq_dist = multi_quat_norm_v2(multi_quat_diff(cur_hpose[3:], tgt_hpose[3:])).mean()
    hq_reward = math.exp(-k_hq * (hq_dist**2))

    cur_bquat = env.get_body_quat()
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)
    tgt_bquat, tgt_wbpos = env.target['bquat'], env.target['wbpos']

    pose_quat_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat.flatten(), tgt_bquat.flatten())).mean()
    pose_pos_diff = np.linalg.norm(cur_wbpos - tgt_wbpos, axis=1).mean()

    p_reward = math.exp(-k_p * (pose_quat_diff**2))
    jp_reward = math.exp(-k_jp * (pose_pos_diff**2))

    tgt_bquat = env.ar_context['ar_bquat'][ind].flatten()
    tgt_prev_bquat = env.ar_context['ar_bquat'][ind - 1].flatten()
    prev_bquat = env.prev_bquat

    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    tgt_bangvel = get_angvel_fd(tgt_prev_bquat, tgt_bquat, env.dt)
    vel_dist = np.linalg.norm(cur_bangvel - tgt_bangvel, ord=v_ord)
    act_v_reward = math.exp(-k_act_v * (vel_dist**2))

    reward = w_hp * hp_reward + w_hq * hq_reward + w_p * p_reward + w_jp * jp_reward + w_act_v * act_v_reward

    # if flags.debug:
    #     import pdb
    #     pdb.set_trace()
    #     np.set_printoptions(precision=4, suppress=1)
    #     print(reward, np.array([p_reward, jp_reward, act_v_reward]))

    return reward, np.array([hp_reward, hq_reward, p_reward, jp_reward, act_v_reward])


def reprojection_reward(env, state, action, info):
    # reward coefficients
    cfg = env.kin_cfg
    ws = cfg.policy_specs.get("reward_weights")
    w_p, w_v, w_e, w_c, w_vf, w_kp = (
        ws.get("w_p", 0.6),
        ws.get("w_v", 0.1),
        ws.get("w_e", 0.2),
        ws.get("w_c", 0.1),
        ws.get("w_vf", 0.0),
        ws.get("w_kp", 0.1),
    )
    k_p, k_v, k_e, k_c, k_vf, k_kp = (
        ws.get("k_p", 2),
        ws.get("k_v", 0.005),
        ws.get("k_e", 20),
        ws.get("k_c", 1000),
        ws.get("k_vf", 1),
        ws.get("k_kp", 0.001),
    )
    v_ord = ws.get("v_ord", 2)
    # data from env
    ind = env.cur_t
    prev_bquat = env.prev_bquat
    bm, smpl2op_map, aR, atr, R, tr, K = env.bm, env.smpl2op_map, env.camera_params['aR'], env.camera_params['atr'], env.camera_params['R'], env.camera_params['tr'], env.camera_params['K']
    betas = torch.from_numpy(env.context_dict['beta'][0:1])

    # 2D reprojection reward
    qpos = env.data.qpos.copy()[None]
    pose_aa, trans = qpos_to_smpl(qpos, env.smpl_model, env.cc_cfg.robot_cfg.get("model", "smpl"))
    pose_aa = pose_aa[:, :22].reshape((-1, 22 * 3))
    root_orient = pose_aa[:, :3]
    pose_body = pose_aa[:, 3:]

    pose_body = torch.from_numpy(pose_body).float()
    root_orient = torch.from_numpy(root_orient).float()
    trans = torch.from_numpy(trans).float()

    pred_body = bm(pose_body=pose_body, pose_hand=None, betas=betas, root_orient=root_orient, trans=trans)

    pred_joints3d = pred_body.Jtr.numpy()
    pred_joints3d = pred_joints3d.reshape(1, -1, 3)
    pred_joints3d = pred_joints3d[:, smpl2op_map, :]
    pred_joints3d = pred_joints3d @ aR.T + atr
    pred_joints3d = pred_joints3d @ R.T + tr
    pred_joints2d = pred_joints3d @ (K.T)[None]
    z = pred_joints2d[:, :, 2:]
    pred_joints2d = pred_joints2d[:, :, :2] / z

    joints2d = env.context_dict["joints2d"][ind]
    inliers = joints2d[:, 2] > 0.5

    pred_joints2d = pred_joints2d[0, inliers]
    joints2d = joints2d[inliers, :2]

    dist = np.linalg.norm(pred_joints2d - joints2d, axis=1).mean()
    kp_reward = math.exp(-k_kp * (dist**2))
    reward = kp_reward
    # print(reward, env.cur_t)
    return reward, np.array([kp_reward])


def reprojection_reward3d(env, state, action, info):
    # reward coefficients
    cfg = env.kin_cfg
    ws = cfg.reward_weights
    w_p, w_e, w_c, w_mp, w_kp, w_pc, w_rp = (
        ws.get("w_p", 0.6),
        ws.get("w_e", 0.2),
        ws.get("w_c", 0.1),
        ws.get("w_mp", 0.1),
        ws.get("w_kp", 0.1),
        ws.get("w_pc", 0.1),
        ws.get("w_rp", 0.1),
    )
    k_p, k_e, k_c, k_mp, k_kp, k_rp, k_pc, = (
        ws.get("k_p", 2),
        ws.get("k_e", 20),
        ws.get("k_c", 1000),
        ws.get("k_mp", 0.01),
        ws.get("k_kp", 0.01),
        ws.get("k_rp", 50),
        ws.get("k_pc", 0.01),
    )

    bm, smpl2op_map, aR, atr, R, tr, K = env.bm, env.smpl2op_map, env.camera_params['aR'], env.camera_params['atr'], env.camera_params['R'], env.camera_params['tr'], env.camera_params['K']
    curr_humor_state = env.cur_humor_state
    betas = torch.from_numpy(env.context_dict['beta'][0:1])

    # past_in, next_out = env.motion_prior.canonicalize_input_double(env.prev_humor_state, env.cur_humor_state, split_input=False, cam_params = env.camera_params_torch)
    # pm, pv = env.motion_prior.prior(past_in.reshape(1, -1))
    # qm, qv = env.motion_prior.posterior(past_in.reshape(1, -1), next_out.reshape(1, -1))
    # prior_loss = kl_normal(qm, qv, pm, pv).mean().numpy()

    # mp_reward = math.exp(-k_mp * prior_loss)

    v_ord = ws.get("v_ord", 2)
    # data from env
    ind = t = env.cur_t

    prev_bquat = env.prev_bquat

    # 2D reprojection reward
    qpos = env.data.qpos.copy()[None]
    pose_aa, trans = qpos_to_smpl(qpos, env.smpl_model, env.cc_cfg.robot_cfg.get("model", "smpl"))
    pose_aa = pose_aa[:, :22].reshape((-1, 22 * 3))
    root_orient = pose_aa[:, :3]
    pose_body = pose_aa[:, 3:]

    # pred_vertices = curr_humor_state['pred_vertices'].squeeze()
    points3d = env.context_dict["points3d"][ind]
    # dist = points3d_loss(points3d, pred_vertices)
    # pc_reward = math.exp(-k_pc * (dist**2))

    pred_joints2d = curr_humor_state['pred_joints2d'].squeeze()
    joints2d = env.context_dict["joints2d"][ind]

    inliers = joints2d[:, 2] > 0.5
    if np.sum(inliers) > 0:
        pred_joints2d = pred_joints2d[inliers]
        joints2d = joints2d[inliers, :2]
        dist = np.linalg.norm(pred_joints2d - joints2d, axis=1).mean()
    else:
        dist = 0.0
    kp_reward = math.exp(-k_kp * (dist**2))

    # Dynamics Regulated Reward
    cur_ee = env.get_ee_pos(None)
    cur_bquat = env.get_body_quat()

    # expert
    e_ee = env.target["ee_wpos"]
    cur_com = env.get_com()
    e_com = env.target["com"]
    e_bquat = env.target["bquat"].squeeze()

    # pose reward
    pose_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat, e_bquat))
    pose_diff[1:] *= env.body_diffw
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist**2))

    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist**2))
    # com reward
    com_dist = np.linalg.norm(cur_com - e_com)
    com_reward = math.exp(-k_c * (com_dist**2))

    # overall reward
    reward = (w_p * pose_reward + w_e * ee_reward + w_c * com_reward + w_kp * kp_reward)
    reward /= w_p + w_e + w_c + w_kp

    # if flags.debug:
    #     np.set_printoptions(precision=4, suppress=1)
    #     print(
    #         np.array(
    #             [pose_reward, ee_reward, com_reward, kp_reward]))

    return reward, np.array([pose_reward, ee_reward, com_reward, kp_reward])


def reprojection_reward3d_gt(env, state, action, info):
    # reward coefficients
    cfg = env.kin_cfg
    ws = cfg.reward_weights
    w_p, w_e, w_c, w_p_gt, w_e_gt, w_c_gt, w_kp = (
        ws.get("w_p", 0.6),
        ws.get("w_e", 0.2),
        ws.get("w_c", 0.1),
        ws.get("w_p_gt", 0.6),
        ws.get("w_e_gt", 0.2),
        ws.get("w_c_gt", 0.1),
        ws.get("w_kp", 0.1),
    )
    k_p, k_e, k_c, k_kp = (
        ws.get("k_p", 2),
        ws.get("k_e", 20),
        ws.get("k_c", 1000),
        ws.get("k_kp", 0.01),
    )

    curr_humor_state = env.cur_humor_state

    v_ord = ws.get("v_ord", 2)
    # data from env
    ind = t = env.cur_t

    # 2D reprojection reward
    qpos = env.data.qpos.copy()[None]
    pose_aa, trans = qpos_to_smpl(qpos, env.smpl_model, env.cc_cfg.robot_cfg.get("model", "smpl"))
    pose_aa = pose_aa[:, :22].reshape((-1, 22 * 3))

    pred_joints2d = curr_humor_state['pred_joints2d'].squeeze()
    joints2d = env.context_dict["joints2d"][ind]

    inliers = joints2d[:, 2] > 0.5
    if np.sum(inliers) > 0:
        pred_joints2d = pred_joints2d[inliers]
        joints2d = joints2d[inliers, :2]
        dist = np.linalg.norm(pred_joints2d - joints2d, axis=1).mean()
    else:
        dist = 0.0
    kp_reward = math.exp(-k_kp * (dist**2))

    # Dynamics Regulated Reward
    cur_ee = env.get_ee_pos(None)
    cur_bquat = env.get_body_quat()

    ######## ARNet Matching
    e_ee = env.target["ee_wpos"]
    cur_com = env.get_com()
    e_com = env.target["com"]
    e_bquat = env.target["bquat"].squeeze()

    # pose reward
    pose_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat, e_bquat))
    pose_diff[1:] *= env.body_diffw
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist**2))

    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist**2))
    # com reward
    com_dist = np.linalg.norm(cur_com - e_com)
    com_reward = math.exp(-k_c * (com_dist**2))

    ######## ARNet Matching
    gt_ee = env.gt_targets["ee_wpos"][ind]
    gt_com = env.gt_targets["com"][ind]
    gt_bquat = env.gt_targets["bquat"][ind].squeeze()

    # pose reward
    gt_pose_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat, gt_bquat))
    gt_pose_diff[1:] *= env.body_diffw
    gt_pose_dist = np.linalg.norm(gt_pose_diff)
    gt_pose_reward = math.exp(-k_p * (gt_pose_dist**2))

    # ee reward
    gt_ee_dist = np.linalg.norm(cur_ee - gt_ee)
    gt_ee_reward = math.exp(-k_e * (gt_ee_dist**2))
    # com reward
    gt_com_dist = np.linalg.norm(cur_com - gt_com)
    gt_com_reward = math.exp(-k_c * (gt_com_dist**2))

    # overall reward
    reward = (w_p * pose_reward + w_e * ee_reward + w_c * com_reward + w_p_gt * gt_pose_reward + w_e_gt * gt_ee_reward + w_c_gt * gt_com_reward + w_kp * kp_reward)
    reward /= w_p + w_e + w_c + w_p_gt + w_e_gt + w_c_gt + w_kp
    assert (w_p + w_e + w_c + w_p_gt + w_e_gt + w_c_gt + w_kp == 1)

    return reward, np.array([pose_reward, ee_reward, com_reward, gt_pose_reward, gt_ee_reward, gt_com_reward, kp_reward])


reward_func = {
    "dynamic_supervision_v1": dynamic_supervision_v1,
    "dynamic_supervision_v2": dynamic_supervision_v2,
    "dynamic_supervision_v3": dynamic_supervision_v3,
    "dynamic_supervision_v4": dynamic_supervision_v4,
    "dynamic_supervision_v5": dynamic_supervision_v5,
    "dynamic_supervision_v6": dynamic_supervision_v6,
    "reprojection_reward": reprojection_reward,
    "reprojection_reward3d": reprojection_reward3d,
    "reprojection_reward3d_gt": reprojection_reward3d_gt,
}
