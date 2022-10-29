import argparse
from email.policy import default
import os.path as osp
import joblib
import json
import sys
import os
import pdb

import torch
import numpy as np

sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
from copycat.smpllib.smpl_parser import SMPL_Parser, SMPLX_Parser
from embodiedpose.models.humor.torch_humor_loss import chamfer_loss
from embodiedpose.utils.scene_utils import get_sdf, load_simple_scene
from tqdm import tqdm
from collections import defaultdict

PEN_THRESH = 0.1
TRIM_EDGE = 90


def load_camera_params(prox_path, scene_name):

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

    return {
        "K": K,
        "R": R,
        "tr": tr,
        "aR": aR,
        "atr": atr,
        "full_R": full_R,
        "full_t": full_t
    }


def compute_accel(joints):
    """
    Computes acceleration of 3D joints.
    Args:
        joints (Nx25x3).
    Returns:
        Accelerations (N-2).
    """
    velocities = joints[1:] - joints[:-1]
    acceleration = velocities[1:] - velocities[:-1]
    acceleration_normed = torch.norm(acceleration, dim=2)
    return torch.mean(acceleration_normed, dim=1)


def evaluate():
    # prox_path = '/data/method/prox_method'
    prox_path = "/hdd/zen/data/video_pose/prox/qualitative"
    res_path = f'results/third'

    if args.method == 'lemo':
        # LEMO
        filepath = osp.join(res_path, 'lemo_processed.pkl')
        results = joblib.load(filepath)
    elif args.method == 'humor':
        # HuMoR
        filepath = osp.join(res_path, 'thirdeye_anns_prox_overlap.pkl')
        results = joblib.load(filepath)
    elif args.method == 'humord':
        filepath = osp.join(res_path, 'thirdeye_anns_proxd_overlap.pkl')
        results = joblib.load(filepath)
    elif args.method == 'thirdeye':
        # Thirdeye
        res_path = f'results/scene+/{args.cfg}/results/'
        os.system(f"python sceneplus/data_process/process_thirdeye.py --cfg {args.cfg} --epoch {args.epoch} --data {args.data}")
        filepath = osp.join(res_path, 'thirdeye_final_processed.pkl')
        results = joblib.load(filepath)
    elif args.method == 'prox':
        # PROX
        filepath = osp.join(res_path, 'prox_processed.pkl')
        results = joblib.load(filepath)
    elif args.method == 'proxd':
        # PROXD
        filepath = osp.join(res_path, 'proxd_processed.pkl')
        results = joblib.load(filepath)
    elif args.method == 'hybrik':
        # Hybrik
        filepath = osp.join(res_path, 'hybrik_processed.pkl')
        results = joblib.load(filepath)

    print(f'Loaded {filepath}')

    if args.chamfer:
        filepath = osp.join(prox_path, 'thirdeye_anns_proxd_overlap_full.pkl')
        points3d_results = joblib.load(filepath)
    

    if args.method == 'lemo' or args.method == 'proxd' or args.method == 'prox':
        smpl_parser_n = SMPLX_Parser(model_path='./data/smpl/smplx', gender="neutral", use_pca=False, create_transl=False).cuda()
        smpl_parser_m = SMPLX_Parser(model_path='./data/smpl/smplx', gender="male", use_pca=False, create_transl=False).cuda()
        smpl_parser_f = SMPLX_Parser(model_path='./data/smpl/smplx', gender="female", use_pca=False, create_transl=False).cuda()
    else:
        data_dir = "data/smpl"
        
        smpl_parser_n = SMPL_Parser(model_path=data_dir,gender="neutral").cuda()
        smpl_parser_m = SMPL_Parser(model_path=data_dir,gender="male").cuda()
        smpl_parser_f = SMPL_Parser(model_path=data_dir,gender="female").cuda()


    frame_num = 0
    gp_sum = 0
    gp_num = 0
    gp_freq = 0
    sp_sum = 0
    sp_num = 0
    sp_freq = 0
    accel_sum = 0
    chamfer_sum = 0
    res_dict = defaultdict(dict)

    with torch.no_grad():
        for seq_name in tqdm(results.keys()):

            if seq_name == 'success_rate':
                continue
            # elif seq_name == "N0Sofa_00034_01":
            #     continue
            # elif seq_name == "MPH1Library_00145_01":
            #     continue
            # elif seq_name == "N3Office_03301_01":
            #     continue

            cam_param = load_camera_params(prox_path, seq_name[:-9])
            full_R = torch.from_numpy(cam_param['full_R']).cuda()
            full_t = torch.from_numpy(cam_param['full_t']).cuda()
            sdfs, _ = load_simple_scene(seq_name[:-9])

            seq_results = results[seq_name]
            gender = seq_results['gender']
            betas = seq_results['betas'][None]
            pose_aa = seq_results['pose_aa']
            trans = seq_results['trans']
            if 'fail' in seq_results and seq_results['fail']:
                print(seq_name, "failed")
                continue
            
            if args.method == 'prox' or args.method == 'proxd':
                ind1 = np.sum(np.isnan(trans), axis=1)
                ind2 = np.sum(np.isnan(pose_aa), axis=1)
                ind = ind1 + ind2
                pose_aa = pose_aa[ind == 0]
                trans = trans[ind == 0]
            else:
                ind = None


            pose_aa = torch.from_numpy(pose_aa).cuda()
            betas = torch.from_numpy(betas).cuda()
            trans = torch.from_numpy(trans).cuda()

            if args.method == 'lemo' or args.method == 'proxd' or args.method == 'prox':
                pad = torch.zeros((pose_aa.shape[0], 84)).cuda()
                pose_aa = torch.cat([pose_aa, pad], dim=1)
                pad = torch.zeros((1, 10)).cuda()
                betas = torch.cat([betas, pad], dim=1)
                if gender == "male": smpl_parser = smpl_parser_m
                elif gender == "female": smpl_parser = smpl_parser_f
                elif gender == "neutral": smpl_parser = smpl_parser_n
                else: raise ValueError
                
            else:
                smpl_parser = smpl_parser_n
          
            try:
                vertices, joints = smpl_parser.get_joints_verts(
                    pose_aa, betas, trans)
            except Exception as e:
                # import ipdb; ipdb.set_trace()
                print(e)
                continue

            if args.method == 'lemo' or args.method == 'hybrik' or \
                    args.method == 'prox' or args.method == 'proxd':
                vertices = (vertices - full_t) @ full_R
                joints = (joints - full_t) @ full_R

            frame_num += trans.shape[0]

            # GP penetration
            z = vertices[:, :, 2].reshape(-1)
            z = z[z < 0.0]
            gp_sum += -z.sum().item()
            gp_num += z.shape[0]

            # # GP frequency
            z_pf = torch.clamp(vertices[:, :, 2], max=0.0)
            gp_mean_pf = -z_pf.sum(dim=1) / ((z_pf < 0.0).sum(dim=1) + 1e-10)
            gp_freq += (gp_mean_pf > PEN_THRESH).sum().item()

            # Acceleration
            accel_sum += compute_accel(joints).sum().item()

            # Average Chamfer Distance
            if args.chamfer:
                # seq_results_d = points3d_results[seq_name]
                # points3d = torch.from_numpy(seq_results_d['points3d']).float().cuda()
                
                if args.method == "thirdeye":
                    vertices = vertices[TRIM_EDGE:]

                num_seq = vertices.shape[0]
                points3d = joblib.load(osp.join(prox_path, f'point_cloud/{seq_name}.pkl')).cuda()[TRIM_EDGE:-TRIM_EDGE]

                if args.method == 'prox' or args.method == 'proxd':
                    num_seq = ind.shape[0]
                    points3d = points3d[:num_seq]
                    points3d = points3d[ind == 0].reshape(-1, 3)
                else:
                    points3d = points3d[:num_seq].reshape(-1, 3)
                
                points3d = (points3d - full_t) @ full_R
                points3d = points3d.reshape(vertices.shape[0], -1, 3)
                is_valid = points3d.isnan().sum(dim=1).sum(dim=1) == 0
                points3d_valid = points3d[is_valid]
                vertices_valid = vertices[is_valid]
                # dist, _ = chamfer_loss(points3d_valid, vertices_valid)
                dist, _ = chamfer_loss(points3d_valid, vertices_valid)

                chamfer_dist = dist.mean(dim=1)
                # if (chamfer_dist > 20).sum():
                    # print("chamfer fail!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", seq_name)
                    # continue

                chamfer_sum += chamfer_dist.sum().item()
                res_dict['chamfer_sum'][seq_name] = chamfer_dist.sum().item()
                # if seq_name == "N0Sofa_00034_01":
                #     print(seq_name)
                #     import ipdb; ipdb.set_trace()
                #     plt.plot(dist.mean(dim=1).squeeze().cpu().numpy()); plt.show()

            # Compute the scene penetration
            for i in range(len(sdfs)):
                sdfs[i] = sdfs[i].cuda()
            vertices = vertices.reshape(-1, 3)
            N = vertices.shape[0] // 5000 + 1
            for i in range(N):
                verts = vertices[i * 5000:(i + 1) * 5000]
                if i == 0:
                    dist = get_sdf(sdfs, verts)[:, 0]
                else:
                    dist = torch.cat([dist, get_sdf(sdfs, verts)[:, 0]], dim=0)

            assert dist.shape[0] == vertices.shape[0]

            dist_n = dist[dist < 0.0]
            sp_sum += -dist_n.sum().item()
            sp_num += dist_n.shape[0]

            if args.method == 'lemo' or args.method == 'proxd' \
                    or args.method == 'prox':
                dist_pf = torch.clamp(dist.reshape(-1, 10475), max=0.0)
            else:
                dist_pf = torch.clamp(dist.reshape(-1, 6890), max=0.0)
            sp_mean_pf = -dist_pf.sum(dim=1) / (
                (dist_pf < 0.0).sum(dim=1) + 1e-10)
            sp_freq += (sp_mean_pf > PEN_THRESH).sum().item()

            torch.cuda.empty_cache()

        gp_freq = gp_freq / frame_num
        sp_freq = sp_freq / frame_num
        print('Avg Chamfer Distance', chamfer_sum * 1000 / frame_num)
        print('Avg Accel.', accel_sum * 1000 / frame_num)
        print('Avg GP Freq', gp_freq * 100, '%')
        print('Avg GP Dist', (gp_sum / gp_num) * 1000.0)
        print('Avg SP Freq', sp_freq * 100, '%')
        print('Avg SP Dist', (sp_sum / sp_num) * 1000.0)

        print("# Evaluated on:", len(res_dict['chamfer_sum']))
        print(args.method)
        import ipdb; ipdb.set_trace()
        np.sort(list(res_dict['chamfer_sum'].values()))
        np.argsort(list(res_dict['chamfer_sum'].values()))
        np.array(list(res_dict['chamfer_sum'].keys()))[12]
        "N0SittingBooth_00169_02"
        "MPH1Library_00145_01"
        "BasementSittingBooth_03452_01"



        if args.method == 'thirdeye':
            print('Success Rate', results['success_rate'])
        print(filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="thirdeye", type=str)
    parser.add_argument("--data", default="prox", type=str)
    parser.add_argument("--cfg", default=None, type=str)
    parser.add_argument("--chamfer", default = True, action='store_true')
    parser.add_argument("--epoch", default=-1, type=int)
    args = parser.parse_args()
    evaluate()
