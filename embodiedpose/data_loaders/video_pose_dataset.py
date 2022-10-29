import numpy as np
import torch.utils.data as data
import glob
import pickle as pk
import joblib
from collections import defaultdict

from copycat.data_loaders.dataset_amass_batch import DatasetAMASSBatch
import torch
from copycat.utils.math_utils import (
    de_heading,
    transform_vec,
    quaternion_multiply,
    quaternion_inverse,
    rotation_from_quaternion,
    ewma,
)


class VideoPoseDataset(DatasetAMASSBatch):
    def process_data_list(self, data_list):
        data_processed = defaultdict(dict)
        # pbar = tqdm(all_data)
        for take, curr_data in data_list:
            gt_qpos = curr_data["qpos"]
            seq_len = gt_qpos.shape[0]

            if seq_len < self.fr_num:
                continue

            # data that needs pre-processing
            gt_qpos[:, 3:7] /= np.linalg.norm(gt_qpos[:, 3:7], axis=1)[:, None]
            traj_pos = self.get_traj_de_heading(gt_qpos)

            traj_root_vel = self.get_root_vel(gt_qpos)
            traj = np.hstack(
                (traj_pos,
                 traj_root_vel))  # Trajectory and trajectory root velocity
            data_processed["wbpos"][take] = curr_data["wbpos"]
            data_processed["wbquat"][take] = curr_data["wbquat"]
            data_processed["bquat"][take] = curr_data["bquat"]
            data_processed["qvel"][take] = curr_data["qvel"]
            data_processed["target"][take] = traj
            data_processed["qpos"][take] = gt_qpos
            bbox_vels = self.get_2d_vel(curr_data["bbox"])
            data_processed["pose_aa"][take] = curr_data["pose"]
            data_processed["img_feats"][take] = curr_data["features"]
            data_processed["bbox"][take] = np.array(curr_data["bbox"])
            data_processed["bbox_vels"][take] = np.array(bbox_vels)
            data_processed["img_feats"][take] = np.array(curr_data["features"])
            # data_processed["obj_pose"][take] = np.array(curr_data["obj_pose"])

        return data_processed

    # def __getitem__(self, index):
    #     # sample random sequence from data
    #     take_key = self.sample_keys[index]
    #     sample = self.get_sample_from_key(take_key, fr_start=0)
    #     return sample

    def check_has_obj(self, data_key):
        return self.data_raw[data_key]["has_obj"]

    def get_traj_de_heading(self, orig_traj):
        # Remove trajectory-heading + remove horizontal movements
        # results: 57 (-2 for horizontal movements)
        # Contains deheaded root orientation
        traj_pos = orig_traj[:, 3:].copy()  # qpos without x, y, z

        # traj_pos[:, 4:] = np.concatenate(
        #     (traj_pos[1:, 4:], traj_pos[-2:-1, 4:])
        # )  # body pose 1 step forward for autoregressive target

        # for i in range(traj_pos.shape[0]):
        # traj_pos[i, :4] = de_heading(traj_pos[i, :4])

        return traj_pos

    def get_root_vel(self, orig_traj):
        # Get root velocity: 1x6
        traj_root_vel = []
        for i in range(orig_traj.shape[0] - 1):
            # vel = get_qvel_fd(orig_traj[i, :], orig_traj[i + 1, :], self.dt, 'heading')
            curr_qpos = orig_traj[i, :].copy()
            next_qpos = orig_traj[i + 1, :].copy()
            if self.cfg.model_specs.get("remove_base", False):
                curr_qpos[3:7] = self.remove_base_rot(curr_qpos[3:7])
                next_qpos[3:7] = self.remove_base_rot(next_qpos[3:7])

            v = (next_qpos[:3] - curr_qpos[:3]) / self.dt
            v = transform_vec(v, curr_qpos[3:7], "root").copy()
            qrel = quaternion_multiply(next_qpos[3:7],
                                       quaternion_inverse(curr_qpos[3:7]))
            axis, angle = rotation_from_quaternion(qrel, True)

            if angle > np.pi:  # -180 < angle < 180
                angle -= 2 * np.pi  #
            elif angle < -np.pi:
                angle += 2 * np.pi

            rv = (axis * angle) / self.dt
            rv = transform_vec(rv, curr_qpos[3:7], "root")

            traj_root_vel.append(np.concatenate((v, rv)))

        traj_root_vel.append(
            traj_root_vel[-1].copy()
        )  # copy last one since there will be one less through finite difference
        traj_root_vel = np.vstack(traj_root_vel)
        return traj_root_vel

    def get_2d_vel(self, bbox):
        delta_bbox = bbox[1:] - bbox[:-1]
        return np.concatenate([delta_bbox, delta_bbox[-2:-1]])
