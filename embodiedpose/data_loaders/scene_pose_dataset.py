import numpy as np
import torch.utils.data as data
import glob
import pickle as pk
import joblib
from collections import defaultdict

import torch
import random
import math
from uhc.data_loaders.dataset_batch import DatasetBatch
from embodiedpose.models.humor.utils.humor_mujoco import SMPL_2_OP, OP_14_to_OP_12
from scipy.ndimage import gaussian_filter1d


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def fill_zeros_with_last(arr, initial=0):
    ind = np.nonzero(arr[:, 2])[0]
    cnt = np.cumsum(np.array(arr[:, 2], dtype=bool))

    stack = np.stack([np.where(cnt, arr[ind[cnt - 1], 0], initial), np.where(cnt, arr[ind[cnt - 1], 1], initial), arr[:, 2]], axis=1)
    return stack


def filter_2d_kps(j2d, sigma=1):
    new_j2d = []
    for i in range(j2d.shape[1]):
        new_val = fill_zeros_with_last(j2d[:, i, :])
        new_j2d.append(new_val)
    new_j2d = np.stack(new_j2d, axis=1)
    fill_beginning = np.where(new_j2d[0, :, 2] == 0)[0]
    if len(fill_beginning) > 0:
        for curr_jt in fill_beginning:
            leading_idx = np.min(np.nonzero(new_j2d[:, curr_jt, 2]))
            new_j2d[:leading_idx, curr_jt, :2] = new_j2d[leading_idx, curr_jt, :2]

    new_j2d[:, :, :2] = gaussian_filter1d(new_j2d[:, :, :2], sigma, axis=0)
    return new_j2d


class ScenePoseDataset(DatasetBatch):

    def __init__(self, *args, **kwargs):
        super(ScenePoseDataset, self).__init__(*args, **kwargs)

    def post_process_data(self, processed_data, raw_data):

        remove_keys = []
        for k, v in raw_data.items():
            if len(v['pose_aa']) < self.t_min:
                remove_keys.append(k)

        for k in remove_keys:
            del raw_data[k]
            for key in processed_data.keys():
                del processed_data[key][k]
        if len(remove_keys) > 0:
            print(f"Removed {remove_keys} samples due to {self.t_min}")

        return processed_data, raw_data

    def get_sample_from_key(self, take_key, full_sample=False, freq_dict=None, fr_start=-1, fr_num=-1, precision_mode=False, sampling_freq=0.75, full_fr_num=False, return_batch=False, exclude_keys=[]):
        sample = super().get_sample_from_key(
            take_key,
            full_sample=full_sample,
            freq_dict=freq_dict,
            fr_start=fr_start,
            fr_num=fr_num,
            precision_mode=precision_mode,
            sampling_freq=sampling_freq,
            return_batch=return_batch,
            full_fr_num=full_fr_num,  # Full fr_num!!
            exclude_keys=['obj_info'])

        if take_key in self.data["obj_info"]:
            sample['obj_info'] = self.data["obj_info"][take_key]

        sample['cam'] = self.data_raw[take_key]['cam']

        return sample

    def process_data_list(self, data_list):
        data_processed = defaultdict(dict)
        # pbar = tqdm(all_data)
        for take, curr_data in data_list:
            pose_aa = curr_data["pose_aa"]
            seq_len = pose_aa.shape[0]

            # data_processed["joints2d"][take] = curr_data["joints2d"][:, SMPL_2_OP][..., OP_14_to_OP_12, :]
            data_processed["joints2d"][take] = filter_2d_kps(curr_data["joints2d"][:, SMPL_2_OP][..., OP_14_to_OP_12, :])

            data_processed["pose_6d"][take] = curr_data["pose_6d"]
            data_processed["pose_aa"][take] = curr_data["pose_aa"]

            data_processed["trans"][take] = curr_data["trans"]
            data_processed["root_orient"][take] = curr_data["root_orient"]
            data_processed["joints"][take] = curr_data["joints"]

            data_processed["trans_vel"][take] = curr_data["trans_vel"]
            data_processed["root_orient_vel"][take] = curr_data["root_orient_vel"]
            data_processed["joints_vel"][take] = curr_data["joints_vel"]

            data_processed["pose_body"][take] = curr_data["pose_body"]
            # data_processed["points3d"][take] = curr_data["points3d"]
            data_processed["points3d"][take] = np.zeros([seq_len, 4096, 3])
            if "joints_gt" in curr_data:
                data_processed["joints_gt"][take] = curr_data["joints_gt"]

            data_processed["phase"][take] = positionalencoding1d(32, seq_len).numpy()
            data_processed["time"][take] = np.arange(seq_len)[:, None]

            if "qpos" in curr_data:
                data_processed["qpos"][take] = curr_data["qpos"]

            # data_processed["qpos"][take] = curr_data["qpos"]
            data_processed["beta"][take] = (np.repeat(
                curr_data["betas"][None,],
                seq_len,
                axis=0,
            ) if curr_data["betas"].shape[0] != seq_len else curr_data["betas"])

            gender = (curr_data["gender"].item() if isinstance(curr_data["gender"], np.ndarray) else curr_data["gender"])

            if isinstance(gender, bytes):
                gender = gender.decode("utf-8")
            if gender == "neutral":
                gender = [0]
            elif gender == "male":
                gender = [1]
            elif gender == "female":
                gender = [2]
            else:
                import ipdb
                ipdb.set_trace()
                raise Exception("Gender Not Supported!!")

            gender = [0]  # ZL: Using neutral body for everything
            data_processed["gender"][take] = np.repeat(
                gender,
                seq_len,
                axis=0,
            )
            if "obj_info" in curr_data:
                data_processed["obj_info"][take] = curr_data['obj_info']

        return data_processed
