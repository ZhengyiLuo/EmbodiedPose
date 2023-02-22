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
from uhc.smpllib.smpl_mujoco import smplh_to_smpl
import uhc.utils.pytorch3d_transforms as tR


class AMASSDataset(DatasetBatch):

    def process_data_list(self, data_list):
        data_processed = defaultdict(dict)
        # pbar = tqdm(all_data)
        for take, curr_data in data_list:
            pose_aa = curr_data["pose_aa"]
            seq_len = pose_aa.shape[0]

            if seq_len <= self.t_min:
                print(take, f" too short length: {seq_len} < {self.t_min}")
                continue
            # data_processed["pose_6d"][take] = curr_data["pose_6d"].reshape(
            #     seq_len, -1)
            data_processed["pose_aa"][take] = curr_data["pose_aa"].reshape(seq_len, -1)

            data_processed["trans"][take] = curr_data["trans"].reshape(seq_len, -1)

            data_processed["beta"][take] = np.repeat(curr_data["beta"][None,], seq_len, axis=0) if curr_data["beta"].shape[0] != seq_len else curr_data["beta"]

            pose_aa_torch = torch.from_numpy(curr_data["pose_aa"])
            pose_mat_torch = tR.axis_angle_to_matrix(pose_aa_torch[:, :66].reshape(-1, 3)).numpy().reshape(seq_len, -1, 3, 3)

            data_processed["root_orient"][take] = pose_mat_torch[:, 0:1].reshape(seq_len, -1)
            data_processed["pose_body"][take] = pose_mat_torch[:, 1:].reshape(seq_len, -1)

            if "gender" in curr_data:
                gender = (curr_data["gender"].item() if isinstance(curr_data["gender"], np.ndarray) else curr_data["gender"])
            else:
                gender = "neutral"

            if isinstance(gender, bytes):
                gender = gender.decode("utf-8")

            if gender == "neutral":
                gender = [0]
            elif gender == "male":
                gender = [1]
            elif gender == "female":
                gender = [2]
            elif isinstance(gender, list):
                pass
            else:
                import ipdb
                ipdb.set_trace()
                raise Exception("Gender Not Supported!!")

            data_processed["gender"][take] = np.repeat(
                gender,
                seq_len,
                axis=0,
            )
            if "obj_info" in curr_data:
                data_processed["obj_info"][take] = curr_data['obj_info']

        return data_processed

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

        return sample

    def sample_seq(
        self,
        full_sample=False,
        freq_dict=None,
        sampling_temp=0.2,
        sampling_freq=0.5,
        precision_mode=False,
        return_batch=True,
        fr_num=-1,
        full_fr_num=False,
        fr_start=-1,
    ):

        if freq_dict is None or len(freq_dict.keys()) != len(self.data_keys):
            self.curr_key = curr_key = random.choice(self.sample_keys)
        else:
            init_probs = np.exp(-np.array([ewma(np.array(freq_dict[k])[:, 0] == 1) if len(freq_dict[k]) > 0 else 0 for k in freq_dict.keys()]) / sampling_temp)
            init_probs = init_probs / init_probs.sum()
            self.curr_key = curr_key = (np.random.choice(self.data_keys, p=init_probs) if np.random.binomial(1, sampling_freq) else np.random.choice(self.data_keys))
        curr_pose_aa = self.data["pose_aa"][self.curr_key]
        seq_len = curr_pose_aa.shape[0]

        # self.curr_key = "S6-Discussion 1"
        return self.get_sample_from_key(self.curr_key, full_sample=full_sample, precision_mode=precision_mode, fr_num=fr_num, freq_dict=freq_dict, sampling_freq=sampling_freq, return_batch=return_batch, fr_start=fr_start)
