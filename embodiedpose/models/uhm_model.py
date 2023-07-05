import time, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from embodiedpose.models.humor.utils.transforms import (
     compute_world2aligned_mat, rotation_matrix_to_angle_axis)
# from embodiedpose.models.humor.utils.transforms import convert_to_rotmat
from embodiedpose.models.humor.utils.transforms import convert_to_rotmat_res as convert_to_rotmat
from embodiedpose.models.humor.body_model.utils import SMPL_JOINTS, SMPLH_PATH
from embodiedpose.models.humor.body_model.body_model import BodyModel

from human_body_prior.tools.model_loader import load_vposer
from human_body_prior.train.vposer_smpl import VPoser

IN_ROT_REPS = ['aa', '6d', 'mat']
OUT_ROT_REPS = ['aa', '6d', '9d']
ROT_REP_SIZE = {'aa': 3, '6d': 6, 'mat': 9, '9d': 9}
NUM_SMPL_JOINTS = len(SMPL_JOINTS)
NUM_BODY_JOINTS = NUM_SMPL_JOINTS - 1  # no root
BETA_SIZE = 16

POSTERIOR_OPTIONS = ['mlp']
PRIOR_OPTIONS = ['mlp']
DECODER_OPTIONS = ['mlp']

WORLD2ALIGN_NAME_CACHE = {
    'root_orient': None,
    'trans': None,
    'joints': None,
    'verts': None,
    'joints_vel': None,
    'verts_vel': None,
    'trans_vel': None,
    'root_orient_vel': None
}


def step(model,
         loss_func,
         data,
         dataset,
         device,
         cur_epoch,
         mode='train',
         use_gt_p=1.0):
    '''
    Given data for the current training step (batch),
    pulls out the necessary needed data,
    runs the model,
    calculates and returns the loss.

    - use_gt_p : the probability of using ground truth as input to each step rather than the model's own prediction
                 (1.0 is fully supervised, 0.0 is fully autoregressive)
    '''
    use_sched_samp = use_gt_p < 1.0
    batch_in, batch_out, meta = data

    prep_data = model.prepare_input(batch_in,
                                    device,
                                    data_out=batch_out,
                                    return_input_dict=True,
                                    return_global_dict=use_sched_samp)
    if use_sched_samp:
        x_past, x_t, gt_dict, input_dict, global_gt_dict = prep_data
    else:
        x_past, x_t, gt_dict, input_dict = prep_data

    B, T, S_in, _ = x_past.size()
    S_out = x_t.size(2)

    if not use_sched_samp:
        # fully supervised phase
        # start by using gt at every step, so just form all steps from all sequences into one large batch
        #       and get per-step predictions
        x_past_batched = x_past.reshape((B * T, S_in, -1))
        x_t_batched = x_t.reshape((B * T, S_out, -1))
        out_dict = model(x_past_batched, x_t_batched)
    else:
        # in scheduled sampling or fully autoregressive phase
        init_input_dict = dict()
        for k in input_dict.keys():
            init_input_dict[k] = input_dict[
                k][:, 0, :, :]  # only need first step for init
        # this out_dict is the global state
        sched_samp_out = model.scheduled_sampling(
            x_past,
            x_t,
            init_input_dict,
            p=use_gt_p,
            gender=meta['gender'],
            betas=meta['betas'].to(device),
            need_global_out=(not model.detach_sched_samp))
        if model.detach_sched_samp:
            out_dict = sched_samp_out
        else:
            out_dict, _ = sched_samp_out
        # gt must be global state for supervision in this case
        if not model.detach_sched_samp:
            print('USING global supervision')
            gt_dict = global_gt_dict

    # loss can be computed per output step in parallel
    # batch dicts accordingly
    for k in out_dict.keys():
        if k == 'posterior_distrib' or k == 'prior_distrib':
            m, v = out_dict[k]
            m = m.reshape((B * T, -1))
            v = v.reshape((B * T, -1))
            out_dict[k] = (m, v)
        else:
            out_dict[k] = out_dict[k].reshape((B * T * S_out, -1))
    for k in gt_dict.keys():
        gt_dict[k] = gt_dict[k].reshape((B * T * S_out, -1))

    gender_in = np.broadcast_to(
        np.array(meta['gender']).reshape((B, 1, 1, 1)), (B, T, S_out, 1))
    gender_in = gender_in.reshape((B * T * S_out, 1))
    betas_in = meta['betas'].reshape((B, T, 1, -1)).expand(
        (B, T, S_out, 16)).to(device)
    betas_in = betas_in.reshape((B * T * S_out, 16))
    loss, stats_dict = loss_func(out_dict,
                                 gt_dict,
                                 cur_epoch,
                                 gender=gender_in,
                                 betas=betas_in)

    return loss, stats_dict


class UHMModel(nn.Module):
    def __init__(
            self,
            in_rot_rep='aa',
            out_rot_rep='aa',
            latent_size=48,
            steps_in=1,
            conditional_prior=True,  # use a learned prior rather than standard normal
            output_delta=True,  # output change in state from decoder rather than next step directly
            posterior_arch='mlp',
            decoder_arch='mlp',
            prior_arch='mlp',
            model_data_config='smpl+joints+contacts',
            detach_sched_samp=True,  # if true, detaches outputs of previous step so gradients don't flow through many steps
            model_use_smpl_joint_inputs=False,  # if true, uses smpl joints rather than regressed joints to input at next step (during rollout and sched samp)
            model_smpl_batch_size=1,  # if using smpl joint inputs this should be batch_size of the smpl model (aka data input to rollout)
            use_vposer=False,
            use_gn=False):
        super(UHMModel, self).__init__()
        self.ignore_keys = []
        self.use_gn = True
        self.steps_in = steps_in
        self.steps_out = 1
        self.out_step_size = 1
        self.detach_sched_samp = detach_sched_samp
        self.output_delta = output_delta

        if self.steps_out > 1:
            raise NotImplementedError(
                'Only supported single step output currently.')

        if out_rot_rep not in OUT_ROT_REPS:
            raise Exception('Not a valid output rotation representation: %s' %
                            (out_rot_rep))
        if in_rot_rep not in IN_ROT_REPS:
            raise Exception('Not a valid input rotation representation: %s' %
                            (in_rot_rep))
        self.out_rot_rep = out_rot_rep
        self.in_rot_rep = in_rot_rep

        if posterior_arch not in POSTERIOR_OPTIONS:
            raise Exception('Not a valid encoder architecture: %s' %
                            (posterior_arch))
        if decoder_arch not in DECODER_OPTIONS:
            raise Exception('Not a valid decoder architecture: %s' %
                            (decoder_arch))
        if conditional_prior and prior_arch not in PRIOR_OPTIONS:
            raise Exception('Not a valid prior architecture: %s' %
                            (prior_arch))
        self.posterior_arch = posterior_arch
        self.decoder_arch = decoder_arch
        self.prior_arch = prior_arch

        # get the list of data names for the PROX dataset
        # self.data_names = [
        #     'trans', 'trans_vel', 'root_orient', 'root_orient_vel',
        #     'pose_body', 'joints', 'joints_vel'
        # ]
        # self.input_dim_list = [3, 3, 9, 3, 189, 66, 66]

        self.data_names = ['trans', 'root_orient', 'pose_body', 'joints']
        self.input_dim_list = [3, 9, 189, 66]

        self.out_data_names = ['trans', 'root_orient', 'pose_body']
        self.aux_in_data_names = self.aux_out_data_names = None  # auxiliary data will be returned as part of the input/output dictionary, but not the actual network input/output tensor
        self.pred_contacts = False
        if model_data_config.find(
                'contacts'
        ) >= 0:  # network is outputting contact classification as well and need to supervise, but not given as input to net.
            self.data_names.remove('contacts')
            self.aux_out_data_names = ['contacts']
            self.pred_contacts = True

        self.need_trans2joint = 'joints' in self.data_names or 'verts' in self.data_names
        self.model_data_config = model_data_config

        self.input_rot_dim = ROT_REP_SIZE[self.in_rot_rep]

        

        self.output_rot_dim = ROT_REP_SIZE[self.out_rot_rep]


        if out_rot_rep == "aa":
            self.output_dim_list = [3, 3, 63]
        elif out_rot_rep == "6d":
            self.output_dim_list = [3, 6, 126]

        self.delta_output_dim_list = [3, 9, 189]

        if self.pred_contacts:
            # account for contact classification output
            # data_dim('contacts') = 9
            self.output_dim_list.append(9)
            self.delta_output_dim_list.append(9)

        self.output_data_dim = sum(self.output_dim_list)
        self.delta_output_dim = sum(self.delta_output_dim_list)

        self.latent_size = latent_size
        past_data_dim = self.steps_in * sum(self.input_dim_list)
        t_data_dim = self.steps_out * sum(self.input_dim_list)

        # posterior encoder (given past and future, predict latent transition distribution)
        print('Using posterior architecture: %s' % (self.posterior_arch))
        if self.posterior_arch == 'mlp':
            # layer_list = [
            #     past_data_dim + t_data_dim, 1024, 1024, 1024, 1024,
            #     self.latent_size * 2
            # ]
            layer_list = [
                past_data_dim + t_data_dim, 1024, 1024, 1024, 1024,
                self.latent_size
            ]

            self.encoder = MLP(
                layers=layer_list,  # mu and sigma output
                nonlinearity=nn.GELU,
                use_gn=self.use_gn)

        # decoder (given past and latent transition, predict future) for the immediate next step
        print('Using decoder architecture: %s' % (self.decoder_arch))
        decoder_input_dim = past_data_dim + self.latent_size
        if self.decoder_arch == 'mlp':
            layer_list = [
                decoder_input_dim, 1024, 1024, 512, self.output_data_dim
            ]
            self.decoder = MLP(
                layers=layer_list,
                nonlinearity=nn.GELU,
                use_gn=self.use_gn,
                skip_input_idx=
                past_data_dim  # skip connect the latent to every layer
            )

        # prior (if conditional, given past predict latent transition distribution)
        self.use_conditional_prior = conditional_prior
        if self.use_conditional_prior:
            print('Using prior architecture: %s' % (self.prior_arch))
            layer_list = [
                past_data_dim, 1024, 1024, 1024, 1024, self.latent_size * 2
            ]
            self.prior_net = MLP(
                layers=layer_list,  # mu and sigma output
                nonlinearity=nn.GELU,
                use_gn=True)
        else:
            print('Using standard normal prior.')

        self.use_smpl_joint_inputs = model_use_smpl_joint_inputs
        self.smpl_batch_size = model_smpl_batch_size
        # if self.use_smpl_joint_inputs:
        # need a body model to compute the joints after each step.
        # print(
        #     'Using SMPL joints rather than regressed joints as input at each step for roll out and scheduled sampling...'
        # )
        # male_bm_path = os.path.join(SMPLH_PATH, 'male/model.npz')
        male_bm_path = os.path.join(SMPLH_PATH, 'SMPLH_MALE.npz')

        self.male_bm = BodyModel(bm_path=male_bm_path,
                                 num_betas=16,
                                 batch_size=self.smpl_batch_size,
                                 use_vtx_selector=True)
        # female_bm_path = os.path.join(SMPLH_PATH, 'female/model.npz')
        female_bm_path = os.path.join(SMPLH_PATH, 'SMPLH_FEMALE.npz')
        self.female_bm = BodyModel(bm_path=female_bm_path,
                                   num_betas=16,
                                   batch_size=self.smpl_batch_size,
                                   use_vtx_selector=True)
        # neutral_bm_path = os.path.join(SMPLH_PATH, 'neutcral/model.npz')
        neutral_bm_path = os.path.join(SMPLH_PATH, 'SMPLH_NEUTRAL.npz')
        self.neutral_bm = BodyModel(bm_path=neutral_bm_path,
                                    num_betas=16,
                                    batch_size=self.smpl_batch_size,
                                    use_vtx_selector=True)
        self.bm_dict = {
            'male': self.male_bm,
            'female': self.female_bm,
            'neutral': self.neutral_bm
        }
        for p in self.male_bm.parameters():
            p.requires_grad = False
        for p in self.female_bm.parameters():
            p.requires_grad = False
        for p in self.neutral_bm.parameters():
            p.requires_grad = False
        # self.ignore_keys = ['male_bm', 'female_bm', 'neutral_bm']
        if use_vposer:
            expr_dir = '/hdd/zen/data/SMPL/vposer_v1_0/'  # directory for the trained model along with the model code. obtain from https://smpl-x.is.tue.mpg.de/downloads
            self.vp, ps = load_vposer(expr_dir, vp_model=VPoser)
            for p in self.vp.parameters():
                p.requires_grad = False

    def split_by_name_and_dim(self,
                              output,
                              idx_list,
                              name_list,
                              convert_rots=True):
        '''
        Given the output of the decoder, splits into each state component.
        Also transform rotation representation to matrices.

        Input:
        - decoder_out  (B x steps_out x D)

        Returns:
        - output dict
        '''
        B = output.size(0)
        output = output.reshape((B, self.steps_out, -1))
        assert (output.shape[-1] == self.delta_output_dim)

        # collect outputs
        out_dict = dict()
        sidx = 0
        for cur_name, cur_idx in zip(name_list, idx_list):
            eidx = sidx + cur_idx
            out_dict[cur_name] = output[:, :, sidx:eidx]
            sidx = eidx

        # transform rotations
        if convert_rots and not self.output_delta:  # output delta already gives rotmats
            if 'root_orient' in self.data_names:
                out_dict['root_orient'] = convert_to_rotmat(
                    out_dict['root_orient'], rep=self.out_rot_rep)
            if 'pose_body' in self.data_names:
                out_dict['pose_body'] = convert_to_rotmat(
                    out_dict['pose_body'], rep=self.out_rot_rep)

        return out_dict

    def split_output(self, decoder_out, convert_rots=True):
        '''
        Given the output of the decoder, splits into each state component.
        Also transform rotation representation to matrices.

        Input:
        - decoder_out  (B x steps_out x D)

        Returns:
        - output dict
        '''
        B = decoder_out.size(0)
        decoder_out = decoder_out.reshape((B, self.steps_out, -1))

        assert (decoder_out.shape[-1] == self.delta_output_dim)

        # collect outputs
        name_list = self.out_data_names
        if self.aux_out_data_names is not None:
            name_list = name_list + self.aux_out_data_names
        idx_list = self.delta_output_dim_list if self.output_delta else self.output_dim_list
        out_dict = dict()
        sidx = 0
        for cur_name, cur_idx in zip(name_list, idx_list):
            eidx = sidx + cur_idx
            out_dict[cur_name] = decoder_out[:, :, sidx:eidx]
            sidx = eidx

        # transform rotations
        if convert_rots and not self.output_delta:  # output delta already gives rotmats
            if 'root_orient' in self.data_names:
                out_dict['root_orient'] = convert_to_rotmat(
                    out_dict['root_orient'], rep=self.out_rot_rep)
            if 'pose_body' in self.data_names:
                out_dict['pose_body'] = convert_to_rotmat(
                    out_dict['pose_body'], rep=self.out_rot_rep)

        return out_dict

    def split_input(self, decoder_out, convert_rots=True):
        '''
        Given the input of the models, splits into each state component.
        Also transform rotation representation to matrices.

        Input:
        - decoder_out  (B x steps_out x D)

        Returns:
        - output dict
        '''
        B = decoder_out.size(0)
        decoder_out = decoder_out.reshape((B, self.steps_out, -1))
        assert (decoder_out.shape[-1] == sum(self.input_dim_list))

        # collect outputs
        name_list = self.data_names
        if self.aux_out_data_names is not None:
            name_list = name_list + self.aux_out_data_names
        idx_list = self.input_dim_list
        out_dict = dict()
        sidx = 0
        for cur_name, cur_idx in zip(name_list, idx_list):
            eidx = sidx + cur_idx
            out_dict[cur_name] = decoder_out[:, :, sidx:eidx]
            sidx = eidx

        # transform rotations
        if convert_rots and not self.output_delta:  # output delta already gives rotmats
            if 'root_orient' in self.data_names:
                out_dict['root_orient'] = convert_to_rotmat(
                    out_dict['root_orient'], rep=self.out_rot_rep)
            if 'pose_body' in self.data_names:
                out_dict['pose_body'] = convert_to_rotmat(
                    out_dict['pose_body'], rep=self.out_rot_rep)

        return out_dict

    def forward(self, x_past, x_t):
        '''
        single step full forward pass. This uses the posterior for sampling, not the prior.

        Input:
        - x_past (B x steps_in x D)
        - x_t    (B x steps_out x D)

        Returns dict of:
        - x_pred (B x steps_out x D)
        - posterior_distrib (Normal(mu, sigma))
        - prior_distrib (Normal(mu, sigma))
        '''

        B, _, D = x_past.size()
        past_in = x_past.reshape((B, -1))
        t_in = x_t.reshape((B, -1))

        x_pred_dict = self.single_step(past_in, t_in)

        return x_pred_dict

    def single_step(self, past_in, t_in):
        '''
        single step that computes both prior and posterior for training. Samples from posterior
        '''
        B = past_in.size(0)
        # use past and future to encode latent transition
        qm, qv = self.posterior(past_in, t_in)

        # prior
        pm, pv = None, None
        if self.use_conditional_prior:
            # predict prior based on past
            pm, pv = self.prior(past_in)
        else:
            # use standard normal
            pm, pv = torch.zeros_like(qm), torch.ones_like(qv)

        # sample from posterior using reparam trick
        z = self.rsample(qm, qv)

        # decode to get next step
        decoder_out = self.decode(z, past_in)
        decoder_out = decoder_out.reshape(
            (B, self.steps_out, -1))  # B x steps_out x D_out

        # split output predictions and transform out rotations to matrices
        x_pred_dict = self.split_output(decoder_out)

        x_pred_dict['posterior_distrib'] = (qm, qv)
        x_pred_dict['prior_distrib'] = (pm, pv)

        return x_pred_dict

    def prior(self, past_in):
        '''
        Encodes the posterior distribution using the past and future states.

        Input:
        - past_in (B x steps_in*D)
        '''
        prior_out = self.prior_net(past_in)
        mean = prior_out[:, :self.latent_size]
        logvar = prior_out[:, self.latent_size:]
        var = torch.exp(logvar)
        return mean, var

    def posterior(self, past_in, t_in=None):
        '''
        Encodes the posterior distribution using the past and future states.

        Input:
        - past_in (B x steps_in*D)
        - t_in    (B x steps_out*D)
        '''
        if not t_in is None:
            encoder_in = torch.cat([past_in, t_in], axis=1)
        else:
            encoder_in = past_in

        encoder_out = self.encoder(encoder_in)
        mean = encoder_out[:, :self.latent_size]
        logvar = encoder_out[:, self.latent_size:]
        var = torch.exp(logvar)

        return mean, var

    def posterior_unit(self, past_in, t_in=None):
        '''
        Encodes the posterior distribution using the past and future states.

        Input:
        - past_in (B x steps_in*D)
        - t_in    (B x steps_out*D)
        '''
        if not t_in is None:
            encoder_in = torch.cat([past_in, t_in], axis=1)
        else:
            encoder_in = past_in

        encoder_out = self.encoder(encoder_in)
        encoder_out = encoder_out / encoder_out.norm(p=2, dim=1, keepdim=True)

        return encoder_out

    def rsample(self, mu, var):
        '''
        Return gaussian sample of (mu, var) using reparameterization trick.
        '''
        eps = torch.randn_like(mu)
        z = mu + eps * torch.sqrt(var)
        return z

    def decode(self, z, past_in):
        '''
        Decodes prediction from the latent transition and past states

        Input:
        - z       (B x latent_size)
        - past_in (B x steps_in*D)

        Returns:
        - decoder_out (B x steps_out*D)
        '''
        B = z.size(0)
        decoder_in = torch.cat([past_in, z], axis=1)
        decoder_out = self.decoder(decoder_in).reshape((B, 1, -1))
        past_in_dict = self.split_input(past_in)

        if self.output_delta:
            # network output is the residual, add to the input to get final output

            final_out_list = []
            in_sidx = out_sidx = 0
            decode_out_dim_list = self.output_dim_list
            if self.pred_contacts:
                decode_out_dim_list = decode_out_dim_list[:
                                                          -1]  # do contacts separately

            for in_dim_idx, out_dim_idx, data_name in zip(
                    self.input_dim_list, decode_out_dim_list,
                    self.out_data_names):
                in_eidx = in_sidx + in_dim_idx
                out_eidx = out_sidx + out_dim_idx

                # add residual to input (and transform as necessary for rotations)
                in_val = past_in_dict[data_name]
                out_val = decoder_out[:, :, out_sidx:out_eidx]
                if data_name in ['root_orient', 'pose_body']:
                    if self.in_rot_rep != 'mat':
                        in_val = convert_to_rotmat(in_val, rep=self.in_rot_rep)
                    out_val = convert_to_rotmat(out_val, rep=self.out_rot_rep)

                    in_val = in_val.reshape((B, 1, -1, 3, 3))
                    out_val = out_val.reshape((B, self.steps_out, -1, 3, 3))

                    rot_in = torch.matmul(out_val, in_val).reshape(
                        (B, self.steps_out,
                         -1))  # rotate by predicted residual
                    final_out_list.append(rot_in)
                else:
                    final_out_list.append(out_val + in_val)

                in_sidx = in_eidx
                out_sidx = out_eidx
            if self.pred_contacts:
                final_out_list.append(decoder_out[:, :, out_sidx:])

            decoder_out = torch.cat(final_out_list, dim=2)

        decoder_out = decoder_out.reshape((B, -1))

        return decoder_out

    def apply_world2local_trans(self,
                                world2local_trans,
                                world2local_rot,
                                trans2joint,
                                input_dict,
                                output_dict,
                                invert=False):
        '''
        Applies the given world2local transformation to the data in input_dict and stores the result in output_dict.
        If invert is true, applies local2world.
        - world2local_trans : B x 3 or B x 1 x 3
        - world2local_rot :   B x 3 x 3 or B x 1 x 3 x 3
        - trans2joint : B x 1 x 1 x 3
        '''
        B = world2local_trans.size(0)
        world2local_rot = world2local_rot.reshape((B, 1, 3, 3))
        world2local_trans = world2local_trans.reshape((B, 1, 3))
        trans2joint = trans2joint.reshape((B, 1, 1, 3))
        if invert:
            local2world_rot = world2local_rot.transpose(3, 2)
        for k, v in input_dict.items():
            # apply differently depending on which data value it is
            if k not in WORLD2ALIGN_NAME_CACHE:
                # frame of reference is irrelevant, just copy to output
                output_dict[k] = input_dict[k]
                continue
            S = input_dict[k].size(1)

            if k in ['root_orient']:
                # rot: B x S x 3 x 3 sized rotation matrix input
                input_mat = input_dict[k].reshape(
                    (B, S, 3, 3))  # make sure not B x S x 9
                if invert:
                    output_dict[k] = torch.matmul(local2world_rot,
                                                  input_mat).reshape((B, S, 9))
                else:
                    output_dict[k] = torch.matmul(world2local_rot,
                                                  input_mat).reshape((B, S, 9))
            elif k in ['trans']:
                # trans + rot : B x S x 3
                input_trans = input_dict[k]
                if invert:
                    output_trans = torch.matmul(
                        local2world_rot, input_trans.reshape(
                            (B, S, 3, 1)))[:, :, :, 0]
                    output_trans = output_trans - world2local_trans
                    output_dict[k] = output_trans
                else:
                    input_trans = input_trans + world2local_trans
                    output_dict[k] = torch.matmul(
                        world2local_rot, input_trans.reshape(
                            (B, S, 3, 1)))[:, :, :, 0]
            elif k in ['joints', 'verts']:
                # trans + joint + rot : B x S x J x 3
                J = input_dict[k].size(2) // 3

                input_pts = input_dict[k].reshape((B, S, J, 3))
                if invert:
                    input_pts = input_pts + trans2joint
                    output_pts = torch.matmul(
                        local2world_rot.reshape((B, 1, 1, 3, 3)),
                        input_pts.reshape((B, S, J, 3, 1)))[:, :, :, :, 0]
                    output_pts = output_pts - trans2joint - world2local_trans.reshape(
                        (B, 1, 1, 3))
                    output_dict[k] = output_pts.reshape((B, S, J * 3))
                else:

                    input_pts = input_pts + world2local_trans.reshape(
                        (B, 1, 1, 3)) + trans2joint
                    output_pts = torch.matmul(
                        world2local_rot.reshape((B, 1, 1, 3, 3)),
                        input_pts.reshape((B, S, J, 3, 1)))[:, :, :, :, 0]
                    output_pts = output_pts - trans2joint
                    output_dict[k] = output_pts.reshape((B, S, J * 3))
            elif k in ['joints_vel', 'verts_vel']:
                # rot : B x S x J x 3
                J = input_dict[k].size(2) // 3
                input_pts = input_dict[k].reshape((B, S, J, 3, 1))
                if invert:
                    outuput_pts = torch.matmul(
                        local2world_rot.reshape((B, 1, 1, 3, 3)),
                        input_pts)[:, :, :, :, 0]
                    output_dict[k] = outuput_pts.reshape((B, S, J * 3))
                else:
                    output_pts = torch.matmul(
                        world2local_rot.reshape((B, 1, 1, 3, 3)),
                        input_pts)[:, :, :, :, 0]
                    output_dict[k] = output_pts.reshape((B, S, J * 3))
            elif k in ['trans_vel', 'root_orient_vel']:
                # rot : B x S x 3
                input_pts = input_dict[k].reshape((B, S, 3, 1))
                if invert:
                    output_dict[k] = torch.matmul(local2world_rot,
                                                  input_pts)[:, :, :, 0]
                else:
                    output_dict[k] = torch.matmul(world2local_rot,
                                                  input_pts)[:, :, :, 0]
            else:
                print(
                    'Received an unexpected key when transforming world2local: %s!'
                    % (k))
                exit()

        return output_dict

    def zero_pad_tensors(self, pad_list, pad_size):
        '''
        Assumes tensors in pad_list are B x D
        '''
        new_pad_list = []
        for pad_idx, pad_tensor in enumerate(pad_list):
            padding = torch.zeros(
                (pad_size, pad_tensor.size(1))).to(pad_tensor)
            new_pad_list.append(torch.cat([pad_tensor, padding], dim=0))
        return new_pad_list

    def canonicalize_input(self, global_dict, trans2joint=None):
        '''
        Converts the input dict to the canonical form.
        '''
        B, seq_len, _ = global_dict["trans"].shape
        global_dict = {
            k: v.reshape(B * seq_len, 1, *v.shape[2:])
            for k, v in global_dict.items()
        }

        root_orient_mat = global_dict["root_orient"]
        root_orient_mat = root_orient_mat[:, -1].reshape((B * seq_len, 3, 3))

        world2aligned_rot = compute_world2aligned_mat(root_orient_mat)

        world2aligned_trans = torch.cat([
            -global_dict['trans'][:, 0, :2],
            torch.zeros((B * seq_len, 1)).to(root_orient_mat)
        ],
                                        axis=1)

        if trans2joint is None:
            trans2joint = -(global_dict['joints'][:, 0, :2] +
                            world2aligned_trans[:, :2])
            trans2joint = torch.cat(
                [trans2joint,
                 torch.zeros((B * seq_len, 1)).to(trans2joint)],
                axis=1).reshape((B * seq_len, 1, 1, 3))

        local_dict = dict()
        local_dict = self.apply_world2local_trans(world2aligned_trans,
                                                  world2aligned_rot,
                                                  trans2joint,
                                                  global_dict,
                                                  local_dict,
                                                  invert=False)
        return local_dict, world2aligned_rot, world2aligned_trans

    def decode_global(self,
                      past_global_in_dict,
                      z,
                      trans2joint=None,
                      cam_params=None,
                      extra_info=None):
        '''
        Given a past global state, formats it (transform each step into local frame and makde B x steps_in x D)
        and runs inference.

        Rotations should be in in_rot_rep format.
        trans2joint: Need to this value to be fixed due to the instability of Humor's joint predicting code.
        '''

        B, device = past_global_in_dict["trans"].shape[0], past_global_in_dict[
            "trans"].device
        # get world2aligned rot and translation
        if cam_params is not None:
            R, aR = cam_params['R'], cam_params['aR']
            past_global_in_dict["root_orient"] = torch.matmul(
                torch.matmul(R.T, aR.T),
                past_global_in_dict["root_orient"].reshape(B, 3, 3)).reshape(
                    past_global_in_dict["root_orient"].shape)

        root_orient_mat = past_global_in_dict["root_orient"]
        root_orient_mat = root_orient_mat[:, -1].reshape((B, 3, 3))

        world2aligned_rot = compute_world2aligned_mat(root_orient_mat).double()
        world2aligned_trans = torch.cat([
            -past_global_in_dict['trans'][:, 0, :2],
            torch.zeros((B, 1)).to(root_orient_mat)
        ],
                                        axis=1)

        if trans2joint is None:
            trans2joint = -(
                past_global_in_dict['joints'][:, 0, :2] +
                world2aligned_trans[:, :2]
            )  # we cannot make the assumption that the first frame is already canonical

            trans2joint = torch.cat(
                [trans2joint, torch.zeros(
                    (B, 1)).to(trans2joint)], axis=1).reshape((B, 1, 1, 3))

        past_in_dict = dict()
        # past_global_in_dict = {k: v.double() for k, v in past_global_in_dict.items() }
        past_in_dict = self.apply_world2local_trans(world2aligned_trans,
                                                    world2aligned_rot,
                                                    trans2joint,
                                                    past_global_in_dict,
                                                    past_in_dict,
                                                    invert=False)

        past_in = []
        for k in self.data_names:
            past_in.append(past_in_dict[k].reshape(B, -1))
        past_in = torch.cat(past_in, axis=1)

        # To compute the motion prior loss
        if self.use_conditional_prior:
            pm, pv = self.prior(past_in.double())
        z, past_in = z.double(), past_in.double()
        decoder_out = self.decode(z, past_in)

        decoder_out = decoder_out.reshape((B, 1, -1))  # B x 1 x D_out

        # import ipdb; ipdb.set_trace()
        # next_out = self.split_input(extra_info['obs_next_gt'])
        next_out = self.split_output(decoder_out)

        next_global_out = dict()
        next_global_out = self.apply_world2local_trans(world2aligned_trans,
                                                       world2aligned_rot,
                                                       trans2joint,
                                                       next_out,
                                                       next_global_out,
                                                       invert=True)

        if cam_params is not None:
            R, aR = cam_params['R'], cam_params['aR']
            next_global_out['root_orient'] = torch.matmul(
                torch.matmul(R, aR), next_global_out['root_orient'].reshape(
                    (B, 3, 3))).reshape(next_global_out['root_orient'].shape)

        if self.use_conditional_prior:
            return next_global_out, pm, pv
        else:
            return next_global_out, None, None

    def step_action(self, past_in, action, output_delta=True):
        '''
        Decodes prediction from the latent transition and past states

        Input:
        - z       (B x latent_size)
        - past_in (B x steps_in*D)

        Returns:
        - decoder_out (B x steps_out*D)
        '''
        B = action.size(0)
        action = action.reshape((B, 1, -1))

        past_in_dict = self.split_input(past_in)

        if self.output_delta:
            # network output is the residual, add to the input to get final output

            final_out_list = []
            in_sidx = out_sidx = 0
            decode_out_dim_list = self.output_dim_list
            if self.pred_contacts:
                decode_out_dim_list = decode_out_dim_list[:
                                                          -1]  # do contacts separately

            for in_dim_idx, out_dim_idx, data_name in zip(
                    self.input_dim_list, decode_out_dim_list,
                    self.out_data_names):
                in_eidx = in_sidx + in_dim_idx
                out_eidx = out_sidx + out_dim_idx

                # add residual to input (and transform as necessary for rotations)
                in_val = past_in_dict[data_name]
                out_val = action[:, :, out_sidx:out_eidx]
                if data_name in ['root_orient']:
                    if self.in_rot_rep != 'mat':
                        in_val = convert_to_rotmat(in_val, rep=self.in_rot_rep)
                    out_val = convert_to_rotmat(out_val, rep=self.out_rot_rep)

                    in_val = in_val.reshape((B, 1, -1, 3, 3))
                    out_val = out_val.reshape((B, self.steps_out, -1, 3, 3))

                    rot_in = torch.matmul(out_val, in_val).reshape(
                        (B, self.steps_out,
                         -1))  # rotate by predicted residual
                    final_out_list.append(rot_in)
                elif data_name in ['pose_body']:
                    if self.in_rot_rep != 'mat':
                        in_val = convert_to_rotmat(in_val, rep=self.in_rot_rep)
                    out_val = convert_to_rotmat(out_val, rep=self.out_rot_rep)

                    in_val = in_val.reshape((B, 1, -1, 3, 3))
                    out_val = out_val.reshape((B, self.steps_out, -1, 3, 3))

                    rot_in = torch.matmul(out_val, in_val).reshape(
                        (B, self.steps_out,
                         -1))  # rotate by predicted residual
                    final_out_list.append(rot_in)
                    ######################## Direct prediction test ########################
                    # if self.in_rot_rep != 'mat':
                    #     in_val = convert_to_rotmat(in_val, rep=self.in_rot_rep)
                    
                    # rot_in = in_val.reshape((B, self.steps_out, -1)) 
                    # final_out_list.append(rot_in)
                else:
                    final_out_list.append(out_val + in_val)

                in_sidx = in_eidx
                out_sidx = out_eidx
            if self.pred_contacts:
                final_out_list.append(action[:, :, out_sidx:])

            action = torch.cat(final_out_list, dim=2)

        action = action.reshape((B, -1))

        return action

    def compute_local_feat(self, past_global_in_dict):
        B, device = past_global_in_dict["trans"].shape[0], past_global_in_dict[
            "trans"].device
        # get world2aligned rot and translation
        root_orient_mat = past_global_in_dict["root_orient"]
        root_orient_mat = root_orient_mat[:, -1].reshape((B, 3, 3))
        world2aligned_rot = compute_world2aligned_mat(root_orient_mat).double()
        world2aligned_trans = torch.cat([
            -past_global_in_dict['trans'][:, 0, :2],
            torch.zeros((B, 1)).to(root_orient_mat)
        ],
                                        axis=1)

        # trans2joint = torch.tensor([[[[0.0018, 0.2233, 0.0000]]]
        # ]).repeat(B, 1, 1, 1).to(root_orient_mat)
        trans2joint = torch.zeros((B, 1, 1, 3)).to(root_orient_mat)

        # transform to local frame
        past_in_dict = dict()
        past_global_in_dict = {
            k: v.double()
            for k, v in past_global_in_dict.items()
        }
        past_in_dict = self.apply_world2local_trans(world2aligned_trans,
                                                    world2aligned_rot,
                                                    trans2joint,
                                                    past_global_in_dict,
                                                    past_in_dict,
                                                    invert=False)
        past_in = []
        for k in self.data_names:
            past_in.append(past_in_dict[k].reshape(B, -1))
        past_in = torch.cat(past_in, axis=1)
        return past_in

    def step_state(self,
                   past_global_in_dict,
                   residual,
                   cam_params=None,
                   output_delta=True):
        '''
        Given a past global state, and residual action, computes the next global state.
        '''

        B, device = past_global_in_dict["trans"].shape[0], past_global_in_dict[
            "trans"].device
        # get world2aligned rot and translation

        if cam_params is not None:
            R, aR = cam_params['R'], cam_params['aR']
            past_global_in_dict["root_orient"] = torch.matmul(
                torch.matmul(R.T, aR.T),
                past_global_in_dict["root_orient"].reshape(B, 3, 3)).reshape(
                    past_global_in_dict["root_orient"].shape)

        root_orient_mat = past_global_in_dict["root_orient"]
        root_orient_mat = root_orient_mat[:, -1].reshape((B, 3, 3))

        world2aligned_rot = compute_world2aligned_mat(root_orient_mat).double()
        world2aligned_trans = torch.cat([
            -past_global_in_dict['trans'][:, 0, :2],
            torch.zeros((B, 1)).to(root_orient_mat)
        ],
                                        axis=1)

        # trans2joint = torch.tensor([[[[0.0018, 0.2233, 0.0000]]]
        # ]).repeat(B, 1, 1, 1).to(root_orient_mat)
        trans2joint = torch.zeros((B, 1, 1, 3)).to(root_orient_mat)

        # transform to local frame
        past_in_dict = dict()
        past_global_in_dict = {
            k: v.double()
            for k, v in past_global_in_dict.items()
        }
        past_in_dict = self.apply_world2local_trans(world2aligned_trans,
                                                    world2aligned_rot,
                                                    trans2joint,
                                                    past_global_in_dict,
                                                    past_in_dict,
                                                    invert=False)
        past_in = []
        for k in self.data_names:
            past_in.append(past_in_dict[k].reshape(B, -1))
        past_in = torch.cat(past_in, axis=1)
        # To compute the motion prior loss
        # pm, pv = self.prior(past_in.double())
        decoder_out = self.step_action(past_in, residual, output_delta)
        decoder_out = decoder_out.reshape((B, 1, -1))  # B x 1 x D_out

        next_out = self.split_output(decoder_out)

        next_global_out = dict()
        next_global_out = self.apply_world2local_trans(world2aligned_trans,
                                                       world2aligned_rot,
                                                       trans2joint,
                                                       next_out,
                                                       next_global_out,
                                                       invert=True)
        if cam_params is not None:
            R, aR = cam_params['R'], cam_params['aR']
            next_global_out['root_orient'] = torch.matmul(
                torch.matmul(R, aR), next_global_out['root_orient'].reshape(
                    (B, 3, 3))).reshape(next_global_out['root_orient'].shape)

        # return next_global_out, pm, pv
        return next_global_out

    def direct_state(self, action, use_vposer=True):
        '''
        Given a past global state, and action, computes the prior and next global state directly.. 
        hack stuff
        '''
        # Computes the direct state
        next_global_out = {}
        trans = action[:, None, :3]
        root_orient = convert_to_rotmat(action[:, None, 3:9], rep="6d")

        if use_vposer:
            pose_body = self.vp.decode(action[:, None, 9:41])
        else:
            pose_body = convert_to_rotmat(action[:, None, 6:132], rep="6d")

        next_global_out['trans'] = trans
        next_global_out['root_orient'] = root_orient
        next_global_out['pose_body'] = pose_body

        return next_global_out

    def sample_step(self,
                    past_in,
                    t_in=None,
                    use_mean=False,
                    z=None,
                    return_prior=False,
                    return_z=False):
        '''
        Given past, samples next future state by sampling from prior or posterior and decoding.
        If z (B x D) is not None, uses the given z instead of sampling from posterior or prior

        Returns:
        - decoder_out : (B x steps_out x D) output of the decoder for the immediate next step
        '''
        B = past_in.size(0)

        pm, pv = None, None
        if t_in is not None:
            # use past and future to encode latent transition
            pm, pv = self.posterior(past_in, t_in)
        else:
            # prior
            if self.use_conditional_prior:
                # predict prior based on past
                pm, pv = self.prior(past_in)
            else:
                # use standard normal
                pm, pv = torch.zeros(
                    (B, self.latent_size)).to(past_in), torch.ones(
                        (B, self.latent_size)).to(past_in)

        # sample from distrib or use mean
        if z is None:
            if not use_mean:
                z = self.rsample(pm, pv)
            else:
                z = pm  # NOTE: use mean

        # decode to get next step
        decoder_out = self.decode(z, past_in)
        decoder_out = decoder_out.reshape(
            (B, self.steps_out, -1))  # B x steps_out x D_out

        out_dict = {'decoder_out': decoder_out}
        if return_prior:
            out_dict['prior'] = (pm, pv)
        if return_z:
            out_dict['z'] = z

        return out_dict

    def canonicalize_input_double(self,
                                  past_global_in_dict,
                                  next_global_in_dict,
                                  cam_params=None,
                                  split_input=True,
                                  trans2joint=None,
                                  split_output=True,
                                  return_info=False, 
                                  ):
        if split_input:
            past_global_in_dict = self.split_input(past_global_in_dict)
            next_global_in_dict = self.split_input(next_global_in_dict)

        B, seq_len, _ = past_global_in_dict["trans"].shape

        if cam_params is not None:
            R, aR = cam_params['R'], cam_params['aR']
            past_global_in_dict["root_orient"] = torch.matmul(
                torch.matmul(R.T, aR.T),
                past_global_in_dict["root_orient"].reshape(
                    B, seq_len, 3,
                    3)).reshape(past_global_in_dict["root_orient"].shape)
            next_global_in_dict["root_orient"] = torch.matmul(
                torch.matmul(R.T, aR.T),
                next_global_in_dict["root_orient"].reshape(
                    B, seq_len, 3,
                    3)).reshape(next_global_in_dict["root_orient"].shape)

        # Need to make sure that the sequence is perserved.
        past_global_in_dict = {
            k: v.reshape(B * seq_len, 1, *v.shape[2:])
            for k, v in past_global_in_dict.items()
        }
        next_global_in_dict = {
            k: v.reshape(B * seq_len, 1, *v.shape[2:])
            for k, v in next_global_in_dict.items()
        }

        # if seq_len > 1:
        #     import ipdb; ipdb.set_trace()

        root_orient_mat = past_global_in_dict["root_orient"]
        root_orient_mat = root_orient_mat[:, -1].reshape((B * seq_len, 3, 3))

        world2aligned_rot = compute_world2aligned_mat(root_orient_mat)

        world2aligned_trans = torch.cat([
            -past_global_in_dict['trans'][:, 0, :2],
            torch.zeros((B * seq_len, 1)).to(root_orient_mat)
        ],
                                        axis=1)

        if trans2joint is None:
            trans2joint = -(
                past_global_in_dict['joints'][:, 0, :2] +
                world2aligned_trans[:, :2]
            )  # we cannot make the assumption that the first frame is already canonical
            trans2joint = torch.cat(
                [trans2joint,
                 torch.zeros((B * seq_len, 1)).to(trans2joint)],
                axis=1).reshape((B * seq_len, 1, 1, 3))

        # transform to local frame
        past_in_dict = dict()
        past_in_dict = self.apply_world2local_trans(world2aligned_trans,
                                                    world2aligned_rot,
                                                    trans2joint,
                                                    past_global_in_dict,
                                                    past_in_dict,
                                                    invert=False)
        # transform to local frame
        next_in_dict = dict()

        next_in_dict = self.apply_world2local_trans(world2aligned_trans,
                                                    world2aligned_rot,
                                                    trans2joint,
                                                    next_global_in_dict,
                                                    next_in_dict,
                                                    invert=False)

        out_info  = {
            "world2aligned_trans": world2aligned_trans, 
            "world2aligned_rot": world2aligned_rot, 
            "trans2joint": trans2joint
        }

        if split_output:
            if return_info:
                return past_in_dict, next_in_dict, out_info
            else:
                return past_in_dict, next_in_dict
        else:
            past_in = []
            for k in self.data_names:
                past_in.append(past_in_dict[k].reshape(B, seq_len, -1))
            past_in = torch.cat(past_in, dim=2)

            next_out = []
            for k in self.data_names:
                next_out.append(next_in_dict[k].reshape(B, seq_len, -1))
            next_out = torch.cat(next_out, dim=2)

            if return_info:
                return past_in, next_out, out_info
            else:
                return past_in, next_out

    def infer_global_seq(self, global_seq, full_forward_pass=False):
        '''
        Given a sequence of global states, formats it (transform each step into local frame and makde B x steps_in x D)
        and runs inference (compute prior/posterior of z for the sequence).

        If full_forward_pass is true, does an entire forward pass at each step rather than just inference.

        Rotations should be in in_rot_rep format.
        '''
        # used to compute output zero padding
        needed_future_steps = (self.steps_out - 1) * self.out_step_size

        prior_m_seq = []
        prior_v_seq = []
        post_m_seq = []
        post_v_seq = []
        pred_dict_seq = []
        B, T, _ = global_seq[list(global_seq.keys())[0]].size()
        J = len(SMPL_JOINTS)
        trans2joint = None
        for t in range(T - 1):
            # get world2aligned rot and translation
            world2aligned_rot = world2aligned_trans = None

            root_orient_mat = global_seq['root_orient'][:, t, :].reshape(
                (B, 3, 3))
            world2aligned_rot = compute_world2aligned_mat(root_orient_mat)
            world2aligned_trans = torch.cat([
                -global_seq['trans'][:, t, :2],
                torch.zeros((B, 1)).to(root_orient_mat)
            ],
                                            axis=1)

            # compute trans2joint at first step
            if t == 0 and self.need_trans2joint:
                trans2joint = -(
                    global_seq['joints'][:, t, :2] + world2aligned_trans[:, :2]
                )  # we cannot make the assumption that the first frame is already canonical
                trans2joint = torch.cat(
                    [trans2joint,
                     torch.zeros((B, 1)).to(trans2joint)], axis=1).reshape(
                         (B, 1, 1, 3))

            # get current window
            cur_data_dict = dict()
            for k in global_seq.keys():
                # get in steps
                in_sidx = max(0, t - self.steps_in + 1)
                cur_in_seq = global_seq[k][:, in_sidx:(t + 1), :]
                if cur_in_seq.size(1) < self.steps_in:
                    # must zero pad front
                    num_pad_steps = self.steps_in - cur_in_seq.size(1)
                    cur_padding = torch.zeros(
                        (cur_in_seq.size(0), num_pad_steps,
                         cur_in_seq.size(2))).to(
                             cur_in_seq)  # assuming all data is B x T x D
                    cur_in_seq = torch.cat([cur_padding, cur_in_seq], axis=1)

                # get out steps
                cur_out_seq = global_seq[k][:, (t + 1):(
                    t + 2 + needed_future_steps):self.out_step_size]
                if cur_out_seq.size(1) < self.steps_out:
                    # zero pad
                    num_pad_steps = self.steps_out - cur_out_seq.size(1)
                    cur_padding = torch.zeros_like(cur_out_seq[:, 0])
                    cur_padding = torch.stack([cur_padding] * num_pad_steps,
                                              axis=1)
                    cur_out_seq = torch.cat([cur_out_seq, cur_padding], axis=1)
                cur_data_dict[k] = torch.cat([cur_in_seq, cur_out_seq], axis=1)

            # transform to local frame
            cur_data_dict = self.apply_world2local_trans(world2aligned_trans,
                                                         world2aligned_rot,
                                                         trans2joint,
                                                         cur_data_dict,
                                                         cur_data_dict,
                                                         invert=False)

            # create x_past and x_t
            # cat all inputs together to form past_in
            in_data_list = []
            for k in self.data_names:
                in_data_list.append(cur_data_dict[k][:, :self.steps_in, :])
            x_past = torch.cat(in_data_list, axis=2)
            # cat all outputs together to form x_t
            out_data_list = []
            for k in self.data_names:
                out_data_list.append(cur_data_dict[k][:, self.steps_in:, :])
            x_t = torch.cat(out_data_list, axis=2)

            if full_forward_pass:
                x_pred_dict = self(x_past, x_t)
                pred_dict_seq.append(x_pred_dict)
            else:
                # perform inference
                prior_z, posterior_z = self.infer(x_past, x_t)
                # save z
                prior_m_seq.append(prior_z[0])
                prior_v_seq.append(prior_z[1])
                post_m_seq.append(posterior_z[0])
                post_v_seq.append(posterior_z[1])

        if full_forward_pass:
            # pred_dict_seq
            pred_seq_out = dict()
            for k in pred_dict_seq[0].keys():
                # print(k)
                if k == 'posterior_distrib' or k == 'prior_distrib':
                    m = torch.stack([
                        pred_dict_seq[i][k][0]
                        for i in range(len(pred_dict_seq))
                    ],
                                    axis=1)
                    v = torch.stack([
                        pred_dict_seq[i][k][1]
                        for i in range(len(pred_dict_seq))
                    ],
                                    axis=1)
                    pred_seq_out[k] = (m, v)
                else:
                    pred_seq_out[k] = torch.stack([
                        pred_dict_seq[i][k] for i in range(len(pred_dict_seq))
                    ],
                                                  axis=1)

            return pred_seq_out
        else:
            prior_m_seq = torch.stack(prior_m_seq, axis=1)
            prior_v_seq = torch.stack(prior_v_seq, axis=1)
            post_m_seq = torch.stack(post_m_seq, axis=1)
            post_v_seq = torch.stack(post_v_seq, axis=1)

            return (prior_m_seq, prior_v_seq), (post_m_seq, post_v_seq)

    def infer(self, x_past, x_t):
        '''
        Inference (compute prior and posterior distribution of z) for a batch of single steps.
        NOTE: must do processing before passing in to ensure correct format that this function expects.
        
        Input:
        - x_past (B x steps_in x D)
        - x_t    (B x steps_out x D)

        Returns:
        - prior_distrib (mu, var)
        - posterior_distrib (mu, var)
        '''

        B, _, D = x_past.size()
        past_in = x_past.reshape((B, -1))
        t_in = x_t.reshape((B, -1))

        prior_z, posterior_z = self.infer_step(past_in, t_in)

        return prior_z, posterior_z

    def infer_step(self, past_in, t_in):
        '''
        single step that computes both prior and posterior for training. Samples from posterior
        '''
        B = past_in.size(0)
        # use past and future to encode latent transition
        qm, qv = self.posterior(past_in, t_in)

        # prior
        pm, pv = None, None
        if self.use_conditional_prior:
            # predict prior based on past
            pm, pv = self.prior(past_in)
        else:
            # use standard normal
            pm, pv = torch.zeros_like(qm), torch.ones_like(qv)

        return (pm, pv), (qm, qv)

    def dict2vec_input(self, hdict):
        B = hdict["trans"].shape[0]
        T = hdict["trans"].shape[1]
        hvec = []
        for k in self.data_names:
            hvec.append(hdict[k].reshape(B, T, -1))
        hvec = torch.cat(hvec, dim=2)
        return hvec


class MLP(nn.Module):
    def __init__(self,
                 layers=[3, 128, 128, 3],
                 nonlinearity=nn.GELU,
                 use_gn=True,
                 skip_input_idx=None):
        '''
        If skip_input_idx is not None, the input feature after idx skip_input_idx will be skip connected to every later of the MLP.
        '''
        super(MLP, self).__init__()

        in_size = layers[0]
        out_channels = layers[1:]

        # input layer
        layers = []
        layers.append(nn.Linear(in_size, out_channels[0]))
        skip_size = 0 if skip_input_idx is None else (in_size - skip_input_idx)
        # now the rest
        for layer_idx in range(1, len(out_channels)):
            fc_layer = nn.Linear(out_channels[layer_idx - 1] + skip_size,
                                 out_channels[layer_idx])
            if use_gn:
                bn_layer = nn.GroupNorm(16, out_channels[layer_idx - 1])
                layers.append(bn_layer)
            layers.extend([nonlinearity(), fc_layer])
        self.net = nn.ModuleList(layers)
        self.skip_input_idx = skip_input_idx

    def forward(self, x):
        '''
        B x D x * : batch norm done over dim D
        '''
        skip_in = None
        if self.skip_input_idx is not None:
            skip_in = x[:, self.skip_input_idx:]
        for i, layer in enumerate(self.net):
            if self.skip_input_idx is not None and i > 0 and isinstance(
                    layer, nn.Linear):
                x = torch.cat([x, skip_in], dim=1)
            x = layer(x)
        return x
