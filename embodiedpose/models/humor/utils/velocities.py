import torch
from embodiedpose.models.humor.utils.transforms import rotation_matrix_to_angle_axis, batch_rodrigues


def estimate_velocities(trans, root_orient, joints3d, data_fps, aa_to_mat = True):
    '''
    From the SMPL sequence, estimates velocity inputs to the motion prior.

    - trans : root translation
    - root_orient : aa root orientation
    - body_pose
    '''
    B, T, _ = trans.size()
    h = 1.0 / data_fps
    trans_vel = estimate_linear_velocity(trans, h)
    joints_vel = estimate_linear_velocity(joints3d, h)
    if aa_to_mat:
        root_orient = batch_rodrigues(root_orient.reshape((-1, 3))).reshape((B, T, 3, 3))

    if root_orient.shape[-1] != 3: root_orient = root_orient.reshape((B, T, 3, 3))
    root_orient_vel = estimate_angular_velocity(root_orient, h)
    return trans_vel, joints_vel, root_orient_vel


def estimate_linear_velocity(data_seq, h):
    '''
    Given some batched data sequences of T timesteps in the shape (B, T, ...), estimates
    the velocity for the middle T-2 steps using a second order central difference scheme.
    The first and last frames are with forward and backward first-order 
    differences, respectively
    - h : step size
    '''
    # first steps is forward diff (t+1 - t) / h
    init_vel = (data_seq[:, 1:2] - data_seq[:, :1]) / h
    # middle steps are second order (t+1 - t-1) / 2h
    middle_vel = (data_seq[:, 2:] - data_seq[:, 0:-2]) / (2 * h)
    # last step is backward diff (t - t-1) / h
    final_vel = (data_seq[:, -1:] - data_seq[:, -2:-1]) / h

    vel_seq = torch.cat([init_vel, middle_vel, final_vel], dim=1)
    return vel_seq


def estimate_angular_velocity(rot_seq, h):
    '''
    Given a batch of sequences of T rotation matrices, estimates angular velocity at T-2 steps.
    Input sequence should be of shape (B, T, ..., 3, 3)
    '''
    # see https://en.wikipedia.org/wiki/Angular_velocity#Calculation_from_the_orientation_matrix
    dRdt = estimate_linear_velocity(rot_seq, h)
    R = rot_seq
    RT = R.transpose(-1, -2)
    # compute skew-symmetric angular velocity tensor
    w_mat = torch.matmul(dRdt, RT)
    # pull out angular velocity vector
    # average symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = torch.stack([w_x, w_y, w_z], axis=-1)
    return w


def estimate_single_angular_velocity(rot1, rot2, h):
    '''
    Given a sequence of 2 rotation matrices,
    estimates angular velocity at the first step.
    '''
    dRdt = (rot1 - rot2) / h
    R = rot1
    RT = R.transpose(-1, -2)
    # compute skew-symmetric angular velocity tensor
    w_mat = torch.matmul(dRdt, RT)
    # pull out angular velocity vector
    # average symmetric entries
    w_x = (-w_mat[1, 2] + w_mat[2, 1]) / 2.0
    w_y = (w_mat[0, 2] - w_mat[2, 0]) / 2.0
    w_z = (-w_mat[0, 1] + w_mat[1, 0]) / 2.0
    w = torch.stack([w_x, w_y, w_z], axis=-1)
    return w
