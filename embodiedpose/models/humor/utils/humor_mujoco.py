from uhc.smpllib.smpl_parser import SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES
from uhc.khrylib.utils import get_body_qposaddr
import numpy as np

MUJOCO_2_SMPL = np.array([0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 14, 19, 13, 15, 20, 16, 21, 17, 22, 18, 23])
SMPL_2_OP = np.array([False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False])
OP_14_to_OP_12 = np.array([True, True, True, True, True, True, True, True, False, True, True, False, True, True])


def reorder_joints_to_humor(joints, mj_model, model="smpl"):
    if model == "smpl":
        joint_names = SMPL_BONE_ORDER_NAMES
    elif model == "smplh":
        joint_names = SMPLH_BONE_ORDER_NAMES

    mujoco_joint_names = list(get_body_qposaddr(mj_model).keys())
    mujoco_2_smpl = [mujoco_joint_names.index(q) for q in joint_names if q in mujoco_joint_names]
    ordered_joints = joints.reshape(-1, 24, 3)[:, mujoco_2_smpl].reshape(-1, 72)
    return ordered_joints
