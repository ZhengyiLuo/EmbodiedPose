from math import degrees
import os
import sys
import time
import argparse
import torch
import pdb
import os.path as osp
import glob

sys.path.append(os.getcwd())

from stl import mesh
import numpy as np
import mujoco_py
from mujoco_py import load_model_from_path, load_model_from_xml, MjSim, MjViewer

from uhc.smpllib.smpl_robot import Robot
from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from uhc.utils.config_utils.copycat_config import Config as CC_Config
from copy import deepcopy
import shutil
from lxml import etree
from scipy.spatial.transform import Rotation as sRot

import numpy as np
from copy import deepcopy
from collections import defaultdict
from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from lxml import etree
from io import BytesIO
from uhc.utils.config_utils.copycat_config import Config
from mujoco_py import load_model_from_path, load_model_from_xml, MjSim, MjViewer
from uhc.khrylib.mocap.skeleton import Skeleton
from uhc.khrylib.mocap.skeleton_mesh import Skeleton as SkeletonMesh
from uhc.khrylib.mocap.skeleton_mesh_v2 import Skeleton as SkeletonMeshV2
from uhc.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)
from collections import defaultdict
from scipy.spatial import ConvexHull
from stl import mesh
from uhc.utils.geom import quadric_mesh_decimation, center_scale_mesh
import uuid
import atexit
import shutil
import joblib
# from scipy.spatial.qhull import _Qhull
from uhc.utils.flags import flags
import cv2
from uhc.smpllib.smpl_robot import get_joint_geometries, denormalize_range
import copy


def add_suffix(dictionary, suffix = "00", add_val = False):
    """
    Add a suffix to all keys in a dictionary.
    """
    new_dict = {}
    for key, value in dictionary.items():
        new_dict[key + "_" + suffix] = value + "_" + suffix if (
            add_val and (not value is None)) else value
    return new_dict


class Multi_Robot(Robot):

    def load_from_skeleton(
        self,
        betas=None,
        v_template=None,
        gender=[0],
        objs_info=None,
        obj_pose=None,
        params=None,
    ):
        self.remove_geoms()
        self.num_people = num_people = len(gender)
        self.tree = None  # xml tree
        self.bodies = []  ### Cleaning bodies list


        for p in range(num_people):
            curr_gender = gender[p]
            if curr_gender == 0:
                self.smpl_parser = smpl_parser = self.smpl_parser_n
            elif curr_gender == 1:
                self.smpl_parser = smpl_parser = self.smpl_parser_m
            elif curr_gender == 2:
                self.smpl_parser = smpl_parser = self.smpl_parser_f
            else:
                print(gender)
                raise Exception("Gender Not Supported!!")

            if betas is None and self.beta is None:
                betas = (torch.zeros(
                    (1, 10)).float() if self.smpl_model == "smpl" else torch.zeros(
                        (1, 16)).float())
            else:
                if params is None:
                    self.beta = betas[p:(p+1)] if not betas is None else self.beta
                else:
                    # If params is not none, we need to set the beta first
                    betas = self.map_params(betas)
                    self.beta = torch.from_numpy(
                        denormalize_range(
                            betas.numpy().squeeze(),
                            self.param_specs["beta"]["lb"],
                            self.param_specs["beta"]["ub"],
                        )[None, ])
            if flags.debug:
                print(self.beta)

            ## Clear up beta for smpl and smplh
            if self.smpl_model == "smpl" and self.beta.shape[1] == 16:
                self.beta = self.beta[:, :10]
                # print(f"Incorrect shape size for {self.model}!!!")
            elif self.smpl_model == "smplh" and self.beta.shape[1] == 10:
                self.beta = torch.hstack([self.beta, torch.zeros((1, 6)).float()])
                # print(f"Incorrect shape size for {self.model}!!!")

            if self.mesh:
                self.model_dirs.append(f"/tmp/smpl/{uuid.uuid4()}")

                if self.cfg.get("ball", False):
                    self.skeleton = SkeletonMeshV2(self.model_dirs[-1])
                else:
                    self.skeleton = SkeletonMesh(self.model_dirs[-1])

                (
                    verts,
                    joints,
                    skin_weights,
                    joint_names,
                    joint_offsets,
                    joint_parents,
                    joint_axes,
                    joint_dofs,
                    joint_range,
                    contype,
                    conaffinity,
                ) = (smpl_parser.get_mesh_offsets(betas=self.beta[0:1], flatfoot=self.flatfoot) if
                    self.smpl_model != "smplx" else smpl_parser.get_mesh_offsets(
                        v_template=v_template))

                if self.rel_joint_lm:
                    joint_range["L_Knee"][0] = np.array([-np.pi / 16, np.pi / 16])
                    joint_range["L_Knee"][1] = np.array([-np.pi / 16, np.pi / 16])
                    joint_range["L_Knee"][2] = np.array([-np.pi / 16, np.pi])

                    joint_range["R_Knee"][0] = np.array([-np.pi / 16, np.pi / 16])
                    joint_range["R_Knee"][1] = np.array([-np.pi / 16, np.pi / 16])
                    joint_range["R_Knee"][2] = np.array([-np.pi / 16, np.pi])

                    joint_range["L_Ankle"][0] = np.array([-np.pi / 2, np.pi / 2])
                    joint_range["L_Ankle"][1] = np.array([-np.pi / 2, np.pi / 2])
                    joint_range["L_Ankle"][2] = np.array([-np.pi / 2, np.pi / 2])

                    joint_range["R_Ankle"][0] = np.array([-np.pi / 2, np.pi / 2])
                    joint_range["R_Ankle"][1] = np.array([-np.pi / 2, np.pi / 2])
                    joint_range["R_Ankle"][2] = np.array([-np.pi / 2, np.pi / 2])

                    joint_range["L_Toe"][0] = np.array([-np.pi / 4, np.pi / 4])
                    joint_range["L_Toe"][1] = np.array([-np.pi / 4, np.pi / 4])
                    joint_range["L_Toe"][2] = np.array([-np.pi / 2, np.pi / 2])

                    joint_range["R_Toe"][0] = np.array([-np.pi / 4, np.pi / 4])
                    joint_range["R_Toe"][1] = np.array([-np.pi / 4, np.pi / 4])
                    joint_range["R_Toe"][2] = np.array([-np.pi / 2, np.pi / 2])

                self.height = np.max(verts[:, 1]) - np.min(verts[:, 1])

                size_dict = {}

                if (len(self.get_params(get_name=True)) > 1
                        and not params is None):  # ZL: dank code, very dank code
                    self.set_params(params)
                    size_dict = self.get_size()
                    size_dict = self.enforce_length_size(size_dict)

                self.hull_dict = get_joint_geometries(
                    verts,
                    joints,
                    skin_weights,
                    joint_names,
                    scale_dict=size_dict,
                    suffix = f"{p:02d}",
                    geom_dir=f"{self.model_dirs[-1]}/geom", min_num_vert=30)

                joint_offsets = add_suffix(joint_offsets,f"{p:02d}")
                joint_parents = add_suffix(joint_parents,f"{p:02d}", add_val = True)
                joint_axes = add_suffix(joint_axes,f"{p:02d}")
                joint_dofs = add_suffix(joint_dofs,f"{p:02d}")
                joint_range = add_suffix(joint_range, f"{p:02d}")

                coaffin = 1 << p
                conaffinity = {
                    coaffin : [j + "_" + f"{p:02d}" for j in v]
                    for k, v in conaffinity.items()
                }
                contype = {
                    ~coaffin: [j + "_" + f"{p:02d}" for j in v]
                    for k, v in contype.items()
                }

                # print(conaffinity, contype)
                if p == 0:
                    rand_color =  np.array([0.8, 0.6, 0.4])
                else:
                    rand_color = np.array([.98, .54, .56])

                color_dict = {
                    k: f"{rand_color[0]:.3f} {rand_color[1]:.3f} {rand_color[2]:.3f} 1" for k in joint_offsets.keys()
                }

                self.skeleton.load_from_offsets(
                    joint_offsets,
                    joint_parents,
                    joint_axes,
                    joint_dofs,
                    joint_range,
                    sites={},
                    scale=1,
                    equalities={},
                    collision_groups=contype,
                    conaffinity=conaffinity,
                    simple_geom=False,
                    color_dict=color_dict
                )

            else:
                self.skeleton = Skeleton()
                offset_smpl_dict, parents_dict, channels = smpl_parser.get_offsets(
                    betas=self.beta)

                self.skeleton.load_from_offsets(offset_smpl_dict, parents_dict, 1,
                                                {}, channels, {})

            self.bone_length = np.array(
                [np.linalg.norm(i) for i in joint_offsets.values()])
            parser = XMLParser(remove_blank_text=True)
            curr_tree = parse(
                BytesIO(self.skeleton.write_str(bump_buffer=False)),
                parser=parser,
            )



            if p == 0:
                self.tree = curr_tree  # First tree is the base
                self.local_coord = (self.tree.getroot().find(
                    ".//compiler").attrib["coordinate"] == "local")
                root = curr_tree.getroot().find("worldbody").find("body")
                # self.add_body(root, None) # Do not need this for now
            else:
                main_root = self.tree.getroot()
                main_body = main_root.find("worldbody")
                main_asset = main_root.find("asset")
                main_motor = main_root.find("actuator")
                new_root = curr_tree.getroot()
                new_body = new_root.find("worldbody").find("body")
                new_asset = new_root.find("asset")

                main_body.append(new_body)
                for mesh in new_asset.findall("mesh"):
                    main_asset.append(mesh)
                for motor in new_root.find("actuator").findall("motor"):
                    main_motor.append(motor)
                # self.add_body(new_body, None)

        self.init_tree = copy.deepcopy(self.tree)
        SubElement(self.tree.getroot(), "size", {"njmax": "10000", "nconmax": "5000"})

        self.init_bodies()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--cfg", default="copycat_ball_1")
    parser.add_argument("--cfg", default="copycat_40")
    args = parser.parse_args()

    cc_cfg = CC_Config(cfg_id="copycat_e_1",
                       base_dir="/hdd/zen/dev/copycat/Copycat/")

    smpl_robot = Multi_Robot(
        cc_cfg.robot_cfg,
        data_dir=osp.join(cc_cfg.base_dir, "data/smpl"),
        masterfoot=cc_cfg.masterfoot,
    )

    smpl_robot.load_from_skeleton(torch.zeros((5, 16)),
                                  gender=[0] * 5)
    smpl_robot.write_xml("test.xml")
    model = mujoco_py.load_model_from_xml(
        smpl_robot.export_xml_string().decode("utf-8"))

    print(f"mass {mujoco_py.functions.mj_getTotalmass(model)}")
    sim = MjSim(model)

    viewer = MjViewer(sim)

    while True:
        import ipdb
        ipdb.set_trace()

        # sim.data.qpos[54] = np.pi / 2
        # sim.data.qpos[1] -= 0.005
        # sim.data.qpos[76:] = grab_data[key]["obj_pose"][i % seq_len]

        # sim.forward()
        viewer.render()
