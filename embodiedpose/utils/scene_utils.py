import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import torch
import numpy as np
from embodiedpose.models.implicit_sdfs import SphereSDF_F, BoxSDF_F, TorusSDF_F

def get_sdf(scene_sdfs, points, topk = 1):
    points = points.view(-1, 3)
    if len(scene_sdfs) > 0:
        dists = []
        for sdf in scene_sdfs:
            dist = sdf(points)
            dists.append(dist)

        dists = torch.cat(dists, dim=1)

        if len(scene_sdfs) < topk:
            vals, locs = torch.topk(dists, k=len(scene_sdfs), largest=False)
            vals = torch.cat([
                vals,
                torch.ones([vals.shape[0], topk - len(scene_sdfs)]) *
                100
            ], dim = 1)
        else:
            vals, locs = torch.topk(dists, k = topk, largest= False)

    else:
        vals = torch.ones([points.shape[0], topk]) * 100

    return vals

def load_simple_scene(scene_name):
    cwd = os.getcwd()
    filename = f'{cwd}/data/scenes/{scene_name}_planes.txt'
    scene_sdfs = []
    obj_pos = []

    if os.path.exists(filename):
        planes = np.loadtxt(filename)
        planes = planes.reshape(-1, 4, 3)
        for plane in planes:
            pos, size, xyaxes = get_scene_attrs_from_plane(plane)
            xy = np.array([float(i) for i in xyaxes.split(' ')]).reshape(2, 3).T
            xyz = np.hstack([xy, np.cross(xy[:, 0], xy[:, 1])[:, None]])
            sides = [float(i) * 2 for i in size.split(" ")]
            pos = [float(i) for i in pos.split(" ")]
            obj_pos.append(pos)

            scene_sdfs.append(
                BoxSDF_F(trans=pos,
                            orientation=xyz,
                            side_lengths=sides))


    filename = f'{cwd}/data/scenes/{scene_name}_rectangles.txt'
    if os.path.exists(filename):
        rectangles = np.loadtxt(filename)
        rectangles = rectangles.reshape(-1, 8, 3)
        for rectangle in rectangles:
            pos, size, xyaxes = get_scene_attrs_from_rectangle(
                rectangle)
            xy = np.array([float(i) for i in xyaxes.split(' ')]).reshape(2, 3).T
            xyz = np.hstack([xy, np.cross(xy[:, 0], xy[:, 1])[:, None]])
            sides = [float(i) * 2 for i in size.split(" ")]
            pos = [float(i) for i in pos.split(" ")]
            obj_pos.append(pos)
            scene_sdfs.append(
                BoxSDF_F(trans=pos, orientation = xyz, side_lengths = sides))

    return scene_sdfs, obj_pos

def get_scene_attrs_from_plane(plane):
        c, x, y, w, h = get_cxy(plane)
        pos = f'{c[0]:04f} {c[1]:04f} {c[2]:04f}'
        size = f'{w/2:04f} {h/2:04f} 0.01'
        xaxis = f'{x[0]:04f} {x[1]:04f} {x[2]:04f}'
        yaxis = f'{y[0]:04f} {y[1]:04f} {y[2]:04f}'
        xyaxes = xaxis + ' ' + yaxis
        return pos, size, xyaxes

def get_scene_attrs_from_rectangle( rectangle):
    ind = np.argsort(rectangle[:, 2])
    rectangle = rectangle[ind]
    c, x, y, w, h = get_cxy(rectangle[:4])
    c[2] = (c[2] + rectangle[5, 2]) / 2
    pos = f'{c[0]:04f} {c[1]:04f} {c[2]:04f}'
    size = f'{w/2:04f} {h/2:04f} {c[2]:04f}'
    xaxis = f'{x[0]:04f} {x[1]:04f} {x[2]:04f}'
    yaxis = f'{y[0]:04f} {y[1]:04f} {y[2]:04f}'
    xyaxes = xaxis + ' ' + yaxis
    return pos, size, xyaxes

def get_cxy( points, c=None):
    if c is None:
        c = np.mean(points, axis=0)
    vec1 = points[0, :] - c
    vec2 = points[1, :] - c
    vec3 = points[2, :] - c
    x = vec1 + vec2
    y = vec2 + vec3
    w = np.linalg.norm(x)
    h = np.linalg.norm(y)
    x = x / (w + 1e-6)
    y = y / (h + 1e-6)
    if w < 0.01:
        _, x, y, w, h = get_cxy(points[[0, 2, 1]], c)
    if h < 0.01:
        _, x, y, w, h = get_cxy(points[[1, 0, 2]], c)
    return c, x, y, w, h
