from unittest import skip
import torch
import torch.nn.functional as F

from torch import autograd

# Sphere SDF class
class SphereSDF_F(torch.nn.Module):
    def __init__(self, center=[0, 0, 0], radius=1.0):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(center).float().unsqueeze(0), requires_grad=False)
        self.radius = torch.nn.Parameter(torch.tensor(radius).float(),
                                         requires_grad=False)

    def forward(self, points):
        points = points.view(-1, 3)

        return torch.linalg.norm(points - self.center, dim=-1,
                                 keepdim=True) - self.radius


# Box SDF class
class BoxSDF_F(torch.nn.Module):
    def __init__(self,
                 orientation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 trans=[0, 0, 0],
                 side_lengths=[1.75, 1.75, 1.75],
                 ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(trans).double().unsqueeze(0), requires_grad=True)
        self.rotmat = torch.nn.Parameter(torch.tensor(orientation).double(), requires_grad=True)
        self.side_lengths = torch.nn.Parameter(
            torch.tensor(side_lengths).double().unsqueeze(0),
            requires_grad=True)

    def forward(self, points):
        points = points.view(-1, 3)

        point_local = torch.matmul((points - self.center), self.rotmat)

        diff = torch.abs(point_local) - self.side_lengths / 2.0
        signed_distance = torch.linalg.norm(
            torch.maximum(
                diff, torch.zeros_like(diff)), dim=-1) + torch.minimum(
                    torch.max(diff, dim=-1)[0], torch.zeros_like(diff[..., 0]))
        return signed_distance.unsqueeze(-1)

# Torus SDF class
class CylinderSDF_F(torch.nn.Module):
    def __init__(self, orientation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], trans=[0, 0, 0], size = [0.03, 0.35]): # radius, height
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(trans).double().unsqueeze(0), requires_grad=True)
        self.rotmat = torch.nn.Parameter(torch.tensor(orientation).double(), requires_grad=True)
        self.size = torch.nn.Parameter(
            torch.tensor(size).double().unsqueeze(0),
            requires_grad=True)

    def forward(self, points):
        #  vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r);
        # return min(max(d.x,d.y),0.0) + length(max(d,0.0));

        points = points.view(-1, 3)
        point_local = torch.matmul((points - self.center), self.rotmat)
        diff = torch.abs(torch.cat([torch.norm(point_local[:, [0, 2]], dim = 1)[:, None], point_local[:, 1:2]], dim = 1)) - self.size
        signed_distance = torch.min(torch.max(diff[:, 0], diff[:, 1]), torch.zeros_like(diff[:, 0])) + torch.norm(torch.max(diff, torch.zeros_like(diff)), dim = -1)

        return signed_distance.unsqueeze(-1)


# Torus SDF class
class TorusSDF_F(torch.nn.Module):
    def __init__(self, center=[0, 0, 0], radii=[1.0, 0.25]):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(center).float().unsqueeze(0), requires_grad=False)
        self.radii = torch.nn.Parameter(
            torch.tensor(radii).float().unsqueeze(0), requires_grad=False)

    def forward(self, points):
        points = points.view(-1, 3)
        diff = points - self.center
        q = torch.stack([
            torch.linalg.norm(diff[..., :2], dim=-1) - self.radii[..., 0],
            diff[..., -1],
        ],
                        dim=-1)
        return (torch.linalg.norm(q, dim=-1) -
                self.radii[..., 1]).unsqueeze(-1)
