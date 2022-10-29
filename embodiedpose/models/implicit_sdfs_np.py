from matplotlib.pyplot import axis
import autograd.numpy as np
from autograd import grad
from autograd import elementwise_grad as egrad

# # Sphere SDF class
class SphereSDF_N():
    def __init__(self, center=[0, 0, 0], radius=1.0):
        super().__init__()

        self.center = center.unsqueeze(0)
        self.radius = radius
        self.egrad_fun = egrad(self.compute_grad)


    def forward(self, points):
        points = points.reshape(-1, 3)

        return np.linalg.norm(points - self.center, dim=-1,
                                 keepdim=True) - self.radius

    def compute_grad(self, points):
        dist = self.forward(points)
        return (dist - 0).sum()



# Box SDF class
class BoxSDF_N():
    def __init__(self,
                 orientation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 trans=[0, 0, 0],
                 side_lengths=[1.75, 1.75, 1.75],
                 ):
        super().__init__()

        self.center = trans[None, ]
        self.rotmat = orientation
        self.side_lengths = side_lengths[None, ]
        self.egrad_fun = egrad(self.compute_grad)

    def forward(self, points):
        points = points.reshape(-1, 3)

        point_local = np.matmul((points - self.center), self.rotmat)
        diff = np.abs(point_local) - self.side_lengths / 2.0

        signed_distance = np.linalg.norm(np.maximum(diff, np.zeros_like(diff)), axis=-1) + np.minimum(np.max(diff, axis=-1), np.zeros_like(diff[..., 0]))
        return signed_distance[:, None]

    def compute_grad(self, points):
        dist = self.forward(points)
        return (dist - 0).sum()


# Torus SDF class
class CylinderSDF_N():
    def __init__(self, orientation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], trans=[0, 0, 0], size = [0.03, 0.35]): # radius, height
        super().__init__()

        self.center = trans[None, ]
        self.rotmat = orientation
        self.size = size
        self.egrad_fun = egrad(self.compute_grad)

    def forward(self, points):
        #  vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r);
        # return min(max(d.x,d.y),0.0) + length(max(d,0.0));

        points = points.reshape(-1, 3)
        point_local = np.matmul((points - self.center), self.rotmat)
        diff = np.abs(np.concatenate([np.linalg.norm(point_local[:, [0, 2]], axis = 1)[:, None], point_local[:, 1:2]], axis = 1)) - self.size
        signed_distance = np.minimum(np.maximum(diff[:, 0], diff[:, 1]), np.zeros_like(diff[:, 0])) + np.linalg.norm(
                                     np.maximum(diff, np.zeros_like(diff)), axis=-1)

        return signed_distance[:, None]

    def compute_grad(self, points):
        dist = self.forward(points)
        return (dist - 0).sum()

# Torus SDF class
class TorusSDF_N():
    def __init__(self, center=[0, 0, 0], radii=[1.0, 0.25]):
        super().__init__()

        self.center = np.nn.Parameter(
            np.tensor(center).float().unsqueeze(0), requires_grad=False)
        self.radii = np.nn.Parameter(
            np.tensor(radii).float().unsqueeze(0), requires_grad=False)

    def forward(self, points):
        points = points.reshape(-1, 3)
        diff = points - self.center
        q = np.stack([
            np.linalg.norm(diff[..., :2], dim=-1) - self.radii[..., 0],
            diff[..., -1],
        ],
                        dim=-1)
        return (np.linalg.norm(q, dim=-1) -
                self.radii[..., 1]).unsqueeze(-1)
