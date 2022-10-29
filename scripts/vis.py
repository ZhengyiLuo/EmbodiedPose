import joblib

j3d_pred, j3d_gt_root, joints2d = joblib.load("a.pkl")
j3d_pred, j3d_gt_root, joints2d = j3d_pred.detach().cpu().numpy(
), j3d_gt_root.detach().cpu().numpy(), j3d_gt_root.detach().cpu().numpy()

# j3d = j3d_gt_root[:, 0]
j3d = j3d_pred[:, 0]
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(90, -90)
ax.scatter(j3d[0, :, 0], j3d[0, :, 1], j3d[0, :, 2])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
drange = 1
ax.set_xlim(-drange, drange)
ax.set_ylim(drange, -drange)
ax.set_zlim(-drange, drange)
plt.show()
