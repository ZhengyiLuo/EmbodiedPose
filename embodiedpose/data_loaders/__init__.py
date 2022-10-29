from embodiedpose.data_loaders.amass_dataset import AMASSDataset
from embodiedpose.data_loaders.video_pose_dataset import VideoPoseDataset
from embodiedpose.data_loaders.scene_pose_dataset import ScenePoseDataset
from embodiedpose.data_loaders.amass_multi_dataset import AMASSDatasetMulti

data_dict = {
    "scene_pose": ScenePoseDataset,
    "video_pose": VideoPoseDataset,
    "amass": AMASSDataset,
    "amass_multi": AMASSDatasetMulti
}
