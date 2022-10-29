from embodiedpose.agents.agent_scene import AgentScene
from embodiedpose.agents.agent_scene_v2 import AgentSceneV2
from embodiedpose.agents.agent_scene_sliding import AgentSceneSliding
from embodiedpose.agents.agent_scene_sliding_z import AgentSceneSlidingZ
from embodiedpose.agents.agent_scene_direct_opt import AgentSceneDirectOpt
from embodiedpose.agents.agent_embodied_pose import AgentScenePretrain
from embodiedpose.agents.agent_uhm_v1 import AgentUHMV1
from embodiedpose.agents.agent_multi import AgentMulti

agent_dict = {
    "scene_v1": AgentScene,
    "scene_v2": AgentSceneV2,
    "scene_direct_opt": AgentSceneDirectOpt,
    "scene_sliding": AgentSceneSliding,
    "scene_sliding_z": AgentSceneSlidingZ,
    "scene_pretrain": AgentScenePretrain,
    "uhm_v1": AgentUHMV1,
    "scene_multi": AgentMulti,
}
