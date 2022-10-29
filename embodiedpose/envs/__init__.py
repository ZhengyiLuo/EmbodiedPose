from .humanoid_kin_z import HumanoidKinEnvZ
from .humanoid_kin_direct import HumanoidKinEnvDirect
from .humanoid_kin_res import HumanoidKinEnvRes
from .humanoid_kin_direct_opt import HumanoidKinEnvDirectOpt
from .humanoid_kin_uhm import HumanoidKinEnvUHM
from .humanoid_kin_multi import HumanoidKinEnvMulti

env_dict = {
    "kin_z": HumanoidKinEnvZ,
    "kin_direct": HumanoidKinEnvDirect,
    "kin_direct_opt": HumanoidKinEnvDirectOpt,
    "kin_res": HumanoidKinEnvRes,
    "kin_uhm": HumanoidKinEnvUHM,
    "kin_multi": HumanoidKinEnvMulti,
}
