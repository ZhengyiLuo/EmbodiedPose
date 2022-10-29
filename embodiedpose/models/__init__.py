from .kin_net_humor_z import KinNetHumorZ
from .kin_net_humor import KinNetHumor
from .kin_net_humor_direct import KinNetHumorDirect
from .kin_net_humor_direct_opt import KinNetHumorDirectOpt
from .kin_net_humor_res import KinNetHumorRes
from .kin_net_uhm import KinNetUHM
from .kin_net_multi import KinNetMulti

model_dict = {
    "kin_net_humor_z": KinNetHumorZ,
    "kin_net_humor": KinNetHumor,
    "kin_net_humor_direct": KinNetHumorDirect,
    "kin_net_humor_direct_opt": KinNetHumorDirectOpt,
    "kin_net_humor_res": KinNetHumorRes,
    "kin_net_uhm": KinNetUHM,
    "kin_net_multi": KinNetMulti,
}

from .kin_policy_humor import KinPolicyHumor
from .kin_policy_humor_res import KinPolicyHumorRes
from .kin_policy_humor_z import KinPolicyHumorZ
from .kin_policy_uhm import KinPolicyUHM
from .kin_policy_multi import KinPolicyMulti

policy_dict = {
    "kin_policy_humor": KinPolicyHumor,
    "kin_policy_humor_z": KinPolicyHumorZ,
    "kin_policy_uhm": KinPolicyUHM,
    "kin_policy_humor_res": KinPolicyHumorRes, 
    "kin_policy_multi": KinPolicyMulti,
}
