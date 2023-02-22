from uhc.utils.config_utils.base_config import Base_Config
from uhc.utils.config_utils.copycat_config import Config as CC_Config


class Config(Base_Config):

    def __init__(self, **kwargs):
        # training config
        super().__init__(**kwargs)

        self.agent_name = self.cfg_dict.get("agent_name", "scene_v1")
        self.env_name = self.cfg_dict.get("env_name", "kin_v1")

        self.policy_optimizer = self.cfg_dict['policy_optimizer']
        self.scene_specs = self.cfg_dict.get("scene_specs", {})
        self.policy_specs = self.cfg_dict.get("policy_specs", {})

        ## Model Specs
        self.autoregressive = self.model_specs.get("autoregressive", True)
        self.remove_base = self.model_specs.get("remove_base", True)

        # Policy Specs
        self.policy_name = self.policy_specs.get("policy_name", "kin_net_humor")
        self.reward_weights = self.policy_specs.get("reward_weights", {})
        self.env_term_body = self.policy_specs.get("env_term_body", "body")
        self.env_episode_len = self.policy_specs.get("env_episode_len", "body")

        self.model_name = self.model_specs.get("model_name", "kin_net")
        ## Data Specs
        self.fr_num = self.data_specs.get("fr_num", 80)
        self.cc_cfg = self.cfg_dict.get("cc_cfg", "uhc_explicit")
        self.cc_cfg = CC_Config(cfg_id=self.cc_cfg, base_dir="UniversalHumanoidControl/")
