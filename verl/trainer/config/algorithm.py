

from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig

@dataclass
class KLControlConfig(BaseConfig):

    _frozen_fields = ["type", "kl_coef", "horizon", "target_kl"]
    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: int = 10000
    target_kl: float = 0.1

@dataclass
class PFPPOConfig(BaseConfig):

    _frozen_fields = ["reweight_method", "weight_pow"]
    reweight_method: str = "pow"
    weight_pow: float = 2.0

@dataclass
class FilterGroupsConfig(BaseConfig):

    _frozen_fields = ["enable", "metric", "max_num_gen_batches"]

    enable: bool = False
    metric: Optional[str] = None
    max_num_gen_batches: int = 0

@dataclass
class AlgoConfig(BaseConfig):

    _frozen_fields = [
        "gamma",
        "lam",
        "adv_estimator",
        "norm_adv_by_std_in_grpo",
        "use_kl_in_reward",
        "kl_penalty",
        "use_pf_ppo",
    ]

    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)
    use_pf_ppo: bool = False
    pf_ppo: Optional[PFPPOConfig] = None
    filter_groups: Optional[FilterGroupsConfig] = None
