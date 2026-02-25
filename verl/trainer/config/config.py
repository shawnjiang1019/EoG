

from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

@dataclass
class CriticConfig(BaseConfig):

    _frozen_fields = [
        "rollout_n",
        "strategy",
        "use_dynamic_bsz",
        "ppo_max_token_len_per_gpu",
        "forward_max_token_len_per_gpu",
        "ppo_epochs",
        "shuffle",
        "cliprange_value",
        "loss_agg_mode",
    ]

    rollout_n: int = 1
    strategy: str = "fsdp"
    optim: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    ppo_mini_batch_size: int = 1
    ppo_micro_batch_size: Optional[int] = None
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 32768
    forward_max_token_len_per_gpu: int = 32768
    ppo_epochs: int = 1
    shuffle: bool = True
    cliprange_value: float = 0.5
    loss_agg_mode: str = "token-mean"
    checkpoint: dict[str, Any] = field(default_factory=dict)
    profiler: dict[str, Any] = field(default_factory=dict)

@dataclass
class MegatronCriticConfig(CriticConfig):

    _frozen_fields = CriticConfig._frozen_fields + [
        "nccl_timeout",
        "load_weight",
        "data_loader_seed",
    ]

    strategy: str = "megatron"
    nccl_timeout: int = 600
    megatron: dict[str, Any] = field(default_factory=dict)
    load_weight: bool = True
    data_loader_seed: Optional[int] = None

@dataclass
class FSDPCriticConfig(CriticConfig):

    _frozen_fields = CriticConfig._frozen_fields + [
        "ulysses_sequence_parallel_size",
        "grad_clip",
    ]

    strategy: str = "fsdp"
    forward_micro_batch_size: int = 1
    forward_micro_batch_size_per_gpu: int = 1
    ulysses_sequence_parallel_size: int = 1
    grad_clip: float = 1.0
