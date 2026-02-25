

from typing import Callable

import torch

from verl import DataProto

from ..base import BaseEngine, EngineRegistry

@EngineRegistry.register("megatron")
class MegatronEngine(BaseEngine):
    def __init__(self, config):
        raise NotImplementedError

    def init_model(self):
        raise NotImplementedError

    def train_mode(self):
        raise NotImplementedError

    def eval_mode(self):
        raise NotImplementedError

    def infer_batch(
        self,
        data: DataProto,
        post_fn: Callable[[DataProto, torch.Tensor], tuple[torch.Tensor, dict[str, torch.Tensor]]],
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def train_batch(
        self,
        data: DataProto,
        loss_fn: Callable[[DataProto, torch.Tensor], tuple[torch.Tensor, dict[str, torch.Tensor]]],
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def optimizer_zero_grad(self):
        raise NotImplementedError

    def optimizer_step(self):
        raise NotImplementedError

    def lr_scheduler_step(self):
        raise NotImplementedError

    def shard_data(self, data):
        raise NotImplementedError

    def unshard_data(self, data):
        raise NotImplementedError

    def to(self, device: str, model: bool = True, optimizer: bool = True):
        raise NotImplementedError

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        raise NotImplementedError

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        raise NotImplementedError
