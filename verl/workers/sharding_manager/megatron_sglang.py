

import asyncio
import logging
import os

import torch.distributed as dist
from omegaconf import DictConfig
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from sglang.srt.utils import MultiprocessingSerializer
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from verl.protocol import DataProto, all_gather_data_proto
from verl.utils.device import get_torch_device
from verl.utils.megatron_utils import (
    load_megatron_model_to_gpu,
    offload_megatron_model_to_cpu,
    per_tensor_generator,
)
from verl.utils.profiler import GPUMemoryLogger, log_gpu_memory_usage, simple_timer
from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))



class MegatronSGLangShardingManager(BaseShardingManager):

    def __init__(
        self,
        actor_module: nn.ModuleList,
        inference_engine: Engine,
        model_config: DictConfig,
        rollout_config: DictConfig,
        transformer_config,
        layer_name_mapping,
        weight_converter,
        device_mesh: DeviceMesh | None = None,
        offload_param: bool = False,
        bridge=None,
    ):
        self.actor_module = actor_module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.rollout_config = rollout_config
        self.transformer_config = transformer_config
        self.layer_name_mapping = layer_name_mapping
        self.weight_converter = weight_converter
        self.device_mesh = device_mesh
        self.bridge = bridge
        self.offload_param = offload_param

        if self.device_mesh is not None:
            self.infer_tp_size = self.device_mesh["tp"].mesh.size()[0]
        else:
            self.infer_tp_size = self.inference_engine._tp_size

        self.torch_random_states = get_torch_device().get_rng_state()

        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            get_torch_device().manual_seed(gen_dp_rank + 1000)
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    @GPUMemoryLogger(role="MegatronSGLangShardingManager enter", logger=logger)
    def __enter__(self):
        self.timing = {}
        with simple_timer("reshard", self.timing):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.wake_up())

    @GPUMemoryLogger(role="MegatronSGLangShardingManager exit", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.sleep())

    async def update_weights(self, params):
        if self.device_mesh["tp"].get_local_rank() == 0 and self.rollout_config.free_cache_engine:
            await self.inference_engine.resume_memory_occupation()
        named_tensors = params
        load_format = None

        update_weights_bucket_bytes = int(self.rollout_config.update_weights_bucket_megabytes) << 20
        for batch in get_named_tensor_buckets(named_tensors, update_weights_bucket_bytes):

            named_tensors_batch = [
                (name, MultiprocessingSerializer.serialize(tensor.detach())) for name, tensor in batch
            ]

            if self.device_mesh["tp"].get_local_rank() == 0:

                gathered_serialized_batches = [None for _ in range(self.device_mesh["tp"].mesh.size()[0])]
            else:
                gathered_serialized_batches = None

            dist.gather_object(
                obj=named_tensors_batch,
                object_gather_list=gathered_serialized_batches,
                dst=self.device_mesh["tp"].mesh.tolist()[0],
                group=self.device_mesh["tp"].get_group(),
            )

            if self.device_mesh["tp"].get_local_rank() == 0:

                logical_tensors = zip(*gathered_serialized_batches, strict=False)
                await self.inference_engine.update_weights_from_tensor(
                    named_tensors=[

                        (
                            tensor_group[0][0],
                            LocalSerializedTensor(

                                values=[rank_part[1] for rank_part in tensor_group]
                            ),
                        )
                        for tensor_group in logical_tensors

                    ],
                    load_format=load_format,
                    flush_cache=False,
                )

        if self.device_mesh["tp"].get_local_rank() == 0:
            await self.inference_engine.flush_cache()

    async def release_memory(self):
        if self.device_mesh["tp"].get_local_rank() == 0 and self.rollout_config.free_cache_engine:
            await self.inference_engine.release_memory_occupation()

    @GPUMemoryLogger(role="MegatronSGLangShardingManager enter", logger=logger)
    async def wake_up(self):
        if self.offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        if self.bridge is not None:
            per_tensor_param = self.bridge.export_weights(self.actor_module)
        else:
            per_tensor_param = per_tensor_generator(
                self.actor_module,
                self.model_config,
                self.weight_converter,
                self.transformer_config,
                self.layer_name_mapping,
            )
        await self.update_weights(per_tensor_param)
        if self.offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        get_torch_device().empty_cache()

        if self.device_mesh is not None:
            self.torch_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="MegatronSGLangShardingManager exit", logger=logger)
    async def sleep(self):
        if self.rollout_config.free_cache_engine:
            log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
            await self.release_memory()
            log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)

        for model in self.actor_module:
            model.train()

        get_torch_device().empty_cache()

        if self.device_mesh is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)

    @GPUMemoryLogger(role="megatron sglang sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:

        if self.infer_tp_size == 1:
            return data
        all_gather_data_proto(data, self.device_mesh["tp"].get_group())
        return data

    @GPUMemoryLogger(role="megatron sglang sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:

        if self.infer_tp_size == 1:
            return data
        return data.chunk(chunks=self.infer_tp_size)[self.device_mesh["tp"].get_local_rank()]
