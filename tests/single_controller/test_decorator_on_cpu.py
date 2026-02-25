

import asyncio
import time

import pytest
import ray
import torch
from tensordict import TensorDict

from verl.protocol import DataProto, DataProtoFuture
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.base.worker import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

@pytest.fixture
def ray_init_shutdown():
    ray.init(num_cpus=100)
    yield
    ray.shutdown()

@ray.remote
class DecoratorTestWorker(Worker):
    def __init__(self, initial_value=0):
        super().__init__()
        self.value = initial_value

        time.sleep(0.1)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def dp_compute(self, data: DataProto) -> DataProto:
        time.sleep(0.1)
        rank_value = torch.tensor(self.rank, device=data.batch["input"].device, dtype=data.batch["input"].dtype)
        data.batch["output"] = data.batch["input"] + self.value + rank_value
        return data

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
    async def async_dp_compute(self, data: DataProto) -> DataProto:

        await asyncio.sleep(0.1)
        rank_value = torch.tensor(self.rank, device=data.batch["input"].device, dtype=data.batch["input"].dtype)
        data.batch["output_async"] = data.batch["input"] * 2 + self.value + rank_value
        return data

def test_decorator_dp_compute(ray_init_shutdown):
    num_workers = 2
    resource_pool = RayResourcePool([num_workers], use_gpu=False, max_colocate_count=1)
    cls_with_args = RayClassWithInitArgs(cls=DecoratorTestWorker, initial_value=10)
    worker_group = RayWorkerGroup(
        resource_pool, cls_with_args, name_prefix=f"decorator_test_sync_dp_{int(time.time())}"
    )

    input_tensor = torch.arange(4, dtype=torch.float32)
    data = DataProto(batch=TensorDict({"input": input_tensor}, batch_size=[4]))

    output = worker_group.dp_compute(data)

    assert isinstance(output, DataProto), "Expected DataProto result"
    assert "output" in output.batch.keys()
    assert len(output) == len(data), "Output length should match input length"

    expected_output_part1 = torch.tensor([0, 1], dtype=torch.float32) + 10 + 0
    expected_output_part2 = torch.tensor([2, 3], dtype=torch.float32) + 10 + 1
    expected_output = torch.cat([expected_output_part1, expected_output_part2])

    torch.testing.assert_close(output.batch["output"], expected_output, msg="Sync DP compute output data mismatch")

def test_decorator_async_function(ray_init_shutdown):
    num_workers = 2
    resource_pool = RayResourcePool([num_workers], use_gpu=False, max_colocate_count=1)
    cls_with_args = RayClassWithInitArgs(cls=DecoratorTestWorker, initial_value=5)
    worker_group = RayWorkerGroup(
        resource_pool, cls_with_args, name_prefix=f"decorator_test_async_dp_{int(time.time())}"
    )

    input_tensor = torch.arange(4, dtype=torch.float32)
    data = DataProto(batch=TensorDict({"input": input_tensor}, batch_size=[4]))

    future_output: DataProtoFuture = worker_group.async_dp_compute(data)

    assert isinstance(future_output, DataProtoFuture), "Expected DataProtoFuture for async def call"

    result_data = future_output.get()

    assert isinstance(result_data, DataProto)
    assert "output_async" in result_data.batch.keys()
    assert len(result_data) == len(data), "Output length should match input length"

    expected_output_part1 = (torch.tensor([0, 1], dtype=torch.float32) * 2) + 5 + 0
    expected_output_part2 = (torch.tensor([2, 3], dtype=torch.float32) * 2) + 5 + 1
    expected_output = torch.cat([expected_output_part1, expected_output_part2])

    torch.testing.assert_close(
        result_data.batch["output_async"], expected_output, msg="Async DP compute output data mismatch"
    )
