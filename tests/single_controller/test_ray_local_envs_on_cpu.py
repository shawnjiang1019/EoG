

import os

import ray

from verl.single_controller.base.worker import Worker
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

@ray.remote
class TestActor(Worker):
    def __init__(self) -> None:
        super().__init__()

    def getenv(self, key):
        val = os.getenv(key, f"{key} not set")
        return val

def test_basics():
    ray.init(num_cpus=100)

    resource_pool = RayResourcePool([4], use_gpu=False)
    class_with_args = RayClassWithInitArgs(cls=TestActor)

    worker_group = RayWorkerGroup(
        resource_pool=resource_pool, ray_cls_with_init=class_with_args, name_prefix="worker_group_basic"
    )

    output = worker_group.execute_all_sync("getenv", key="RAY_LOCAL_WORLD_SIZE")
    assert output == ["4", "4", "4", "4"]

    output = worker_group.execute_all_sync("getenv", key="RAY_LOCAL_RANK")
    assert set(output) == set(["0", "1", "2", "3"])

    ray.shutdown()

if __name__ == "__main__":
    test_basics()
