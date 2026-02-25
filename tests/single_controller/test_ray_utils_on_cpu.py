

import pytest
import ray

from verl.utils.ray_utils import parallel_put

@pytest.fixture()
def init_ray():
    ray.init(num_cpus=4)
    yield
    ray.shutdown()

def test_parallel_put_basic(init_ray):
    data = [1, "hello", {"a": 2}, [3, 4]]
    refs = parallel_put(data)
    assert len(refs) == len(data)
    retrieved_data = [ray.get(ref) for ref in refs]
    assert retrieved_data == data

def test_parallel_put_empty(init_ray):
    data = []
    with pytest.raises(AssertionError):
        _ = parallel_put(data)

def test_parallel_put_workers(init_ray):
    data = list(range(20))

    refs = parallel_put(data, max_workers=4)
    assert len(refs) == len(data)
    retrieved_data = [ray.get(ref) for ref in refs]
    assert retrieved_data == data

    refs_default = parallel_put(data)
    assert len(refs_default) == len(data)
    retrieved_data_default = [ray.get(ref) for ref in refs_default]
    assert retrieved_data_default == data
