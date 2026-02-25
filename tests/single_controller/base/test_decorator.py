

import pytest

import verl.single_controller.base.decorator as decorator_module
from verl.single_controller.base.decorator import (
    DISPATCH_MODE_FN_REGISTRY,
    Dispatch,
    _check_dispatch_mode,
    get_predefined_dispatch_fn,
    register_dispatch_mode,
    update_dispatch_mode,
)

@pytest.fixture
def reset_dispatch_registry():

    original_registry = DISPATCH_MODE_FN_REGISTRY.copy()
    yield

    decorator_module.DISPATCH_MODE_FN_REGISTRY.clear()
    decorator_module.DISPATCH_MODE_FN_REGISTRY.update(original_registry)

def test_register_new_dispatch_mode(reset_dispatch_registry):

    def dummy_dispatch(worker_group, *args, **kwargs):
        return args, kwargs

    def dummy_collect(worker_group, output):
        return output

    register_dispatch_mode("TEST_MODE", dummy_dispatch, dummy_collect)

    _check_dispatch_mode(Dispatch.TEST_MODE)

    assert get_predefined_dispatch_fn(Dispatch.TEST_MODE) == {
        "dispatch_fn": dummy_dispatch,
        "collect_fn": dummy_collect,
    }

    Dispatch.remove("TEST_MODE")

def test_update_existing_dispatch_mode(reset_dispatch_registry):

    original_mode = Dispatch.ONE_TO_ALL

    def new_dispatch(worker_group, *args, **kwargs):
        return args, kwargs

    def new_collect(worker_group, output):
        return output

    update_dispatch_mode(original_mode, new_dispatch, new_collect)

    assert get_predefined_dispatch_fn(original_mode)["dispatch_fn"] == new_dispatch
    assert get_predefined_dispatch_fn(original_mode)["collect_fn"] == new_collect
