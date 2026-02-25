

import pytest

from verl.workers.reward_manager.registry import REWARD_MANAGER_REGISTRY, get_reward_manager_cls, register

@pytest.fixture
def setup():
    REWARD_MANAGER_REGISTRY.clear()
    REWARD_MANAGER_REGISTRY.update({"manager1": "Manager1Class", "manager2": "Manager2Class"})
    return REWARD_MANAGER_REGISTRY

def test_get_existing_manager(setup):
    assert get_reward_manager_cls("manager1") == "Manager1Class"
    assert get_reward_manager_cls("manager2") == "Manager2Class"

def test_get_nonexistent_manager(setup):
    with pytest.raises(ValueError) as excinfo:
        get_reward_manager_cls("unknown_manager")
    assert "Unknown reward manager: unknown_manager" in str(excinfo.value)

def test_case_sensitivity(setup):
    with pytest.raises(ValueError):
        get_reward_manager_cls("MANAGER1")
    with pytest.raises(ValueError):
        get_reward_manager_cls("Manager1")

def test_empty_registry(setup):
    REWARD_MANAGER_REGISTRY.clear()
    with pytest.raises(ValueError) as excinfo:
        get_reward_manager_cls("any_manager")
    assert "Unknown reward manager: any_manager" in str(excinfo.value)

def test_register_new_class(setup):

    @register("test_manager")
    class TestManager:
        pass

    assert "test_manager" in REWARD_MANAGER_REGISTRY
    assert REWARD_MANAGER_REGISTRY["test_manager"] == TestManager

def test_register_different_classes_same_name(setup):

    @register("conflict_manager")
    class Manager1:
        pass

    with pytest.raises(ValueError):

        @register("conflict_manager")
        class Manager2:
            pass

    assert REWARD_MANAGER_REGISTRY["conflict_manager"] == Manager1

def test_decorator_returns_original_class(setup):

    @register("return_test")
    class OriginalClass:
        def method(setup):
            return 42

    assert OriginalClass().method() == 42
    assert REWARD_MANAGER_REGISTRY["return_test"] == OriginalClass
