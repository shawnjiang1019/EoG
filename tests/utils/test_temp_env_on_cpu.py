

import os

import pytest

from verl.utils.py_functional import temp_env_var

@pytest.fixture(autouse=True)
def clean_env():

    original_env = dict(os.environ)

    test_vars = ["TEST_VAR", "TEST_VAR_2", "EXISTING_VAR"]
    for var in test_vars:
        if var in os.environ:
            del os.environ[var]

    yield

    os.environ.clear()
    os.environ.update(original_env)

def test_set_new_env_var():

    assert "TEST_VAR" not in os.environ

    with temp_env_var("TEST_VAR", "test_value"):

        assert os.environ["TEST_VAR"] == "test_value"
        assert "TEST_VAR" in os.environ

    assert "TEST_VAR" not in os.environ

def test_restore_existing_env_var():

    os.environ["EXISTING_VAR"] = "original_value"

    with temp_env_var("EXISTING_VAR", "temporary_value"):

        assert os.environ["EXISTING_VAR"] == "temporary_value"

    assert os.environ["EXISTING_VAR"] == "original_value"

def test_env_var_restored_on_exception():

    os.environ["EXISTING_VAR"] = "original_value"

    with pytest.raises(ValueError):
        with temp_env_var("EXISTING_VAR", "temporary_value"):

            assert os.environ["EXISTING_VAR"] == "temporary_value"

            raise ValueError("Test exception")

    assert os.environ["EXISTING_VAR"] == "original_value"

def test_nested_context_managers():

    os.environ["TEST_VAR"] = "original"

    with temp_env_var("TEST_VAR", "level1"):
        assert os.environ["TEST_VAR"] == "level1"

        with temp_env_var("TEST_VAR", "level2"):
            assert os.environ["TEST_VAR"] == "level2"

        assert os.environ["TEST_VAR"] == "level1"

    assert os.environ["TEST_VAR"] == "original"

def test_multiple_different_vars():

    os.environ["EXISTING_VAR"] = "existing_value"

    with temp_env_var("EXISTING_VAR", "modified"):
        with temp_env_var("TEST_VAR", "new_value"):
            assert os.environ["EXISTING_VAR"] == "modified"
            assert os.environ["TEST_VAR"] == "new_value"

    assert os.environ["EXISTING_VAR"] == "existing_value"
    assert "TEST_VAR" not in os.environ

def test_empty_string_value():
    with temp_env_var("TEST_VAR", ""):
        assert os.environ["TEST_VAR"] == ""
        assert "TEST_VAR" in os.environ

    assert "TEST_VAR" not in os.environ

def test_overwrite_with_empty_string():
    os.environ["EXISTING_VAR"] = "original"

    with temp_env_var("EXISTING_VAR", ""):
        assert os.environ["EXISTING_VAR"] == ""

    assert os.environ["EXISTING_VAR"] == "original"

def test_context_manager_returns_none():
    with temp_env_var("TEST_VAR", "value") as result:
        assert result is None
        assert os.environ["TEST_VAR"] == "value"
