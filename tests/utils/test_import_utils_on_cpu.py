

import os

import pytest

from verl.utils.import_utils import load_extern_type

TEST_MODULE_PATH = os.path.join(os.path.dirname(__file__), "_test_module.py")

def test_load_extern_type_class():
    TestClass = load_extern_type(TEST_MODULE_PATH, "TestClass")

    assert TestClass is not None
    assert TestClass.__name__ == "TestClass"

    instance = TestClass()
    assert instance.value == "default"

    custom_instance = TestClass("custom")
    assert custom_instance.get_value() == "custom"

def test_load_extern_type_function():
    test_function = load_extern_type(TEST_MODULE_PATH, "test_function")

    assert test_function is not None
    assert callable(test_function)

    result = test_function()
    assert result == "test_function_result"

def test_load_extern_type_constant():
    constant = load_extern_type(TEST_MODULE_PATH, "TEST_CONSTANT")

    assert constant is not None
    assert constant == "test_constant_value"

def test_load_extern_type_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_extern_type("/nonexistent/path.py", "SomeType")

def test_load_extern_type_nonexistent_type():
    with pytest.raises(AttributeError):
        load_extern_type(TEST_MODULE_PATH, "NonExistentType")

def test_load_extern_type_none_path():
    result = load_extern_type(None, "SomeType")
    assert result is None

def test_load_extern_type_invalid_module():

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w+", delete=False) as temp_file:
        temp_file.write("This is not valid Python syntax :")
        temp_path = temp_file.name

    try:
        with pytest.raises(RuntimeError):
            load_extern_type(temp_path, "SomeType")
    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)
