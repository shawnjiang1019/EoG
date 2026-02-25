

import pytest

from verl.base_config import BaseConfig

@pytest.fixture
def base_config_mock():
    mock_config = BaseConfig()
    mock_config.test_attr = "test_value"
    return mock_config

def test_getitem_success(base_config_mock):
    assert base_config_mock["test_attr"] == "test_value"

def test_getitem_nonexistent_attribute(base_config_mock):
    with pytest.raises(AttributeError):
        _ = base_config_mock["nonexistent_attr"]

def test_getitem_invalid_key_type(base_config_mock):
    with pytest.raises(TypeError):
        _ = base_config_mock[123]
