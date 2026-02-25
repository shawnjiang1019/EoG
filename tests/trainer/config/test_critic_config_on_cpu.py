

import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from verl.trainer.config.config import CriticConfig, FSDPCriticConfig, MegatronCriticConfig
from verl.utils.config import omega_conf_to_dataclass

class TestCriticConfig:

    @pytest.fixture
    def config_dir(self):
        return Path(__file__).parent.parent.parent.parent / "verl" / "trainer" / "config" / "critic"

    def test_megatron_critic_config_instantiation_from_yaml(self, config_dir):
        yaml_path = config_dir / "megatron_critic.yaml"
        assert yaml_path.exists(), f"Config file not found: {yaml_path}"

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/critic")):
            test_config = compose(config_name="megatron_critic")

        megatron_config_obj = omega_conf_to_dataclass(test_config)

        assert isinstance(megatron_config_obj, MegatronCriticConfig)
        assert isinstance(megatron_config_obj, CriticConfig)

        expected_attrs = [
            "strategy",
            "rollout_n",
            "optim",
            "model",
            "ppo_mini_batch_size",
            "ppo_max_token_len_per_gpu",
            "cliprange_value",
            "get",
            "nccl_timeout",
            "megatron",
            "load_weight",
        ]
        for attr in expected_attrs:
            assert hasattr(megatron_config_obj, attr), f"Missing attribute: {attr}"

        assert callable(megatron_config_obj.get)
        assert megatron_config_obj.strategy == "megatron"

    def test_fsdp_critic_config_instantiation_from_yaml(self, config_dir):
        yaml_path = config_dir / "dp_critic.yaml"
        assert yaml_path.exists(), f"Config file not found: {yaml_path}"

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/critic")):
            test_config = compose(config_name="dp_critic")

        fsdp_config_obj = omega_conf_to_dataclass(test_config)

        assert isinstance(fsdp_config_obj, FSDPCriticConfig)
        assert isinstance(fsdp_config_obj, CriticConfig)

        expected_attrs = [
            "strategy",
            "rollout_n",
            "optim",
            "model",
            "ppo_mini_batch_size",
            "ppo_max_token_len_per_gpu",
            "cliprange_value",
            "get",
            "forward_micro_batch_size",
            "forward_micro_batch_size_per_gpu",
            "ulysses_sequence_parallel_size",
            "grad_clip",
        ]
        for attr in expected_attrs:
            assert hasattr(fsdp_config_obj, attr), f"Missing attribute: {attr}"

        assert callable(fsdp_config_obj.get)
        assert fsdp_config_obj.strategy == "fsdp"

    def test_config_inheritance_hierarchy(self):
        megatron_config = MegatronCriticConfig()
        assert isinstance(megatron_config, CriticConfig)
        assert isinstance(megatron_config, MegatronCriticConfig)

        fsdp_config = FSDPCriticConfig()
        assert isinstance(fsdp_config, CriticConfig)
        assert isinstance(fsdp_config, FSDPCriticConfig)

        critic_config = CriticConfig()
        assert isinstance(critic_config, CriticConfig)
        assert not isinstance(critic_config, MegatronCriticConfig)
        assert not isinstance(critic_config, FSDPCriticConfig)

    def test_config_dict_interface(self):
        config = CriticConfig()

        assert "strategy" in config
        assert config["strategy"] == "fsdp"

        assert config.get("strategy") == "fsdp"
        assert config.get("nonexistent_key", "default") == "default"

        keys = list(config)
        assert "strategy" in keys
        assert "rollout_n" in keys

        assert len(config) > 0

    def test_frozen_fields_immutability(self):
        critic_config = CriticConfig()
        frozen_fields = ["rollout_n", "strategy", "cliprange_value"]

        for field_name in frozen_fields:
            with pytest.raises((AttributeError, TypeError, ValueError)):
                setattr(critic_config, field_name, "modified_value")

        megatron_config = MegatronCriticConfig()
        megatron_frozen_fields = ["nccl_timeout", "load_weight", "data_loader_seed"]

        for field_name in megatron_frozen_fields:
            with pytest.raises((AttributeError, TypeError, ValueError)):
                setattr(megatron_config, field_name, "modified_value")

        fsdp_config = FSDPCriticConfig()
        fsdp_frozen_fields = ["ulysses_sequence_parallel_size", "grad_clip"]

        for field_name in fsdp_frozen_fields:
            with pytest.raises((AttributeError, TypeError, ValueError)):
                setattr(fsdp_config, field_name, "modified_value")

    def test_batch_size_fields_modifiable(self):
        critic_config = CriticConfig()

        critic_config.ppo_mini_batch_size = 8
        critic_config.ppo_micro_batch_size = 4
        critic_config.ppo_micro_batch_size_per_gpu = 2

        assert critic_config.ppo_mini_batch_size == 8
        assert critic_config.ppo_micro_batch_size == 4
        assert critic_config.ppo_micro_batch_size_per_gpu == 2

        fsdp_config = FSDPCriticConfig()

        fsdp_config.forward_micro_batch_size = 16
        fsdp_config.forward_micro_batch_size_per_gpu = 8

        assert fsdp_config.forward_micro_batch_size == 16
        assert fsdp_config.forward_micro_batch_size_per_gpu == 8
