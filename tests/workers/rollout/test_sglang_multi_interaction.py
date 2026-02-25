
import os
import tempfile
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from verl.interactions.base import BaseInteraction
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout

class MockInteraction(BaseInteraction):

    def __init__(self, config):
        super().__init__(config)
        self.started_instances = set()

    async def start_interaction(self, instance_id=None, **kwargs):
        if instance_id is None:
            instance_id = "mock_instance"
        self.started_instances.add(instance_id)
        return instance_id

    async def generate_response(self, instance_id, messages, **kwargs):
        return False, f"Mock response from {self.name}", 1.0, {}

def create_mock_config_with_multi_interactions():

    interaction_config = {
        "interaction": [
            {
                "name": "mock_agent1",
                "class_name": "tests.workers.rollout.test_sglang_multi_interaction.MockInteraction",
                "config": {"param1": "value1"},
            },
            {
                "name": "mock_agent2",
                "class_name": "tests.workers.rollout.test_sglang_multi_interaction.MockInteraction",
                "config": {"param2": "value2"},
            },
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        OmegaConf.save(interaction_config, f.name)
        interaction_config_path = f.name

    config = DictConfig(
        {
            "multi_turn": {
                "interaction_config_path": interaction_config_path,
                "tool_config_path": None,
                "enable": True,
                "max_assistant_turns": 5,
                "max_user_turns": 3,
                "use_inference_chat_template": True,
                "tokenization_sanity_check_mode": "off",
            },
            "prompt_length": 32,
            "response_length": 16,
            "max_model_len": 512,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.8,
            "load_format": "dummy",
            "enforce_eager": True,
            "free_cache_engine": False,
            "calculate_log_probs": False,
            "tensor_model_parallel_size": 1,
            "n": 1,
            "val_kwargs": {"top_k": 1, "top_p": 1.0, "temperature": 0.0},
        }
    )

    return config, interaction_config_path

def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

class TestSGLangMultiInteraction:
    def test_initialize_multiple_interactions(self):
        setup_distributed()
        config, temp_config_path = create_mock_config_with_multi_interactions()

        try:

            with (
                patch.object(SGLangRollout, "_init_distributed_env", return_value=None),
                patch.object(SGLangRollout, "_init_inference_engine", return_value=None),
                patch.object(SGLangRollout, "_init_sampling_params", return_value=None),
            ):

                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", padding_side="left")
                tokenizer.pad_token = tokenizer.eos_token

                mock_model_config = MagicMock()
                mock_model_config.max_position_embeddings = 2048

                mock_model_config.rope_scaling = {
                    "factor": 4.0,
                    "original_max_position_embeddings": 32768,
                    "type": "yarn",
                }

                rollout = SGLangRollout(
                    actor_module="mock_model",
                    config=config,
                    processing_class=tokenizer,
                    model_hf_config=mock_model_config,
                    port=None,
                    trust_remote_code=False,
                    device_mesh=None,
                )

                assert len(rollout.interaction_map) == 2
                assert "mock_agent1" in rollout.interaction_map
                assert "mock_agent2" in rollout.interaction_map

                assert rollout.interaction_map["mock_agent1"].__class__.__name__ == "MockInteraction"
                assert rollout.interaction_map["mock_agent2"].__class__.__name__ == "MockInteraction"

                assert isinstance(rollout.interaction_map["mock_agent1"], BaseInteraction)
                assert isinstance(rollout.interaction_map["mock_agent2"], BaseInteraction)

                assert rollout.interaction_map["mock_agent1"].name == "mock_agent1"
                assert rollout.interaction_map["mock_agent2"].name == "mock_agent2"

        finally:
            os.unlink(temp_config_path)

    def test_interaction_selection_by_name(self):
        setup_distributed()
        config, temp_config_path = create_mock_config_with_multi_interactions()

        try:
            with (
                patch.object(SGLangRollout, "_init_distributed_env", return_value=None),
                patch.object(SGLangRollout, "_init_inference_engine", return_value=None),
                patch.object(SGLangRollout, "_init_sampling_params", return_value=None),
            ):
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", padding_side="left")
                tokenizer.pad_token = tokenizer.eos_token

                mock_model_config = MagicMock()
                mock_model_config.max_position_embeddings = 2048
                mock_model_config.rope_scaling = {
                    "factor": 4.0,
                    "original_max_position_embeddings": 32768,
                    "type": "yarn",
                }

                rollout = SGLangRollout(
                    actor_module="mock_model",
                    config=config,
                    processing_class=tokenizer,
                    model_hf_config=mock_model_config,
                    port=None,
                    trust_remote_code=False,
                    device_mesh=None,
                )

                from verl.workers.rollout.schemas import AsyncRolloutRequest, AsyncRolloutRequestStateEnum, Message

                req = AsyncRolloutRequest(
                    request_id="test_req",
                    state=AsyncRolloutRequestStateEnum.INTERACTING,
                    messages=[Message(role="user", content="test message")],
                    interaction_kwargs={"name": "mock_agent2", "test_param": "value"},
                    input_ids=None,
                    prompt_ids=None,
                    response_ids=None,
                    attention_mask=None,
                    prompt_attention_mask=None,
                    response_attention_mask=None,
                    position_ids=None,
                    prompt_position_ids=None,
                    response_position_ids=None,
                    loss_mask=None,
                    prompt_loss_mask=None,
                    response_loss_mask=None,
                    reward_scores={},
                    max_prompt_len=32,
                    max_response_len=16,
                    max_model_len=512,
                    use_inference_chat_template=True,
                    tokenization_sanity_check_mode="disable",
                    processing_class=tokenizer,
                )

                interaction_name = req.interaction_kwargs.get("name", "gsm8k")
                assert interaction_name == "mock_agent2"
                assert interaction_name in rollout.interaction_map

                selected_interaction = rollout.interaction_map[interaction_name]
                assert selected_interaction.name == "mock_agent2"

        finally:
            os.unlink(temp_config_path)

    def test_fallback_to_default_interaction(self):
        setup_distributed()

        interaction_config = {
            "interaction": [
                {
                    "name": "gsm8k",
                    "class_name": "tests.workers.rollout.test_sglang_multi_interaction.MockInteraction",
                    "config": {},
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            OmegaConf.save(interaction_config, f.name)
            interaction_config_path = f.name

        config = DictConfig(
            {
                "multi_turn": {
                    "interaction_config_path": interaction_config_path,
                    "tool_config_path": None,
                    "enable": True,
                    "max_assistant_turns": 5,
                    "max_user_turns": 3,
                    "use_inference_chat_template": True,
                    "tokenization_sanity_check_mode": "disable",
                },
                "prompt_length": 32,
                "response_length": 16,
                "max_model_len": 512,
                "dtype": "bfloat16",
                "gpu_memory_utilization": 0.8,
                "load_format": "dummy",
                "enforce_eager": True,
                "free_cache_engine": False,
                "calculate_log_probs": False,
                "tensor_model_parallel_size": 1,
                "n": 1,
                "val_kwargs": {"top_k": 1, "top_p": 1.0, "temperature": 0.0},
            }
        )

        try:
            with (
                patch.object(SGLangRollout, "_init_distributed_env", return_value=None),
                patch.object(SGLangRollout, "_init_inference_engine", return_value=None),
                patch.object(SGLangRollout, "_init_sampling_params", return_value=None),
            ):
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", padding_side="left")
                tokenizer.pad_token = tokenizer.eos_token

                mock_model_config = MagicMock()
                mock_model_config.max_position_embeddings = 2048
                mock_model_config.rope_scaling = {
                    "factor": 4.0,
                    "original_max_position_embeddings": 32768,
                    "type": "yarn",
                }

                rollout = SGLangRollout(
                    actor_module="mock_model",
                    config=config,
                    processing_class=tokenizer,
                    model_hf_config=mock_model_config,
                    port=None,
                    trust_remote_code=False,
                    device_mesh=None,
                )

                interaction_kwargs_without_name = {"test_param": "value"}
                default_name = interaction_kwargs_without_name.get("name", "gsm8k")
                assert default_name == "gsm8k"
                assert default_name in rollout.interaction_map

        finally:
            os.unlink(interaction_config_path)

    def test_error_on_missing_interaction(self):
        setup_distributed()
        config, temp_config_path = create_mock_config_with_multi_interactions()

        try:
            with (
                patch.object(SGLangRollout, "_init_distributed_env", return_value=None),
                patch.object(SGLangRollout, "_init_inference_engine", return_value=None),
                patch.object(SGLangRollout, "_init_sampling_params", return_value=None),
            ):
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", padding_side="left")
                tokenizer.pad_token = tokenizer.eos_token

                mock_model_config = MagicMock()
                mock_model_config.max_position_embeddings = 2048
                mock_model_config.rope_scaling = {
                    "factor": 4.0,
                    "original_max_position_embeddings": 32768,
                    "type": "yarn",
                }

                rollout = SGLangRollout(
                    actor_module="mock_model",
                    config=config,
                    processing_class=tokenizer,
                    model_hf_config=mock_model_config,
                    port=None,
                    trust_remote_code=False,
                    device_mesh=None,
                )

                non_existent_name = "non_existent_interaction"
                assert non_existent_name not in rollout.interaction_map

                available_interactions = list(rollout.interaction_map.keys())
                assert "mock_agent1" in available_interactions
                assert "mock_agent2" in available_interactions
                assert non_existent_name not in available_interactions

        finally:
            os.unlink(temp_config_path)

    def test_backward_compatibility_no_interaction_config(self):
        setup_distributed()

        config = DictConfig(
            {
                "multi_turn": {
                    "interaction_config_path": None,
                    "tool_config_path": None,
                    "enable": True,
                    "max_assistant_turns": 5,
                    "max_user_turns": 3,
                    "use_inference_chat_template": True,
                    "tokenization_sanity_check_mode": "disable",
                },
                "prompt_length": 32,
                "response_length": 16,
                "max_model_len": 512,
                "dtype": "bfloat16",
                "gpu_memory_utilization": 0.8,
                "load_format": "dummy",
                "enforce_eager": True,
                "free_cache_engine": False,
                "calculate_log_probs": False,
                "tensor_model_parallel_size": 1,
                "n": 1,
                "val_kwargs": {"top_k": 1, "top_p": 1.0, "temperature": 0.0},
            }
        )

        with (
            patch.object(SGLangRollout, "_init_distributed_env", return_value=None),
            patch.object(SGLangRollout, "_init_inference_engine", return_value=None),
            patch.object(SGLangRollout, "_init_sampling_params", return_value=None),
        ):
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", padding_side="left")
            tokenizer.pad_token = tokenizer.eos_token

            mock_model_config = MagicMock()
            mock_model_config.max_position_embeddings = 2048
            mock_model_config.rope_scaling = {
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
                "type": "yarn",
            }

            rollout = SGLangRollout(
                actor_module="mock_model",
                config=config,
                processing_class=tokenizer,
                model_hf_config=mock_model_config,
                port=None,
                trust_remote_code=False,
                device_mesh=None,
            )

            assert len(rollout.interaction_map) == 0
