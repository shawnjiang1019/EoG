

import math

import pytest

from verl.utils.flops_counter import FlopsCounter

VALID_CONFIG_TYPE = {"llama", "qwen2", "qwen3", "qwen3_moe", "deepseek_v3", "mistral", "gemma3_text"}

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

CONFIG = {
    "llama": {
        "config": {
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
        },
        "batch_seqlens_tuple": ([512, 1024, 2048], [4096, 4096, 4096]),

        "expected_flops_tuple": (153555818250240 / 1e12, 575955114393600 / 1e12),
    },
    "qwen2": {
        "config": {
            "model_type": "qwen2",
            "vocab_size": 152064,
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
        },
        "batch_seqlens_tuple": ([512, 1024, 2048], [4096, 4096, 4096]),

        "expected_flops_tuple": (170388331954176 / 1e12, 622070178250752 / 1e12),
    },
    "qwen3": {
        "config": {
            "model_type": "qwen3",
            "vocab_size": 151936,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
        "batch_seqlens_tuple": ([512, 1024, 2048], [4096, 4096, 4096]),

        "expected_flops_tuple": (185867930959872 / 1e12, 692924253732864 / 1e12),
    },
    "qwen3_moe": {
        "config": {
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "vocab_size": 151936,
            "num_hidden_layers": 48,
            "num_key_value_heads": 4,
            "num_attention_heads": 32,
            "head_dim": 128,
            "moe_intermediate_size": 768,
            "num_experts_per_tok": 8,
            "num_experts": 128,
        },
        "batch_seqlens_tuple": ([512, 1024, 2048], [4096, 4096, 4096]),

        "expected_flops_tuple": (85087060230144 / 1e12, 365944098521088 / 1e12),
    },
    "deepseek_v3": {
        "config": {
            "model_type": "deepseek_v3",
            "hidden_size": 7168,
            "vocab_size": 129280,
            "moe_intermediate_size": 2048,
            "num_hidden_layers": 61,
            "first_k_dense_replace": 3,
            "num_attention_heads": 128,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "intermediate_size": 18432,
            "qk_nope_head_dim": 128,
            "q_lora_rank": 1536,
        },
        "batch_seqlens_tuple": ([512, 1024, 2048], [4096, 4096, 4096]),

        "expected_flops_tuple": (906535995703296 / 1e12, 3674028304760832 / 1e12),
    },
    "mistral": {
        "config": {
            "model_type": "mistral",
            "vocab_size": 131072,
            "hidden_size": 5120,
            "intermediate_size": 32768,
            "num_hidden_layers": 40,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
        "batch_seqlens_tuple": ([512, 1024, 2048], [4096, 4096, 4096]),

        "expected_flops_tuple": (517715357859840 / 1e12, 1836871613153280 / 1e12),
    },
    "gemma3_text": {
        "config": {
            "model_type": "gemma3_text",
            "vocab_size": 262208,
            "hidden_size": 3840,
            "intermediate_size": 15360,
            "num_hidden_layers": 48,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 256,
            "sliding_window": 1024,
            "layer_types": None,

            "sliding_window_pattern": 6,

        },
        "batch_seqlens_tuple": ([512, 1024, 2048], [4096, 4096, 4096]),

        "expected_flops_tuple": (283517065887744 / 1e12, 986195089686528 / 1e12),
    },
}

@pytest.mark.parametrize(
    "config_type",
    ["llama", "qwen2", "qwen3", "qwen3_moe", "deepseek_v3", "mistral", "gemma3_text"],
)
def test_flops_counter(config_type: str):
    test_config = CONFIG[config_type]
    config = Config(test_config["config"])
    flops_counter = FlopsCounter(config)
    for batch_seqlens, expected_flops in zip(
        test_config["batch_seqlens_tuple"], test_config["expected_flops_tuple"], strict=True
    ):

        counted_flops, _ = flops_counter.estimate_flops(batch_seqlens, 1)
        print(f"Expect flops for {test_config['config']} is {expected_flops}, but get {counted_flops}")
        assert math.isclose(counted_flops, expected_flops), (
            f"Expect flops for {test_config['config']} is {expected_flops}, but get {counted_flops}"
        )
