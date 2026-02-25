

import torch
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from transformers import (
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    GemmaConfig,
    LlamaConfig,
    MistralConfig,
    Qwen2Config,
)

from verl.utils.model import compute_position_id_with_mask, create_random_mask
from verl.utils.torch_functional import log_probs_from_logits_all_rmpad, masked_mean

test_configs = [
    LlamaConfig(num_hidden_layers=1),
    MistralConfig(num_hidden_layers=1),
    GemmaConfig(num_hidden_layers=1),
    Qwen2Config(num_hidden_layers=1),
]

def test_hf_casual_models():
    batch_size = 4
    seqlen = 128
    response_length = 127

    for config in test_configs:

        with torch.device("cuda"):
            model = AutoModelForCausalLM.from_config(
                config=config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
            )
            model = model.to(device="cuda")
        input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seqlen), device="cuda")
        attention_mask = create_random_mask(
            input_ids=input_ids,
            max_ratio_of_left_padding=0.1,
            max_ratio_of_valid_token=0.8,
            min_ratio_of_valid_token=0.5,
        )
        position_ids = compute_position_id_with_mask(
            attention_mask
        )

        input_ids_rmpad, indices, *_ = unpad_input(
            input_ids.unsqueeze(-1), attention_mask
        )
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)

        logits_rmpad = model(
            input_ids_rmpad, position_ids=position_ids_rmpad, use_cache=False
        ).logits

        origin_logits = model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
        ).logits
        origin_logits_rmpad, origin_logits_indices, *_ = unpad_input(origin_logits, attention_mask)

        logits_rmpad = logits_rmpad.squeeze(0)
        log_probs = log_probs_from_logits_all_rmpad(
            input_ids_rmpad=input_ids_rmpad,
            logits_rmpad=logits_rmpad,
            indices=indices,
            batch_size=batch_size,
            seqlen=seqlen,
            response_length=response_length,
        )
        origin_log_probs = log_probs_from_logits_all_rmpad(
            input_ids_rmpad=input_ids_rmpad,
            logits_rmpad=origin_logits_rmpad,
            indices=origin_logits_indices,
            batch_size=batch_size,
            seqlen=seqlen,
            response_length=response_length,
        )

        torch.testing.assert_close(
            masked_mean(log_probs, attention_mask[:, -response_length - 1 : -1]),
            masked_mean(origin_log_probs, attention_mask[:, -response_length - 1 : -1]),
            atol=1e-2,
            rtol=1e-5,
        )
    print("Check pass")

def test_hf_value_models():
    batch_size = 4
    seqlen = 128

    for config in test_configs:

        config.num_labels = 1
        config.classifier_dropout = 0
        config.hidden_dropout = 0
        with torch.device("cuda"):
            model = AutoModelForTokenClassification.from_config(
                config=config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
            )
            model = model.to(device="cuda")
        input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seqlen), device="cuda")
        attention_mask = create_random_mask(
            input_ids=input_ids,
            max_ratio_of_left_padding=0.1,
            max_ratio_of_valid_token=0.8,
            min_ratio_of_valid_token=0.5,
        )
        position_ids = compute_position_id_with_mask(
            attention_mask
        )

        input_ids_rmpad, indices, *_ = unpad_input(
            input_ids.unsqueeze(-1), attention_mask
        )
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)

        origin_logits = model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
        ).logits

        rmpad_logits = model(
            input_ids_rmpad, position_ids=position_ids_rmpad, use_cache=False
        ).logits
        rmpad_logits = rmpad_logits.squeeze(0)
        pad_logits = pad_input(rmpad_logits, indices, batch_size, seqlen=seqlen)

        torch.testing.assert_close(
            masked_mean(pad_logits, attention_mask[:, :, None]),
            masked_mean(origin_logits, attention_mask[:, :, None]),
            atol=1e-2,
            rtol=1e-5,
        )
    print("Value model check pass")

if __name__ == "__main__":
    test_hf_casual_models()
    test_hf_value_models()
