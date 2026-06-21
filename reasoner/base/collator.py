"""Batch collation. Default handles tokenized text; embedding variants override."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseCollator(ABC):
    @abstractmethod
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        ...


class DefaultTextCollator(BaseCollator):
    """Right-pads `input_ids`/`attention_mask`/`labels` to the batch max length.

    Expects each feature dict (from a reasoner's `build_inputs`) to contain
    list[int] `input_ids` and `labels`; `attention_mask` is derived if absent.
    Label padding is -100 so padding tokens are ignored by the loss.
    """

    def __init__(self, tokenizer, label_pad_id: int = -100):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.label_pad_id = label_pad_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attn, labels = [], [], []
        for f in features:
            ids = list(f["input_ids"])
            lab = list(f.get("labels", ids))
            mask = list(f.get("attention_mask", [1] * len(ids)))
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            attn.append(mask + [0] * pad)
            labels.append(lab + [self.label_pad_id] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
