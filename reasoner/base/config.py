"""Shared reasoner config. Variants subclass to add their own fields."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BaseReasonerConfig:
    # --- model ---
    llm_path: str = ""
    lora_rank: int = 0                 # 0 = full fine-tune
    lora_alpha: int = 16
    trust_remote_code: bool = True
    bf16: bool = True
    gradient_checkpointing: bool = True

    # --- data ---
    paths_source: str = ""             # explorer artifact jsonl (ReasonerExample records)
    val_source: str = ""               # optional eval jsonl
    num_paths: int = 8                 # how many explorer paths to feed the reasoner
    max_len: int = 4096

    # --- optimization ---
    lr: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    epochs: int = 3
    train_batch_size: int = 16         # global batch (grad-accumulated)
    micro_batch_size: int = 1
    grad_clip: float = 1.0
    seed: int = 42

    # --- io ---
    output_dir: str = "outputs/reasoner"
    save_every_epoch: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "BaseReasonerConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BaseReasonerConfig":
        # Keep only keys this dataclass declares, so subclasses can add fields
        # without the base rejecting them and vice versa.
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in names})
