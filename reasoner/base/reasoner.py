"""BaseReasoner: the interface every variant implements.

All variation is isolated to four required methods + two optional hooks. The
shared trainer / data / eval never branch on the variant.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from .collator import BaseCollator, DefaultTextCollator
from .config import BaseReasonerConfig
from .schema import ReasonerExample, ReasonerOutput


class BaseReasoner(ABC):
    #: config dataclass for this variant; subclasses override with their own.
    config_cls: type = BaseReasonerConfig

    def __init__(self, config: BaseReasonerConfig):
        self.config = config
        self.model = None        # set by build_model()
        self.tokenizer = None    # set by build_model()

    # --- required ---------------------------------------------------------
    @abstractmethod
    def build_model(self) -> None:
        """Instantiate self.model (+ self.tokenizer, projector/GNN if any)."""

    @abstractmethod
    def build_inputs(self, ex: ReasonerExample) -> dict[str, Any]:
        """Turn one example into training features (e.g. input_ids/labels, or
        inputs_embeds + path tensors). The primary point of variation."""

    @abstractmethod
    def compute_loss(self, batch: dict[str, Any]) -> "Any":
        """Forward pass returning a scalar loss tensor for a collated batch."""

    @abstractmethod
    def generate(self, ex: ReasonerExample) -> ReasonerOutput:
        """Produce an answer for one example at inference time."""

    # --- optional hooks ---------------------------------------------------
    def build_collator(self) -> BaseCollator:
        return DefaultTextCollator(self.tokenizer)

    def postprocess(self, out: ReasonerOutput, ex: ReasonerExample) -> ReasonerOutput:
        """Hook for symbolic verification/repair etc. No-op by default."""
        return out

    # --- shared helpers ---------------------------------------------------
    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def to(self, device):
        self.model.to(device)
        return self

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        # Works for HF / PEFT models; variants with extra modules override.
        self.model.save_pretrained(path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)
