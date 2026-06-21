"""Load the explorer artifact into a torch Dataset of reasoner features.

The artifact is a jsonl of `ReasonerExample` records (see base/schema.py). Each
variant's `build_inputs` turns an example into model-ready features; this module
is variant-agnostic.
"""
from __future__ import annotations

from typing import Optional

from torch.utils.data import Dataset

from ..base.reasoner import BaseReasoner
from ..base.schema import ReasonerExample, read_jsonl


def load_examples(path: str, limit: Optional[int] = None) -> list[ReasonerExample]:
    out: list[ReasonerExample] = []
    for ex in read_jsonl(path):
        out.append(ex)
        if limit is not None and len(out) >= limit:
            break
    return out


class ReasonerDataset(Dataset):
    """Lazily maps examples -> features through the reasoner's build_inputs."""

    def __init__(self, examples: list[ReasonerExample], reasoner: BaseReasoner):
        self.examples = examples
        self.reasoner = reasoner

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int):
        return self.reasoner.build_inputs(self.examples[i])
