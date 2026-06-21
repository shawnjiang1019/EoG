"""Evaluate a reasoner: generate over the val set, report Hit@1 / F1.

    python -m reasoner.scripts.eval \
        --reasoner text_paths \
        --config reasoner/variants/text_paths/config.yaml \
        --set llm_path=./outputs/reasoner/text_paths/final val_source=./data/reasoner/2wiki_test.jsonl

Note: for a LoRA checkpoint, merge it first (or set llm_path to a merged dir).
"""
from __future__ import annotations

import argparse

import torch

from .. import variants  # noqa: F401
from ..common.data import load_examples
from ..common.eval import score_predictions
from ._cli import add_common_args, load_config_and_class


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()

    cls, cfg = load_config_and_class(args)
    reasoner = cls(cfg)
    reasoner.build_model()
    reasoner.to("cuda" if torch.cuda.is_available() else "cpu")
    reasoner.eval()

    source = cfg.val_source or cfg.paths_source
    examples = load_examples(source, limit=args.limit)
    outputs = [reasoner.postprocess(reasoner.generate(ex), ex) for ex in examples]

    metrics = score_predictions(examples, outputs)
    print(f"[{args.reasoner}] {metrics}", flush=True)


if __name__ == "__main__":
    main()
