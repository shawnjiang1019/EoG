"""Train a reasoner variant.

    python -m reasoner.scripts.train \
        --reasoner text_paths \
        --config reasoner/variants/text_paths/config.yaml \
        --set num_paths=4 epochs=1
"""
from __future__ import annotations

import argparse

from .. import variants  # noqa: F401  (registers variants)
from ..common.data import load_examples
from ..common.trainer import train
from ._cli import add_common_args, load_config_and_class


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()

    cls, cfg = load_config_and_class(args)
    reasoner = cls(cfg)
    examples = load_examples(cfg.paths_source, limit=args.limit)
    print(f"training '{args.reasoner}' on {len(examples)} examples", flush=True)
    train(reasoner, examples)


if __name__ == "__main__":
    main()
