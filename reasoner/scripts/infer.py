"""Run a reasoner over examples and write predictions to jsonl.

    python -m reasoner.scripts.infer \
        --reasoner text_paths \
        --config reasoner/variants/text_paths/config.yaml \
        --out preds.jsonl
"""
from __future__ import annotations

import argparse
import json

import torch

from .. import variants  # noqa: F401
from ..common.data import load_examples
from ._cli import add_common_args, load_config_and_class


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--out", required=True, help="output predictions jsonl")
    args = ap.parse_args()

    cls, cfg = load_config_and_class(args)
    reasoner = cls(cfg)
    reasoner.build_model()
    reasoner.to("cuda" if torch.cuda.is_available() else "cpu")
    reasoner.eval()

    source = cfg.val_source or cfg.paths_source
    examples = load_examples(source, limit=args.limit)
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in examples:
            out = reasoner.postprocess(reasoner.generate(ex), ex)
            f.write(
                json.dumps(
                    {"id": ex.id, "pred_answers": out.pred_answers, "raw": out.raw},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {len(examples)} predictions -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
