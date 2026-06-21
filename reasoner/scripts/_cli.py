"""Shared CLI helpers: resolve a variant, load its config, apply overrides."""
from __future__ import annotations

import argparse

from ..base.config import BaseReasonerConfig
from ..base.registry import get_reasoner_class


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--reasoner", required=True, help="registered variant name")
    ap.add_argument("--config", required=True, help="path to variant config.yaml")
    ap.add_argument("--limit", type=int, default=None, help="cap #examples (debug)")
    ap.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides"
    )


def _coerce(value: str, typ):
    if typ is bool:
        return str(value).lower() in {"1", "true", "yes"}
    if typ in (int, float):
        return typ(value)
    return value


def apply_overrides(cfg, pairs: list[str]) -> None:
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"override must be KEY=VALUE, got: {pair}")
        key, value = pair.split("=", 1)
        if not hasattr(cfg, key):
            raise KeyError(f"unknown config field: {key}")
        # Coerce against the current value's type ('from __future__ import
        # annotations' makes the declared field types strings, so we can't use them).
        setattr(cfg, key, _coerce(value, type(getattr(cfg, key))))


def load_config_and_class(args):
    cls = get_reasoner_class(args.reasoner)
    cfg_cls = getattr(cls, "config_cls", BaseReasonerConfig)
    cfg = cfg_cls.from_yaml(args.config)
    apply_overrides(cfg, args.set)
    return cls, cfg
