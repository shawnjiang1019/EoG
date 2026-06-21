"""Name -> reasoner-class registry. Adding a variant needs no edits to scripts."""
from __future__ import annotations

from typing import Type

_REGISTRY: dict[str, Type] = {}


def register_reasoner(name: str):
    def deco(cls):
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"reasoner '{name}' already registered to {_REGISTRY[name]}")
        _REGISTRY[name] = cls
        return cls

    return deco


def get_reasoner_class(name: str) -> Type:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown reasoner '{name}'. available: {available_reasoners()}. "
            f"(is its variant module imported in reasoner/variants/__init__.py?)"
        )
    return _REGISTRY[name]


def build_reasoner(name: str, config):
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown reasoner '{name}'. available: {available_reasoners()}. "
            f"(is its variant module imported in reasoner/variants/__init__.py?)"
        )
    return _REGISTRY[name](config)


def available_reasoners() -> list[str]:
    return sorted(_REGISTRY)
