"""Reusable reasoner architecture.

Spine (base/ + common/) is fixed; each experiment is a plugin under variants/.
Importing this package registers all variants.
"""
from . import variants  # noqa: F401  (registers variants)
from .base import (
    BaseReasoner,
    BaseReasonerConfig,
    ReasonerExample,
    ReasonerOutput,
    available_reasoners,
    build_reasoner,
    get_reasoner_class,
)

__all__ = [
    "BaseReasoner",
    "BaseReasonerConfig",
    "ReasonerExample",
    "ReasonerOutput",
    "available_reasoners",
    "build_reasoner",
    "get_reasoner_class",
]
