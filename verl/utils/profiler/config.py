

from dataclasses import dataclass, field
from typing import ClassVar

from verl.base_config import BaseConfig

@dataclass
class ProfilerConfig(BaseConfig):

    _frozen_fields: ClassVar[set[str]] = {"discrete", "all_ranks", "ranks"}

    discrete: bool = False

    all_ranks: bool = False

    ranks: list[int] = field(default_factory=list)

    def union(self, other: "ProfilerConfig") -> "ProfilerConfig":
        return ProfilerConfig(
            all_ranks=self.all_ranks or other.all_ranks,
            ranks=list(set(self.ranks or []) | set(other.ranks or [])),
            discrete=self.discrete or other.discrete,
        )

    def intersect(self, other: "ProfilerConfig") -> "ProfilerConfig":
        return ProfilerConfig(
            all_ranks=self.all_ranks and other.all_ranks,
            ranks=list(set(self.ranks or []) & set(other.ranks or [])),
            discrete=self.discrete and other.discrete,
        )

    def __post_init__(self) -> None:
        assert isinstance(self.ranks, set | list | tuple), (
            f"Profiler ranks must be of type list, got {type(self.ranks)}"
        )
