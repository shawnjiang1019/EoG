

import collections
from dataclasses import (
    dataclass,
    field,
    fields,
)
from typing import Any

@dataclass
class BaseConfig(collections.abc.Mapping):

    extra: dict[str, Any] = field(default_factory=dict)

    def __setattr__(self, name: str, value):

        if hasattr(self, "_frozen_fields") and name in self._frozen_fields and name in self.__dict__:
            from dataclasses import FrozenInstanceError

            raise FrozenInstanceError(f"Field '{name}' is frozen and cannot be modified")

        super().__setattr__(name, value)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __iter__(self):
        for f in fields(self):
            yield f.name

    def __len__(self):
        return len(fields(self))
