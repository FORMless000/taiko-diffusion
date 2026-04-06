from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _coerce_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return {str(k): v for k, v in value.items()}


@dataclass
class ModelSpec:
    """Serializable description of a model architecture choice."""

    name: str = "transformer_baseline"
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ModelSpec":
        if data is None:
            return cls()
        return cls(
            name=str(data.get("name", "transformer_baseline")),
            params=_coerce_mapping(data.get("params")),
        )
