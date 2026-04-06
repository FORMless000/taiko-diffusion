from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .specs import ModelSpec


ArchitectureBuilder = Callable[[ModelSpec, int, tuple[int, ...] | None], Any]

_ARCHITECTURE_REGISTRY: dict[str, ArchitectureBuilder] = {}


def register_architecture(
    name: str,
    builder: ArchitectureBuilder,
    *,
    overwrite: bool = False,
) -> None:
    key = str(name).strip()
    if not key:
        raise ValueError("Architecture name must be non-empty.")
    if key in _ARCHITECTURE_REGISTRY and not overwrite:
        raise ValueError(f"Architecture '{key}' is already registered.")
    _ARCHITECTURE_REGISTRY[key] = builder


def get_architecture_builder(name: str) -> ArchitectureBuilder:
    key = str(name).strip()
    if key not in _ARCHITECTURE_REGISTRY:
        available = ", ".join(sorted(_ARCHITECTURE_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown architecture '{key}'. Registered: {available}")
    return _ARCHITECTURE_REGISTRY[key]


def list_architectures() -> list[str]:
    return sorted(_ARCHITECTURE_REGISTRY)
