from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from .model import TaikoTransformer
from .specs import ArchitectureSpec
from .taiko_context import TaikoContextTransformer


ModelBuilder = Callable[[ArchitectureSpec, int], nn.Module]
_MODEL_BUILDERS: dict[str, ModelBuilder] = {}


def register_model_builder(name: str, builder: ModelBuilder) -> None:
    model_name = str(name).strip()
    if not model_name:
        raise ValueError("Model builder name cannot be empty.")
    _MODEL_BUILDERS[model_name] = builder


def build_model(spec: ArchitectureSpec, vocab_size: int) -> nn.Module:
    builder = _MODEL_BUILDERS.get(spec.name)
    if builder is None:
        available = ", ".join(sorted(_MODEL_BUILDERS)) or "<none>"
        raise ValueError(f"Unknown architecture '{spec.name}'. Available: {available}")
    return builder(spec, vocab_size)


def _build_taiko_transformer(spec: ArchitectureSpec, vocab_size: int) -> nn.Module:
    return TaikoTransformer(vocab_size=vocab_size, **spec.model_kwargs())


def _build_taiko_context_transformer(spec: ArchitectureSpec, vocab_size: int) -> nn.Module:
    return TaikoContextTransformer(
        vocab_size=vocab_size,
        **spec.model_kwargs(),
        **spec.context_kwargs(),
    )


register_model_builder("taiko_transformer", _build_taiko_transformer)
register_model_builder("taiko_context_transformer", _build_taiko_context_transformer)
