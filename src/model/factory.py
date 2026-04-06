from __future__ import annotations

from . import architectures  # noqa: F401
from .registry import get_architecture_builder
from .specs import ModelSpec


def build_model(
    model_spec: ModelSpec | None,
    vocab_size: int,
    input_shape: tuple[int, ...] | None = None,
):
    spec = model_spec or ModelSpec()
    builder = get_architecture_builder(spec.name)
    return builder(spec, vocab_size=vocab_size, input_shape=input_shape)
