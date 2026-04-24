from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch

from .specs import ArchitectureSpec, TrainingSpec


@dataclass
class CheckpointMetadata:
    epoch: int
    global_step: int
    best_val_loss: float | None
    data_root: str
    artifact_paths: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointMetadata":
        return cls(**data)


def capture_rng_states() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_states(state: dict[str, Any] | None) -> None:
    if not state:
        return

    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)

    torch_state = state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)

    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def save_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    architecture_spec: ArchitectureSpec,
    training_spec: TrainingSpec,
    metadata: CheckpointMetadata,
    history: dict[str, Any],
    vocab: dict[str, Any],
    split_ids: dict[str, list[str]],
    adherence_config: dict[str, Any] | None = None,
    scheduler: Any = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": metadata.to_dict(),
        "architecture_spec": architecture_spec.to_dict(),
        "training_spec": training_spec.to_dict(),
        "history": history,
        "vocab": vocab,
        "split_ids": split_ids,
        "adherence_config": adherence_config or {},
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "rng_state": capture_rng_states(),
    }

    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{checkpoint_path.name}.",
        suffix=".tmp",
        dir=str(checkpoint_path.parent),
    )
    os.close(fd)
    tmp_checkpoint_path = Path(tmp_path)
    try:
        torch.save(payload, tmp_checkpoint_path)
        os.replace(tmp_checkpoint_path, checkpoint_path)
    finally:
        if tmp_checkpoint_path.exists():
            tmp_checkpoint_path.unlink()
    return checkpoint_path


def save_inference_bundle(
    bundle_path: str | Path,
    *,
    model: torch.nn.Module,
    architecture_spec: ArchitectureSpec,
    vocab: dict[str, Any],
    global_step: int,
    epoch: int | None = None,
    adherence_config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    bundle_path = Path(bundle_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "artifact_type": "inference_bundle",
        "format_version": 1,
        "metadata": {
            "global_step": int(global_step),
            "epoch": None if epoch is None else int(epoch),
            **(metadata or {}),
        },
        "architecture_spec": architecture_spec.to_dict(),
        "vocab": vocab,
        "adherence_config": adherence_config or {},
        "model_state_dict": model.state_dict(),
    }

    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{bundle_path.name}.",
        suffix=".tmp",
        dir=str(bundle_path.parent),
    )
    os.close(fd)
    tmp_bundle_path = Path(tmp_path)
    try:
        torch.save(payload, tmp_bundle_path)
        os.replace(tmp_bundle_path, bundle_path)
    finally:
        if tmp_bundle_path.exists():
            tmp_bundle_path.unlink()
    return bundle_path


def load_inference_artifacts(checkpoint_path: str | Path, map_location: str | torch.device | None = None) -> dict[str, Any]:
    payload = load_checkpoint(checkpoint_path, map_location=map_location)

    if "model_state_dict" not in payload:
        raise ValueError(f"Checkpoint at '{checkpoint_path}' does not contain model weights.")
    if "architecture_spec" not in payload:
        raise ValueError(f"Checkpoint at '{checkpoint_path}' does not contain architecture_spec.")
    if "vocab" not in payload:
        raise ValueError(f"Checkpoint at '{checkpoint_path}' does not contain vocab.")

    return {
        "artifact_type": str(payload.get("artifact_type", "training_checkpoint")),
        "metadata": dict(payload.get("metadata", {})),
        "architecture_spec": dict(payload["architecture_spec"]),
        "vocab": dict(payload["vocab"]),
        "adherence_config": dict(payload.get("adherence_config", {})),
        "model_state_dict": payload["model_state_dict"],
    }


def load_checkpoint(checkpoint_path: str | Path, map_location: str | torch.device | None = None) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
