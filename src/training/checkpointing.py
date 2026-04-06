from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


CHECKPOINT_FORMAT_VERSION = 1


def get_run_paths(run_dir: str | Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    metadata_dir = run_dir / "metadata"
    data_dir = run_dir / "data"
    checkpoints_dir = run_dir / "checkpoints"
    return {
        "run_dir": run_dir,
        "metadata_dir": metadata_dir,
        "data_dir": data_dir,
        "checkpoints_dir": checkpoints_dir,
        "config_json": metadata_dir / "run_config.json",
        "history_json": metadata_dir / "history.json",
        "raw_manifest_json": metadata_dir / "raw_osz_manifest.json",
        "splits_json": metadata_dir / "splits.json",
        "vocab_json": metadata_dir / "vocab.json",
        "dataset_info_json": metadata_dir / "dataset_info.json",
        "train_state_json": metadata_dir / "train_state.json",
        "manifest_csv": metadata_dir / "manifest.csv",
        "chart_index_dir": data_dir / "chart_index",
        "dataset_dir": data_dir / "beat_aligned_dataset",
        "unpacked_dir": data_dir / "unpacked",
        "latest_checkpoint": checkpoints_dir / "latest.pt",
        "best_checkpoint": checkpoints_dir / "best.pt",
    }


def ensure_run_dirs(paths: dict[str, Path]) -> None:
    for key in ["run_dir", "metadata_dir", "data_dir", "checkpoints_dir", "chart_index_dir", "dataset_dir", "unpacked_dir"]:
        paths[key].mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python_random_state" in state:
        random.setstate(state["python_random_state"])
    if "numpy_random_state" in state:
        np.random.set_state(state["numpy_random_state"])
    if "torch_random_state" in state:
        torch.set_rng_state(state["torch_random_state"])
    if torch.cuda.is_available() and "torch_cuda_random_state_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_random_state_all"])


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    global_step: int,
    history: dict[str, Any],
    model,
    optimizer,
    scheduler,
    scaler,
    run_config: dict[str, Any],
    model_spec: dict[str, Any],
    input_shape: tuple[int, ...],
    splits: dict[str, list[str]],
    vocab_payload: dict[str, Any],
) -> None:
    payload = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "history": history,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "run_config": run_config,
        "model_spec": model_spec,
        "input_shape": tuple(int(x) for x in input_shape),
        "splits": splits,
        "vocab": vocab_payload,
        "rng_state": capture_rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path):
    return torch.load(Path(path), map_location="cpu")
