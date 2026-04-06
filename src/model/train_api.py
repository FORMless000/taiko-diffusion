from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from src.preprocessing.prepare_training_data import prepare_training_data

from .checkpoints import CheckpointMetadata, load_checkpoint, restore_rng_states, save_checkpoint
from .data import (
    build_chart_manifest,
    build_dataset_for_spec,
    build_sequence_index,
    build_vocab_from_all_splits,
    split_chart_manifest,
)
from .factory import build_model
from .specs import ArchitectureSpec, TrainingSpec
from .trainer import train_one_epoch, validate_one_epoch


@dataclass
class TrainingArtifacts:
    data_root: Path
    audio_dir: Path
    token_dir: Path
    chart_metadata_csv: Path
    sequence_metadata_csv: Path
    training_dir: Path
    splits_json: Path
    vocab_json: Path
    checkpoints_dir: Path


@dataclass
class DatasetBundle:
    manifest_df: Any
    split_ids: dict[str, list[str]]
    vocab: dict[str, Any]
    train_seq_index: Any
    val_seq_index: Any
    test_seq_index: Any
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    pad_id: int
    label_ignore_index: int
    adherence_config: dict[str, Any]


@dataclass
class TrainingContext:
    artifacts: TrainingArtifacts
    architecture_spec: ArchitectureSpec
    training_spec: TrainingSpec
    dataset: DatasetBundle
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    criterion: torch.nn.Module
    scheduler: Any = None
    history: dict[str, list[float]] | None = None
    start_epoch: int = 1
    global_step: int = 0
    best_val_loss: float | None = None
    resume_checkpoint: Path | None = None


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _relative_to_root(path: Path, data_root: Path) -> str:
    return str(path.resolve().relative_to(data_root.resolve()))


def _build_history_template() -> dict[str, list[float]]:
    return {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "train_density_proxy_abs_error": [],
        "val_density_proxy_abs_error": [],
        "train_difficulty_proxy_drift": [],
        "val_difficulty_proxy_drift": [],
    }


def _append_history(history: dict[str, list[float]], train_stats: dict[str, Any], val_stats: dict[str, Any], lr: float) -> None:
    history["train_loss"].append(float(train_stats["loss"]))
    history["val_loss"].append(float(val_stats["loss"]))
    history["lr"].append(float(lr))

    if "density_proxy_abs_error" in train_stats:
        history["train_density_proxy_abs_error"].append(float(train_stats["density_proxy_abs_error"]))
        history["val_density_proxy_abs_error"].append(float(val_stats["density_proxy_abs_error"]))
        history["train_difficulty_proxy_drift"].append(float(train_stats["difficulty_proxy_drift"]))
        history["val_difficulty_proxy_drift"].append(float(val_stats["difficulty_proxy_drift"]))


def build_training_artifacts(data_root: str | Path) -> TrainingArtifacts:
    data_root = Path(data_root).resolve()
    training_dir = data_root / "training"
    return TrainingArtifacts(
        data_root=data_root,
        audio_dir=data_root / "beat_aligned_dataset" / "audio_npz",
        token_dir=data_root / "beat_aligned_dataset" / "token_json",
        chart_metadata_csv=data_root / "chart_index" / "chart_build_summary.csv",
        sequence_metadata_csv=data_root / "beat_aligned_dataset" / "sequence_metadata.csv",
        training_dir=training_dir,
        splits_json=training_dir / "splits.json",
        vocab_json=training_dir / "vocab.json",
        checkpoints_dir=training_dir / "checkpoints",
    )


def prepare_sample_data_artifacts(
    osz_inputs: Sequence[str | Path],
    data_root: str | Path,
    *,
    overwrite_unpack: bool = False,
    overwrite_parsed: bool = False,
    reject_offgrid_notes: bool = True,
    offgrid_tolerance_ms: float = 5.0,
    keep_only_max_notes_per_song: bool = False,
) -> TrainingArtifacts:
    prepare_training_data(
        osz_inputs=osz_inputs,
        data_root=data_root,
        overwrite_unpack=overwrite_unpack,
        overwrite_parsed=overwrite_parsed,
        reject_offgrid_notes=reject_offgrid_notes,
        offgrid_tolerance_ms=offgrid_tolerance_ms,
        keep_only_max_notes_per_song=keep_only_max_notes_per_song,
    )
    return build_training_artifacts(data_root)


def _load_or_create_splits(manifest_df, training_spec: TrainingSpec, splits_path: Path) -> dict[str, list[str]]:
    current_chart_ids = sorted(manifest_df["chart_id"].tolist())
    if splits_path.exists():
        split_ids = _load_json(splits_path)
        saved_chart_ids = sorted(split_ids["train"] + split_ids["val"] + split_ids["test"])
        if saved_chart_ids == current_chart_ids:
            return split_ids

    train_ids, val_ids, test_ids = split_chart_manifest(
        manifest_df,
        train_ratio=training_spec.train_ratio,
        val_ratio=training_spec.val_ratio,
        test_ratio=training_spec.test_ratio,
        random_state=training_spec.seed,
    )
    split_ids = {"train": list(train_ids), "val": list(val_ids), "test": list(test_ids)}
    _save_json(splits_path, split_ids)
    return split_ids


def _load_or_create_vocab(train_seq_index, val_seq_index, test_seq_index, vocab_path: Path) -> dict[str, Any]:
    if vocab_path.exists():
        saved = _load_json(vocab_path)
        token_to_id = {str(key): int(value) for key, value in saved["token_to_id"].items()}
        vocab_list = list(saved["vocab_list"])
        id_to_token = {int(idx): token for token, idx in token_to_id.items()}
        return {
            "vocab_list": vocab_list,
            "token_to_id": token_to_id,
            "id_to_token": id_to_token,
        }

    vocab_list, token_to_id, id_to_token = build_vocab_from_all_splits(
        train_seq_index,
        val_seq_index,
        test_seq_index,
    )
    _save_json(vocab_path, {"vocab_list": vocab_list, "token_to_id": token_to_id})
    return {
        "vocab_list": vocab_list,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
    }


def create_dataset_bundle(
    artifacts: TrainingArtifacts,
    training_spec: TrainingSpec,
    architecture_spec: ArchitectureSpec | None = None,
) -> DatasetBundle:
    architecture_spec = architecture_spec or ArchitectureSpec()
    manifest_df = build_chart_manifest(
        artifacts.audio_dir,
        artifacts.token_dir,
        chart_metadata_csv=artifacts.chart_metadata_csv,
    )
    if manifest_df.empty:
        raise RuntimeError("No training samples were found in the dataset artifacts.")

    split_ids = _load_or_create_splits(manifest_df, training_spec, artifacts.splits_json)
    train_seq_index = build_sequence_index(manifest_df, split_ids["train"])
    val_seq_index = build_sequence_index(manifest_df, split_ids["val"])
    test_seq_index = build_sequence_index(manifest_df, split_ids["test"])

    if train_seq_index.empty or val_seq_index.empty:
        raise RuntimeError("Train/validation split produced no samples. Add more charts or adjust split ratios.")

    vocab = _load_or_create_vocab(train_seq_index, val_seq_index, test_seq_index, artifacts.vocab_json)
    token_to_id = vocab["token_to_id"]
    pad_id = int(token_to_id["PAD"])
    train_dataset, collate_fn, label_ignore_index = build_dataset_for_spec(
        train_seq_index,
        token_to_id,
        architecture_spec,
    )
    val_dataset, _, _ = build_dataset_for_spec(
        val_seq_index,
        token_to_id,
        architecture_spec,
    )
    test_dataset, _, _ = build_dataset_for_spec(
        test_seq_index,
        token_to_id,
        architecture_spec,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_spec.batch_size,
        shuffle=True,
        num_workers=training_spec.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_spec.batch_size,
        shuffle=False,
        num_workers=training_spec.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_spec.batch_size,
        shuffle=False,
        num_workers=training_spec.num_workers,
        collate_fn=collate_fn,
    )

    adherence_config = {
        "ts_token_ids": [int(idx) for token, idx in token_to_id.items() if token.startswith("TS_")],
        "pad_id": pad_id,
        "ignore_index": label_ignore_index,
    }

    return DatasetBundle(
        manifest_df=manifest_df,
        split_ids=split_ids,
        vocab=vocab,
        train_seq_index=train_seq_index,
        val_seq_index=val_seq_index,
        test_seq_index=test_seq_index,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        pad_id=pad_id,
        label_ignore_index=label_ignore_index,
        adherence_config=adherence_config,
    )


def create_training_context(
    *,
    data_root: str | Path,
    architecture_spec: ArchitectureSpec | None = None,
    training_spec: TrainingSpec | None = None,
) -> TrainingContext:
    artifacts = build_training_artifacts(data_root)
    architecture_spec = architecture_spec or ArchitectureSpec()
    training_spec = training_spec or TrainingSpec(device=_default_device())

    _set_global_seed(training_spec.seed)
    dataset = create_dataset_bundle(artifacts, training_spec, architecture_spec)

    model = build_model(architecture_spec, vocab_size=len(dataset.vocab["token_to_id"]))
    device = torch.device(training_spec.device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_spec.lr,
        weight_decay=training_spec.weight_decay,
    )
    criterion = CrossEntropyLoss(ignore_index=dataset.label_ignore_index)

    return TrainingContext(
        artifacts=artifacts,
        architecture_spec=architecture_spec,
        training_spec=training_spec,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=None,
        history=_build_history_template(),
        start_epoch=1,
        global_step=0,
        best_val_loss=None,
        resume_checkpoint=None,
    )


def load_training_context_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    data_root: str | Path | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> TrainingContext:
    checkpoint_path = Path(checkpoint_path).resolve()
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    metadata = CheckpointMetadata.from_dict(payload["metadata"])

    resolved_data_root = Path(data_root).resolve() if data_root is not None else Path(metadata.data_root).resolve()
    architecture_spec = ArchitectureSpec.from_dict(payload["architecture_spec"])
    training_spec = TrainingSpec.from_dict(payload["training_spec"])

    if device is not None:
        training_spec.device = device
    if batch_size is not None:
        training_spec.batch_size = batch_size
    if num_workers is not None:
        training_spec.num_workers = num_workers

    context = create_training_context(
        data_root=resolved_data_root,
        architecture_spec=architecture_spec,
        training_spec=training_spec,
    )

    context.model.load_state_dict(payload["model_state_dict"])
    context.optimizer.load_state_dict(payload["optimizer_state_dict"])
    if context.scheduler is not None and payload.get("scheduler_state_dict") is not None:
        context.scheduler.load_state_dict(payload["scheduler_state_dict"])
    restore_rng_states(payload.get("rng_state"))

    context.history = payload.get("history", _build_history_template())
    context.start_epoch = int(metadata.epoch) + 1
    context.global_step = int(metadata.global_step)
    context.best_val_loss = metadata.best_val_loss
    context.resume_checkpoint = checkpoint_path
    context.dataset.split_ids = payload["split_ids"]
    context.dataset.vocab = payload["vocab"]
    context.dataset.adherence_config = payload.get("adherence_config", context.dataset.adherence_config)
    return context


def train_context(
    context: TrainingContext,
    *,
    epochs: int | None = None,
) -> TrainingContext:
    target_epochs = context.training_spec.epochs if epochs is None else int(epochs)
    context.training_spec.epochs = target_epochs

    for epoch in range(context.start_epoch, target_epochs + 1):
        train_stats = train_one_epoch(
            model=context.model,
            dataloader=context.dataset.train_loader,
            optimizer=context.optimizer,
            criterion=context.criterion,
            device=torch.device(context.training_spec.device),
            adherence_config=context.dataset.adherence_config,
        )
        val_stats = validate_one_epoch(
            model=context.model,
            dataloader=context.dataset.val_loader,
            criterion=context.criterion,
            device=torch.device(context.training_spec.device),
            adherence_config=context.dataset.adherence_config,
        )

        current_lr = context.optimizer.param_groups[0]["lr"]
        _append_history(context.history, train_stats, val_stats, current_lr)
        context.global_step += len(context.dataset.train_loader)

        print(
            f"Epoch {epoch}/{target_epochs} | lr: {current_lr:.6f} | "
            f"train loss: {train_stats['loss']:.4f} | val loss: {val_stats['loss']:.4f}"
        )

        current_best = val_stats["loss"] if context.best_val_loss is None else min(context.best_val_loss, val_stats["loss"])
        metadata = CheckpointMetadata(
            epoch=epoch,
            global_step=context.global_step,
            best_val_loss=float(current_best),
            data_root=str(context.artifacts.data_root),
            artifact_paths={
                "audio_dir": _relative_to_root(context.artifacts.audio_dir, context.artifacts.data_root),
                "token_dir": _relative_to_root(context.artifacts.token_dir, context.artifacts.data_root),
                "chart_metadata_csv": _relative_to_root(context.artifacts.chart_metadata_csv, context.artifacts.data_root),
                "sequence_metadata_csv": _relative_to_root(context.artifacts.sequence_metadata_csv, context.artifacts.data_root),
                "splits_json": _relative_to_root(context.artifacts.splits_json, context.artifacts.data_root),
                "vocab_json": _relative_to_root(context.artifacts.vocab_json, context.artifacts.data_root),
                "checkpoints_dir": _relative_to_root(context.artifacts.checkpoints_dir, context.artifacts.data_root),
            },
        )

        save_checkpoint(
            context.artifacts.checkpoints_dir / "last.ckpt",
            model=context.model,
            optimizer=context.optimizer,
            scheduler=context.scheduler,
            architecture_spec=context.architecture_spec,
            training_spec=context.training_spec,
            metadata=metadata,
            history=context.history,
            vocab=context.dataset.vocab,
            split_ids=context.dataset.split_ids,
            adherence_config=context.dataset.adherence_config,
        )

        if context.best_val_loss is None or val_stats["loss"] < context.best_val_loss:
            context.best_val_loss = float(val_stats["loss"])
            best_metadata = CheckpointMetadata(
                epoch=epoch,
                global_step=context.global_step,
                best_val_loss=context.best_val_loss,
                data_root=str(context.artifacts.data_root),
                artifact_paths=metadata.artifact_paths,
            )
            save_checkpoint(
                context.artifacts.checkpoints_dir / "best.ckpt",
                model=context.model,
                optimizer=context.optimizer,
                scheduler=context.scheduler,
                architecture_spec=context.architecture_spec,
                training_spec=context.training_spec,
                metadata=best_metadata,
                history=context.history,
                vocab=context.dataset.vocab,
                split_ids=context.dataset.split_ids,
                adherence_config=context.dataset.adherence_config,
            )

        context.start_epoch = epoch + 1

    return context
