from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
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
from .wandb_utils import WandbConfig, setup_wandb_runtime

_ARTIFACT_ABS_PREFIX = "ABS::"
_INDEX_CACHE_SCHEMA_VERSION = 1


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


def _serialize_artifact_path(path: Path, data_root: Path) -> str:
    resolved = path.resolve()
    root_resolved = data_root.resolve()
    try:
        return str(resolved.relative_to(root_resolved))
    except ValueError:
        return f"{_ARTIFACT_ABS_PREFIX}{resolved}"


def _deserialize_artifact_path(path_value: str, data_root: Path) -> Path:
    if path_value.startswith(_ARTIFACT_ABS_PREFIX):
        return Path(path_value[len(_ARTIFACT_ABS_PREFIX):]).resolve()
    return (data_root / path_value).resolve()


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


def _stage_log(enabled: bool, stage: str, start_ts: float | None = None) -> float:
    if start_ts is None:
        if enabled:
            print(f"[startup] {stage}...")
        return time.perf_counter()
    elapsed = time.perf_counter() - start_ts
    if enabled:
        print(f"[startup] {stage} done in {elapsed:.2f}s")
    return elapsed


def _file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    st = path.stat()
    return {
        "exists": True,
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _dataset_index_signature(
    artifacts: TrainingArtifacts,
    training_spec: TrainingSpec,
    architecture_spec: ArchitectureSpec,
) -> str:
    payload = {
        "schema_version": _INDEX_CACHE_SCHEMA_VERSION,
        "chart_summary": _file_fingerprint(artifacts.chart_metadata_csv),
        "sequence_metadata": _file_fingerprint(artifacts.sequence_metadata_csv),
        "splits_file": _file_fingerprint(artifacts.splits_json),
        "architecture_mode": str(architecture_spec.name),
        "split_seed": int(training_spec.seed),
        "split_ratios": [
            float(training_spec.train_ratio),
            float(training_spec.val_ratio),
            float(training_spec.test_ratio),
        ],
        "data_root": str(artifacts.data_root),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _index_cache_paths(index_cache_dir: Path, signature: str) -> dict[str, Path]:
    cache_root = index_cache_dir / signature
    return {
        "root": cache_root,
        "meta_json": cache_root / "meta.json",
        "split_ids_json": cache_root / "split_ids.json",
        "manifest_pkl": cache_root / "manifest.pkl",
        "train_seq_pkl": cache_root / "train_seq_index.pkl",
        "val_seq_pkl": cache_root / "val_seq_index.pkl",
        "test_seq_pkl": cache_root / "test_seq_index.pkl",
    }


def _load_index_cache(cache_paths: dict[str, Path]) -> dict[str, Any] | None:
    required = [
        "meta_json",
        "split_ids_json",
        "manifest_pkl",
        "train_seq_pkl",
        "val_seq_pkl",
        "test_seq_pkl",
    ]
    if any(not cache_paths[key].exists() for key in required):
        return None
    try:
        meta = _load_json(cache_paths["meta_json"])
        if int(meta.get("schema_version", -1)) != _INDEX_CACHE_SCHEMA_VERSION:
            return None
        return {
            "manifest_df": pd.read_pickle(cache_paths["manifest_pkl"]),
            "split_ids": _load_json(cache_paths["split_ids_json"]),
            "train_seq_index": pd.read_pickle(cache_paths["train_seq_pkl"]),
            "val_seq_index": pd.read_pickle(cache_paths["val_seq_pkl"]),
            "test_seq_index": pd.read_pickle(cache_paths["test_seq_pkl"]),
        }
    except Exception:
        return None


def _save_index_cache(
    cache_paths: dict[str, Path],
    *,
    manifest_df,
    split_ids: dict[str, list[str]],
    train_seq_index,
    val_seq_index,
    test_seq_index,
) -> None:
    cache_paths["root"].mkdir(parents=True, exist_ok=True)
    _save_json(cache_paths["meta_json"], {"schema_version": _INDEX_CACHE_SCHEMA_VERSION})
    _save_json(cache_paths["split_ids_json"], split_ids)
    manifest_df.to_pickle(cache_paths["manifest_pkl"])
    train_seq_index.to_pickle(cache_paths["train_seq_pkl"])
    val_seq_index.to_pickle(cache_paths["val_seq_pkl"])
    test_seq_index.to_pickle(cache_paths["test_seq_pkl"])


def build_training_artifacts(data_root: str | Path, checkpoints_dir: str | Path | None = None) -> TrainingArtifacts:
    data_root = Path(data_root).resolve()
    training_dir = data_root / "training"
    resolved_checkpoints_dir = Path(checkpoints_dir).resolve() if checkpoints_dir is not None else (training_dir / "checkpoints")
    return TrainingArtifacts(
        data_root=data_root,
        audio_dir=data_root / "beat_aligned_dataset" / "audio_npz",
        token_dir=data_root / "beat_aligned_dataset" / "token_json",
        chart_metadata_csv=data_root / "chart_index" / "chart_build_summary.csv",
        sequence_metadata_csv=data_root / "beat_aligned_dataset" / "sequence_metadata.csv",
        training_dir=training_dir,
        splits_json=training_dir / "splits.json",
        vocab_json=training_dir / "vocab.json",
        checkpoints_dir=resolved_checkpoints_dir,
    )


def prepare_sample_data_artifacts(
    osz_inputs: Sequence[str | Path],
    data_root: str | Path,
    *,
    overwrite_unpack: bool = False,
    overwrite_parsed: bool = False,
    overwrite_dataset_outputs: bool = False,
    reject_offgrid_notes: bool = True,
    offgrid_tolerance_ms: float = 5.0,
    keep_only_max_notes_per_song: bool = False,
) -> TrainingArtifacts:
    prepare_training_data(
        osz_inputs=osz_inputs,
        data_root=data_root,
        overwrite_unpack=overwrite_unpack,
        overwrite_parsed=overwrite_parsed,
        overwrite_dataset_outputs=overwrite_dataset_outputs,
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
    *,
    use_index_cache: bool = True,
    index_cache_dir: str | Path | None = None,
    log_startup: bool = True,
    max_cached_charts: int | None = None,
) -> DatasetBundle:
    architecture_spec = architecture_spec or ArchitectureSpec()
    resolved_max_cached = (
        int(max_cached_charts)
        if max_cached_charts is not None
        else int(getattr(architecture_spec, "max_cached_charts", 4))
    )
    architecture_spec.max_cached_charts = max(1, resolved_max_cached)

    manifest_df = None
    split_ids = None
    train_seq_index = None
    val_seq_index = None
    test_seq_index = None
    cache_hit = False

    if use_index_cache:
        stage = _stage_log(log_startup, "index cache lookup")
        cache_root = Path(index_cache_dir).resolve() if index_cache_dir is not None else (artifacts.training_dir / "index_cache")
        signature = _dataset_index_signature(artifacts, training_spec, architecture_spec)
        cache_paths = _index_cache_paths(cache_root, signature)
        cached = _load_index_cache(cache_paths)
        if cached is not None:
            manifest_df = cached["manifest_df"]
            split_ids = cached["split_ids"]
            train_seq_index = cached["train_seq_index"]
            val_seq_index = cached["val_seq_index"]
            test_seq_index = cached["test_seq_index"]
            cache_hit = True
            if log_startup:
                print(f"[startup] index cache hit: {cache_paths['root']}")
        _stage_log(log_startup, "index cache lookup", stage)

    if not cache_hit:
        stage = _stage_log(log_startup, "manifest")
        manifest_df = build_chart_manifest(
            artifacts.audio_dir,
            artifacts.token_dir,
            chart_metadata_csv=artifacts.chart_metadata_csv,
            sequence_metadata_csv=artifacts.sequence_metadata_csv,
            prefer_metadata=True,
        )
        _stage_log(log_startup, "manifest", stage)
        if manifest_df.empty:
            raise RuntimeError("No training samples were found in the dataset artifacts.")

        stage = _stage_log(log_startup, "splits")
        split_ids = _load_or_create_splits(manifest_df, training_spec, artifacts.splits_json)
        _stage_log(log_startup, "splits", stage)

        stage = _stage_log(log_startup, "indexes")
        train_seq_index = build_sequence_index(
            manifest_df,
            split_ids["train"],
            sequence_metadata_csv=artifacts.sequence_metadata_csv,
            prefer_metadata=True,
        )
        val_seq_index = build_sequence_index(
            manifest_df,
            split_ids["val"],
            sequence_metadata_csv=artifacts.sequence_metadata_csv,
            prefer_metadata=True,
        )
        test_seq_index = build_sequence_index(
            manifest_df,
            split_ids["test"],
            sequence_metadata_csv=artifacts.sequence_metadata_csv,
            prefer_metadata=True,
        )
        _stage_log(log_startup, "indexes", stage)

        if use_index_cache:
            stage = _stage_log(log_startup, "index cache save")
            cache_root = Path(index_cache_dir).resolve() if index_cache_dir is not None else (artifacts.training_dir / "index_cache")
            signature = _dataset_index_signature(artifacts, training_spec, architecture_spec)
            cache_paths = _index_cache_paths(cache_root, signature)
            _save_index_cache(
                cache_paths,
                manifest_df=manifest_df,
                split_ids=split_ids,
                train_seq_index=train_seq_index,
                val_seq_index=val_seq_index,
                test_seq_index=test_seq_index,
            )
            _stage_log(log_startup, "index cache save", stage)

    if train_seq_index.empty or val_seq_index.empty:
        raise RuntimeError("Train/validation split produced no samples. Add more charts or adjust split ratios.")

    stage = _stage_log(log_startup, "vocab")
    vocab = _load_or_create_vocab(train_seq_index, val_seq_index, test_seq_index, artifacts.vocab_json)
    _stage_log(log_startup, "vocab", stage)
    token_to_id = vocab["token_to_id"]
    pad_id = int(token_to_id["PAD"])
    stage = _stage_log(log_startup, "dataset objects")
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
    _stage_log(log_startup, "dataset objects", stage)

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
    checkpoints_dir: str | Path | None = None,
    use_index_cache: bool = True,
    index_cache_dir: str | Path | None = None,
    log_startup: bool = True,
    max_cached_charts: int | None = None,
) -> TrainingContext:
    artifacts = build_training_artifacts(data_root, checkpoints_dir=checkpoints_dir)
    architecture_spec = architecture_spec or ArchitectureSpec()
    training_spec = training_spec or TrainingSpec(device=_default_device())

    _set_global_seed(training_spec.seed)
    dataset = create_dataset_bundle(
        artifacts,
        training_spec,
        architecture_spec,
        use_index_cache=use_index_cache,
        index_cache_dir=index_cache_dir,
        log_startup=log_startup,
        max_cached_charts=max_cached_charts,
    )

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
    checkpoints_dir: str | Path | None = None,
    use_index_cache: bool = True,
    index_cache_dir: str | Path | None = None,
    log_startup: bool = True,
    max_cached_charts: int | None = None,
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

    resolved_checkpoints_dir = None
    if checkpoints_dir is not None:
        resolved_checkpoints_dir = Path(checkpoints_dir).resolve()
    else:
        checkpoints_serialized = metadata.artifact_paths.get("checkpoints_dir")
        if checkpoints_serialized:
            resolved_checkpoints_dir = _deserialize_artifact_path(checkpoints_serialized, resolved_data_root)

    context = create_training_context(
        data_root=resolved_data_root,
        architecture_spec=architecture_spec,
        training_spec=training_spec,
        checkpoints_dir=resolved_checkpoints_dir,
        use_index_cache=use_index_cache,
        index_cache_dir=index_cache_dir,
        log_startup=log_startup,
        max_cached_charts=max_cached_charts,
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
    metrics_logger: Callable[[dict[str, Any]], None] | None = None,
    log_every_n_batches: int | None = None,
    wandb_config: WandbConfig | None = None,
) -> TrainingContext:
    wandb_runtime = None
    if metrics_logger is None and wandb_config is not None:
        wandb_runtime = setup_wandb_runtime(
            wandb_config,
            model_name=getattr(context.architecture_spec, "name", "taiko_model"),
        )
        metrics_logger = wandb_runtime.metrics_logger

    target_epochs = context.training_spec.epochs if epochs is None else int(epochs)
    context.training_spec.epochs = target_epochs

    try:
        for epoch in range(context.start_epoch, target_epochs + 1):
            train_stats = train_one_epoch(
                model=context.model,
                dataloader=context.dataset.train_loader,
                optimizer=context.optimizer,
                criterion=context.criterion,
                device=torch.device(context.training_spec.device),
                adherence_config=context.dataset.adherence_config,
                metrics_logger=metrics_logger,
                log_every_n_batches=log_every_n_batches,
                epoch=epoch,
                global_step_start=context.global_step,
            )
            val_stats = validate_one_epoch(
                model=context.model,
                dataloader=context.dataset.val_loader,
                criterion=context.criterion,
                device=torch.device(context.training_spec.device),
                adherence_config=context.dataset.adherence_config,
                metrics_logger=metrics_logger,
                log_every_n_batches=log_every_n_batches,
                epoch=epoch,
                global_step_start=context.global_step + len(context.dataset.train_loader),
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
                    "checkpoints_dir": _serialize_artifact_path(context.artifacts.checkpoints_dir, context.artifacts.data_root),
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

            best_updated = False
            if context.best_val_loss is None or val_stats["loss"] < context.best_val_loss:
                context.best_val_loss = float(val_stats["loss"])
                best_updated = True
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

            if metrics_logger is not None:
                payload: dict[str, Any] = {
                    "epoch": int(epoch),
                    "global_step": int(context.global_step),
                    "train/loss_epoch": float(train_stats["loss"]),
                    "val/loss_epoch": float(val_stats["loss"]),
                    "optimizer/lr": float(current_lr),
                    "checkpoint/last_path": str((context.artifacts.checkpoints_dir / "last.ckpt").resolve()),
                    "checkpoint/best_updated": int(best_updated),
                }
                if "density_proxy_abs_error" in train_stats:
                    payload["train/density_proxy_abs_error"] = float(train_stats["density_proxy_abs_error"])
                    payload["val/density_proxy_abs_error"] = float(val_stats["density_proxy_abs_error"])
                    payload["train/difficulty_proxy_drift"] = float(train_stats["difficulty_proxy_drift"])
                    payload["val/difficulty_proxy_drift"] = float(val_stats["difficulty_proxy_drift"])
                metrics_logger(payload)

            context.start_epoch = epoch + 1
    finally:
        if wandb_runtime is not None and wandb_runtime.run is not None:
            wandb_runtime.run.finish()

    return context
