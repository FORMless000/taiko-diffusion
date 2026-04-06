from __future__ import annotations

import copy
import random
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.model.data import (
    TaikoDataset,
    build_chart_manifest,
    build_sequence_index,
    build_vocab_from_all_splits,
    split_chart_manifest,
    taiko_collate_fn,
)
from src.model.factory import build_model
from src.model.trainer import train_one_epoch, validate_one_epoch
from src.preprocessing.beat_aligned import run_pipeline
from src.preprocessing.osutaiko_parser import parse_unpacked_taiko_charts
from src.preprocessing.unpack_osz import unpack_osz_paths

from .checkpointing import (
    ensure_run_dirs,
    get_run_paths,
    load_checkpoint,
    read_json,
    restore_rng_state,
    save_checkpoint,
    write_json,
)
from .config import TrainingRunConfig


def _resolve_raw_osz_paths(raw_osz_paths: list[str]) -> list[str]:
    resolved = []
    for path_str in raw_osz_paths:
        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Raw .osz file not found: {path}")
        resolved.append(str(path))
    deduped = sorted(dict.fromkeys(resolved))
    if not deduped:
        raise ValueError("At least one raw .osz path is required for a fresh training run.")
    return deduped


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device(device_name: str | None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compare_configs_for_resume(current: TrainingRunConfig, stored: TrainingRunConfig) -> TrainingRunConfig:
    merged = copy.deepcopy(stored)

    if current.raw_osz_paths:
        current_paths = _resolve_raw_osz_paths(current.raw_osz_paths)
        stored_paths = _resolve_raw_osz_paths(stored.raw_osz_paths)
        if current_paths != stored_paths:
            raise ValueError("Resume raw_osz_paths do not match the checkpoint configuration.")

    if current.model_spec.to_dict() != stored.model_spec.to_dict():
        raise ValueError("Resume model_spec does not match the checkpoint configuration.")

    if current.preprocessing != stored.preprocessing:
        raise ValueError("Resume preprocessing config does not match the checkpoint configuration.")

    if current.split != stored.split:
        raise ValueError("Resume split config does not match the checkpoint configuration.")

    merged.run_dir = current.run_dir or stored.run_dir
    merged.resume_checkpoint = current.resume_checkpoint or stored.resume_checkpoint
    merged.device = current.device or stored.device
    merged.optimization.batch_size = current.optimization.batch_size or stored.optimization.batch_size
    merged.optimization.num_workers = current.optimization.num_workers
    merged.optimization.pin_memory = current.optimization.pin_memory
    merged.optimization.num_epochs = max(stored.optimization.num_epochs, current.optimization.num_epochs)
    merged.optimization.checkpoint_every_epochs = current.optimization.checkpoint_every_epochs or stored.optimization.checkpoint_every_epochs
    merged.optimization.use_amp = current.optimization.use_amp
    return merged


def _maybe_write_config(paths: dict[str, Path], config: TrainingRunConfig) -> None:
    payload = config.to_dict()
    payload["run_dir"] = "."
    payload["resume_checkpoint"] = None
    payload["device"] = None
    existing = read_json(paths["config_json"])
    if existing is not None and existing != payload:
        raise ValueError(f"Run directory already contains a different config: {paths['config_json']}")
    write_json(paths["config_json"], payload)


def _prepare_dataset(config: TrainingRunConfig, paths: dict[str, Path]) -> None:
    resolved_raw_paths = _resolve_raw_osz_paths(config.raw_osz_paths)
    write_json(
        paths["raw_manifest_json"],
        {
            "raw_osz_paths": resolved_raw_paths,
        },
    )

    unpack_osz_paths(
        source_paths=resolved_raw_paths,
        destination_root=paths["unpacked_dir"],
        overwrite=config.preprocessing.overwrite_unpacked,
        keep_only_chart_and_audio=True,
    )

    parse_unpacked_taiko_charts(
        unpacked_root=paths["unpacked_dir"],
        include_bpm_events=config.preprocessing.include_bpm_events,
        overwrite=config.preprocessing.overwrite_parsed,
    )

    run_pipeline(
        unpacked_root=paths["unpacked_dir"],
        index_dir=paths["chart_index_dir"],
        dataset_dir=paths["dataset_dir"],
        reject_offgrid_notes=config.preprocessing.reject_offgrid_notes,
        offgrid_tolerance_ms=config.preprocessing.offgrid_tolerance_ms,
        keep_only_max_notes_per_song=config.preprocessing.keep_only_max_notes_per_song,
    )


def _build_or_load_manifest(paths: dict[str, Path]) -> pd.DataFrame:
    manifest_csv = paths["manifest_csv"]
    if manifest_csv.exists():
        return pd.read_csv(manifest_csv)

    manifest_df = build_chart_manifest(
        audio_dir=paths["dataset_dir"] / "audio_npz",
        token_dir=paths["dataset_dir"] / "token_json",
        chart_metadata_csv=paths["chart_index_dir"] / "chart_build_summary.csv",
    )
    manifest_df.to_csv(manifest_csv, index=False)
    return manifest_df


def _build_or_load_splits(manifest_df: pd.DataFrame, config: TrainingRunConfig, paths: dict[str, Path]):
    payload = read_json(paths["splits_json"])
    if payload is not None:
        return payload

    train_ids, val_ids, test_ids = split_chart_manifest(
        manifest_df,
        train_ratio=config.split.train_ratio,
        val_ratio=config.split.val_ratio,
        test_ratio=config.split.test_ratio,
        random_state=config.split.random_state,
    )
    payload = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
    }
    write_json(paths["splits_json"], payload)
    return payload


def _build_or_load_vocab(manifest_df: pd.DataFrame, splits: dict[str, list[str]], paths: dict[str, Path]):
    payload = read_json(paths["vocab_json"])
    if payload is not None:
        token_to_id = {str(k): int(v) for k, v in payload["token_to_id"].items()}
        id_to_token = {int(k): str(v) for k, v in payload["id_to_token"].items()}
        payload["token_to_id"] = token_to_id
        payload["id_to_token"] = id_to_token
        return payload

    train_seq_index = build_sequence_index(manifest_df, splits["train_ids"])
    val_seq_index = build_sequence_index(manifest_df, splits["val_ids"])
    test_seq_index = build_sequence_index(manifest_df, splits["test_ids"])

    vocab_list, token_to_id, id_to_token = build_vocab_from_all_splits(
        train_seq_index,
        val_seq_index,
        test_seq_index,
    )
    payload = {
        "vocab_list": vocab_list,
        "token_to_id": token_to_id,
        "id_to_token": {str(k): v for k, v in id_to_token.items()},
    }
    write_json(paths["vocab_json"], payload)
    payload["id_to_token"] = {int(k): v for k, v in payload["id_to_token"].items()}
    return payload


def _write_dataset_info(manifest_df: pd.DataFrame, paths: dict[str, Path]) -> dict[str, Any]:
    existing = read_json(paths["dataset_info_json"])
    if existing is not None:
        return existing

    if manifest_df.empty:
        raise RuntimeError("Manifest is empty; cannot derive input shape.")

    sample_npz_path = Path(manifest_df.iloc[0]["npz_path"])
    audio_sequences = np.load(sample_npz_path)["audio_sequences"]
    input_shape = list(audio_sequences.shape[1:])
    payload = {
        "input_shape": input_shape,
        "num_charts": int(len(manifest_df)),
        "num_sequences": int(manifest_df["n_sequences_audio"].sum()),
    }
    write_json(paths["dataset_info_json"], payload)
    return payload


def _build_dataloaders(
    manifest_df: pd.DataFrame,
    splits: dict[str, list[str]],
    token_to_id: dict[str, int],
    config: TrainingRunConfig,
):
    train_seq_index = build_sequence_index(manifest_df, splits["train_ids"])
    val_seq_index = build_sequence_index(manifest_df, splits["val_ids"])
    test_seq_index = build_sequence_index(manifest_df, splits["test_ids"])

    train_dataset = TaikoDataset(train_seq_index, token_to_id)
    val_dataset = TaikoDataset(val_seq_index, token_to_id)
    test_dataset = TaikoDataset(test_seq_index, token_to_id)

    collate = partial(taiko_collate_fn, pad_id=token_to_id["PAD"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.optimization.batch_size,
        shuffle=True,
        num_workers=config.optimization.num_workers,
        pin_memory=config.optimization.pin_memory,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.optimization.batch_size,
        shuffle=False,
        num_workers=config.optimization.num_workers,
        pin_memory=config.optimization.pin_memory,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.optimization.batch_size,
        shuffle=False,
        num_workers=config.optimization.num_workers,
        pin_memory=config.optimization.pin_memory,
        collate_fn=collate,
    )

    return train_loader, val_loader, test_loader


def _build_scheduler(optimizer, config: TrainingRunConfig):
    if config.optimization.scheduler_name == "none":
        return None
    if config.optimization.scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.optimization.scheduler_step_size,
            gamma=config.optimization.scheduler_gamma,
        )
    raise ValueError(f"Unsupported scheduler: {config.optimization.scheduler_name}")


def _empty_history() -> dict[str, list[float]]:
    return {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "train_density_proxy_abs_error": [],
        "val_density_proxy_abs_error": [],
        "train_difficulty_proxy_drift": [],
        "val_difficulty_proxy_drift": [],
    }


def _append_history(history: dict[str, list[float]], train_stats, val_stats, lr: float) -> None:
    history["train_loss"].append(float(train_stats["loss"]))
    history["val_loss"].append(float(val_stats["loss"]))
    history["lr"].append(float(lr))

    if "density_proxy_abs_error" in train_stats:
        history["train_density_proxy_abs_error"].append(float(train_stats["density_proxy_abs_error"]))
        history["val_density_proxy_abs_error"].append(float(val_stats["density_proxy_abs_error"]))
        history["train_difficulty_proxy_drift"].append(float(train_stats["difficulty_proxy_drift"]))
        history["val_difficulty_proxy_drift"].append(float(val_stats["difficulty_proxy_drift"]))


def train_from_raw_osz(config: TrainingRunConfig):
    if config.resume_checkpoint:
        checkpoint = load_checkpoint(config.resume_checkpoint)
        stored_config = TrainingRunConfig.from_dict(checkpoint["run_config"])
        config = _compare_configs_for_resume(config, stored_config)
    else:
        checkpoint = None

    paths = get_run_paths(config.run_dir)
    ensure_run_dirs(paths)
    _maybe_write_config(paths, config)

    if not paths["manifest_csv"].exists():
        _prepare_dataset(config, paths)

    manifest_df = _build_or_load_manifest(paths)
    splits = _build_or_load_splits(manifest_df, config, paths)
    vocab_payload = _build_or_load_vocab(manifest_df, splits, paths)
    dataset_info = _write_dataset_info(manifest_df, paths)

    write_json(
        paths["train_state_json"],
        {
            "manifest_csv": str(paths["manifest_csv"]),
            "splits_json": str(paths["splits_json"]),
            "vocab_json": str(paths["vocab_json"]),
            "dataset_info_json": str(paths["dataset_info_json"]),
        },
    )

    token_to_id = vocab_payload["token_to_id"]
    id_to_token = vocab_payload["id_to_token"]
    input_shape = tuple(int(x) for x in dataset_info["input_shape"])

    _set_seed(config.optimization.seed)
    device = _select_device(config.device)

    train_loader, val_loader, test_loader = _build_dataloaders(manifest_df, splits, token_to_id, config)
    model = build_model(config.model_spec, vocab_size=len(token_to_id), input_shape=input_shape).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, config)
    scaler = torch.cuda.amp.GradScaler(enabled=config.optimization.use_amp and device.type == "cuda")
    criterion = torch.nn.CrossEntropyLoss(ignore_index=token_to_id["PAD"])
    adherence_config = {
        "ts_token_ids": [token_id for token, token_id in token_to_id.items() if token.startswith("TS_")],
        "pad_id": token_to_id["PAD"],
    }

    history = _empty_history()
    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = checkpoint.get("history", history)
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")
        restore_rng_state(checkpoint.get("rng_state"))

    if start_epoch > config.optimization.num_epochs:
        return {
            "run_dir": str(paths["run_dir"]),
            "latest_checkpoint": str(paths["latest_checkpoint"]) if paths["latest_checkpoint"].exists() else None,
            "best_checkpoint": str(paths["best_checkpoint"]) if paths["best_checkpoint"].exists() else None,
            "history": history,
            "test_loader_size": len(test_loader),
        }

    for epoch in range(start_epoch, config.optimization.num_epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            adherence_config=adherence_config,
            scaler=scaler,
            use_amp=config.optimization.use_amp,
        )
        val_stats = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            adherence_config=adherence_config,
            use_amp=config.optimization.use_amp,
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        _append_history(history, train_stats, val_stats, current_lr)
        global_step += len(train_loader)

        if scheduler is not None:
            scheduler.step()

        write_json(paths["history_json"], history)

        save_checkpoint(
            paths["latest_checkpoint"],
            epoch=epoch,
            global_step=global_step,
            history=history,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            run_config=config.to_dict(),
            model_spec=config.model_spec.to_dict(),
            input_shape=input_shape,
            splits=splits,
            vocab_payload={
                "vocab_list": vocab_payload["vocab_list"],
                "token_to_id": token_to_id,
                "id_to_token": {str(k): v for k, v in id_to_token.items()},
            },
        )

        if epoch % max(1, config.optimization.checkpoint_every_epochs) == 0:
            save_checkpoint(
                paths["checkpoints_dir"] / f"epoch_{epoch:04d}.pt",
                epoch=epoch,
                global_step=global_step,
                history=history,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                run_config=config.to_dict(),
                model_spec=config.model_spec.to_dict(),
                input_shape=input_shape,
                splits=splits,
                vocab_payload={
                    "vocab_list": vocab_payload["vocab_list"],
                    "token_to_id": token_to_id,
                    "id_to_token": {str(k): v for k, v in id_to_token.items()},
                },
            )

        if float(val_stats["loss"]) <= best_val_loss:
            best_val_loss = float(val_stats["loss"])
            save_checkpoint(
                paths["best_checkpoint"],
                epoch=epoch,
                global_step=global_step,
                history=history,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                run_config=config.to_dict(),
                model_spec=config.model_spec.to_dict(),
                input_shape=input_shape,
                splits=splits,
                vocab_payload={
                    "vocab_list": vocab_payload["vocab_list"],
                    "token_to_id": token_to_id,
                    "id_to_token": {str(k): v for k, v in id_to_token.items()},
                },
            )

        print(
            f"Epoch {epoch}/{config.optimization.num_epochs} | lr: {current_lr:.6f} | "
            f"train loss: {train_stats['loss']:.4f} | val loss: {val_stats['loss']:.4f}"
        )

    return {
        "run_dir": str(paths["run_dir"]),
        "latest_checkpoint": str(paths["latest_checkpoint"]),
        "best_checkpoint": str(paths["best_checkpoint"]) if paths["best_checkpoint"].exists() else None,
        "history": history,
        "test_loader_size": len(test_loader),
    }
