from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

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


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


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


def _artifact_paths_for_root(data_root: Path, checkpoints_dir: Path | None = None) -> dict[str, Path]:
    training_dir = data_root / "training"
    return {
        "audio_dir": data_root / "beat_aligned_dataset" / "audio_npz",
        "token_dir": data_root / "beat_aligned_dataset" / "token_json",
        "chart_metadata_csv": data_root / "chart_index" / "chart_build_summary.csv",
        "sequence_metadata_csv": data_root / "beat_aligned_dataset" / "sequence_metadata.csv",
        "splits_json": training_dir / "splits.json",
        "vocab_json": training_dir / "vocab.json",
        "checkpoints_dir": checkpoints_dir.resolve() if checkpoints_dir is not None else (training_dir / "checkpoints"),
    }


def _artifact_paths_to_relative(artifact_paths: dict[str, Path], data_root: Path) -> dict[str, str]:
    return {key: _serialize_artifact_path(path, data_root) for key, path in artifact_paths.items()}


def _artifact_paths_from_checkpoint(data_root: Path, checkpoint_payload: dict[str, Any] | None) -> dict[str, Path]:
    if checkpoint_payload is None:
        return _artifact_paths_for_root(data_root)

    metadata = CheckpointMetadata.from_dict(checkpoint_payload["metadata"])
    if not metadata.artifact_paths:
        return _artifact_paths_for_root(data_root)

    return {key: _deserialize_artifact_path(path_value, data_root) for key, path_value in metadata.artifact_paths.items()}


def _resolve_data_root(args: argparse.Namespace, checkpoint_payload: dict[str, Any] | None) -> Path:
    if args.data_root:
        return Path(args.data_root).resolve()

    if checkpoint_payload is None:
        raise ValueError("--data-root is required unless resuming from a checkpoint with a valid saved data root.")

    metadata = CheckpointMetadata.from_dict(checkpoint_payload["metadata"])
    if metadata.data_root:
        candidate = Path(metadata.data_root).resolve()
        if candidate.exists():
            return candidate

    raise ValueError("Could not resolve data root from checkpoint. Pass --data-root explicitly.")


def _require_dataset_artifacts(artifact_paths: dict[str, Path]) -> None:
    required = ["audio_dir", "token_dir", "chart_metadata_csv"]
    missing = [name for name in required if not artifact_paths[name].exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Required dataset artifacts are missing: {joined}")


def _load_or_create_splits(
    manifest_df,
    training_spec: TrainingSpec,
    splits_path: Path,
    checkpoint_payload: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    if checkpoint_payload is not None:
        split_ids = checkpoint_payload["split_ids"]
        _save_json(splits_path, split_ids)
        return split_ids

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
    split_ids = {
        "train": list(train_ids),
        "val": list(val_ids),
        "test": list(test_ids),
    }
    _save_json(splits_path, split_ids)
    return split_ids


def _load_or_create_vocab(
    train_seq_index,
    val_seq_index,
    test_seq_index,
    vocab_path: Path,
    checkpoint_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if checkpoint_payload is not None:
        vocab = checkpoint_payload["vocab"]
        serializable = {
            "vocab_list": vocab["vocab_list"],
            "token_to_id": vocab["token_to_id"],
        }
        _save_json(vocab_path, serializable)
        return vocab

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
    _save_json(
        vocab_path,
        {
            "vocab_list": vocab_list,
            "token_to_id": token_to_id,
        },
    )
    return {
        "vocab_list": vocab_list,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
    }


def _build_architecture_spec(args: argparse.Namespace, checkpoint_payload: dict[str, Any] | None) -> ArchitectureSpec:
    if checkpoint_payload is not None:
        return ArchitectureSpec.from_dict(checkpoint_payload["architecture_spec"])

    return ArchitectureSpec(
        name=args.architecture_name,
        input_dim=args.input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=args.max_len,
        history_max_tokens=args.history_max_tokens,
        retrieval_top_k=args.retrieval_top_k,
        retrieval_max_tokens_per_window=args.retrieval_max_tokens_per_window,
        retrieval_exclude_last_n_windows=args.retrieval_exclude_last_n_windows,
        use_motif_retrieval=args.use_motif_retrieval,
    )


def _build_training_spec(args: argparse.Namespace, checkpoint_payload: dict[str, Any] | None) -> TrainingSpec:
    if checkpoint_payload is None:
        return TrainingSpec(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            num_workers=args.num_workers,
        )

    saved = TrainingSpec.from_dict(checkpoint_payload["training_spec"])
    # Resume defaults to the checkpointed training state. Only allow safe runtime
    # overrides that do not silently desynchronize saved optimizer/split state.
    saved.epochs = args.epochs
    saved.batch_size = args.batch_size
    saved.device = args.device
    saved.num_workers = args.num_workers
    return saved


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the taiko model from raw .osz files or existing dataset artifacts.")
    parser.add_argument("raw_osz", nargs="*", help="Raw .osz files, directories, or glob patterns.")
    parser.add_argument("--data-root", help="Dataset root containing unpacked and beat-aligned artifacts.")
    parser.add_argument("--resume-checkpoint", help="Resume training from a saved checkpoint.")
    parser.add_argument("--checkpoints-dir", help="Optional checkpoint output directory override.")
    parser.add_argument("--epochs", type=int, default=50, help="Total number of epochs to train to.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default=_default_device(), help="Torch device string.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--architecture-name", default="taiko_transformer", help="Registered architecture name.")
    parser.add_argument("--input-dim", type=int, default=128, help="Audio input dimension.")
    parser.add_argument("--d-model", type=int, default=256, help="Transformer hidden size.")
    parser.add_argument("--nhead", type=int, default=4, help="Attention head count.")
    parser.add_argument("--num-encoder-layers", type=int, default=4, help="Encoder layer count.")
    parser.add_argument("--num-decoder-layers", type=int, default=4, help="Decoder layer count.")
    parser.add_argument("--dim-feedforward", type=int, default=1024, help="Transformer feedforward width.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout.")
    parser.add_argument("--max-len", type=int, default=512, help="Maximum modeled sequence length.")
    parser.add_argument("--history-max-tokens", type=int, default=1024, help="Recent exact-history budget for the context transformer.")
    parser.add_argument("--retrieval-top-k", type=int, default=2, help="How many prior windows to retrieve for motif reuse.")
    parser.add_argument("--retrieval-max-tokens-per-window", type=int, default=64, help="Maximum retrieved tokens to prepend for each prior window.")
    parser.add_argument("--retrieval-exclude-last-n-windows", type=int, default=2, help="Skip the most recent windows when retrieving repeated motifs.")
    parser.add_argument("--use-motif-retrieval", action=argparse.BooleanOptionalAction, default=True, help="Enable audio-similarity motif retrieval for the context transformer.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-run-name", default="default", help="Run-name tag used in the W&B run name.")
    parser.add_argument("--wandb-log-every-batches", type=int, default=100, help="Log batch metrics to W&B every N batches.")
    parser.add_argument("--wandb-notebook-name", default="", help="Notebook name for W&B code saving (sets WANDB_NOTEBOOK_NAME).")
    parser.add_argument("--wandb-offline", action="store_true", help="Run W&B in offline mode without online auth.")
    parser.add_argument("--wandb-api-key", default="", help="Optional W&B API key passed at runtime (without env setup).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    checkpoint_payload = None
    if args.resume_checkpoint:
        checkpoint_payload = load_checkpoint(args.resume_checkpoint, map_location="cpu")

    data_root = _resolve_data_root(args, checkpoint_payload)
    checkpoints_override = Path(args.checkpoints_dir).resolve() if args.checkpoints_dir else None
    artifact_paths = _artifact_paths_from_checkpoint(data_root, checkpoint_payload)
    if checkpoint_payload is None:
        artifact_paths = _artifact_paths_for_root(data_root, checkpoints_dir=checkpoints_override)
    elif checkpoints_override is not None:
        artifact_paths["checkpoints_dir"] = checkpoints_override

    if args.raw_osz:
        from src.preprocessing.prepare_training_data import prepare_training_data

        prepare_training_data(
            osz_inputs=args.raw_osz,
            data_root=data_root,
        )

    _require_dataset_artifacts(artifact_paths)

    training_spec = _build_training_spec(args, checkpoint_payload)
    _set_global_seed(training_spec.seed)

    manifest_df = build_chart_manifest(
        artifact_paths["audio_dir"],
        artifact_paths["token_dir"],
        chart_metadata_csv=artifact_paths["chart_metadata_csv"],
    )
    if manifest_df.empty:
        raise RuntimeError("No training samples were found in the dataset artifacts.")

    split_ids = _load_or_create_splits(
        manifest_df,
        training_spec=training_spec,
        splits_path=artifact_paths["splits_json"],
        checkpoint_payload=checkpoint_payload,
    )

    train_seq_index = build_sequence_index(manifest_df, split_ids["train"])
    val_seq_index = build_sequence_index(manifest_df, split_ids["val"])
    test_seq_index = build_sequence_index(manifest_df, split_ids["test"])
    if train_seq_index.empty or val_seq_index.empty:
        raise RuntimeError("Train/validation split produced no samples. Add more charts or adjust split ratios.")

    vocab = _load_or_create_vocab(
        train_seq_index,
        val_seq_index,
        test_seq_index,
        vocab_path=artifact_paths["vocab_json"],
        checkpoint_payload=checkpoint_payload,
    )
    token_to_id = vocab["token_to_id"]

    pad_id = int(token_to_id["PAD"])
    architecture_spec = _build_architecture_spec(args, checkpoint_payload)
    wandb_runtime = setup_wandb_runtime(
        WandbConfig(
            enabled=bool(args.wandb),
            run_name=str(args.wandb_run_name),
            log_every_n_batches=int(args.wandb_log_every_batches),
            notebook_name=str(args.wandb_notebook_name),
            offline=bool(args.wandb_offline),
            api_key=str(args.wandb_api_key),
            mode_name_for_run=architecture_spec.name,
        ),
        model_name=architecture_spec.name,
    )
    metrics_logger = wandb_runtime.metrics_logger

    train_dataset, collate, label_ignore_index = build_dataset_for_spec(
        train_seq_index,
        token_to_id,
        architecture_spec,
    )
    val_dataset, _, _ = build_dataset_for_spec(
        val_seq_index,
        token_to_id,
        architecture_spec,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_spec.batch_size,
        shuffle=True,
        num_workers=training_spec.num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_spec.batch_size,
        shuffle=False,
        num_workers=training_spec.num_workers,
        collate_fn=collate,
    )

    model = build_model(architecture_spec, vocab_size=len(token_to_id))
    device = torch.device(training_spec.device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_spec.lr,
        weight_decay=training_spec.weight_decay,
    )
    scheduler = None
    criterion = CrossEntropyLoss(ignore_index=label_ignore_index)

    history = _build_history_template()
    start_epoch = 1
    global_step = 0
    best_val_loss = None
    adherence_config = {
        "ts_token_ids": [int(idx) for token, idx in token_to_id.items() if token.startswith("TS_")],
        "pad_id": pad_id,
        "ignore_index": label_ignore_index,
    }

    if checkpoint_payload is not None:
        model.load_state_dict(checkpoint_payload["model_state_dict"])
        optimizer.load_state_dict(checkpoint_payload["optimizer_state_dict"])
        if scheduler is not None and checkpoint_payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint_payload["scheduler_state_dict"])
        restore_rng_states(checkpoint_payload.get("rng_state"))

        history = checkpoint_payload.get("history", history)
        metadata = CheckpointMetadata.from_dict(checkpoint_payload["metadata"])
        start_epoch = int(metadata.epoch) + 1
        global_step = int(metadata.global_step)
        best_val_loss = metadata.best_val_loss
        adherence_config = checkpoint_payload.get("adherence_config", adherence_config)

    checkpoints_dir = artifact_paths["checkpoints_dir"]
    relative_artifacts = _artifact_paths_to_relative(artifact_paths, data_root)

    if start_epoch > training_spec.epochs:
        print(
            f"Checkpoint already at epoch {start_epoch - 1}, which is >= requested total epochs {training_spec.epochs}. "
            "Nothing to do."
        )
        if wandb_runtime.run is not None:
            wandb_runtime.run.finish()
        return 0

    for epoch in range(start_epoch, training_spec.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            adherence_config=adherence_config,
            metrics_logger=metrics_logger,
            log_every_n_batches=args.wandb_log_every_batches,
            epoch=epoch,
            global_step_start=global_step,
        )
        val_stats = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            adherence_config=adherence_config,
            metrics_logger=metrics_logger,
            log_every_n_batches=args.wandb_log_every_batches,
            epoch=epoch,
            global_step_start=global_step + len(train_loader),
        )

        current_lr = optimizer.param_groups[0]["lr"]
        _append_history(history, train_stats, val_stats, current_lr)
        global_step += len(train_loader)

        print(
            f"Epoch {epoch}/{training_spec.epochs} | lr: {current_lr:.6f} | "
            f"train loss: {train_stats['loss']:.4f} | val loss: {val_stats['loss']:.4f}"
        )
        if "density_proxy_abs_error" in train_stats:
            print(
                f"  adherence | train density err: {train_stats['density_proxy_abs_error']:.3f} | "
                f"val density err: {val_stats['density_proxy_abs_error']:.3f} | "
                f"train difficulty drift: {train_stats['difficulty_proxy_drift']:.3f} | "
                f"val difficulty drift: {val_stats['difficulty_proxy_drift']:.3f}"
            )

        current_best = val_stats["loss"] if best_val_loss is None else min(best_val_loss, val_stats["loss"])
        metadata = CheckpointMetadata(
            epoch=epoch,
            global_step=global_step,
            best_val_loss=float(current_best),
            data_root=str(data_root),
            artifact_paths=relative_artifacts,
        )

        save_checkpoint(
            checkpoints_dir / "last.ckpt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            architecture_spec=architecture_spec,
            training_spec=training_spec,
            metadata=metadata,
            history=history,
            vocab=vocab,
            split_ids=split_ids,
            adherence_config=adherence_config,
        )

        best_updated = False
        if best_val_loss is None or val_stats["loss"] < best_val_loss:
            best_val_loss = float(val_stats["loss"])
            best_updated = True
            best_metadata = CheckpointMetadata(
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
                data_root=str(data_root),
                artifact_paths=relative_artifacts,
            )
            save_checkpoint(
                checkpoints_dir / "best.ckpt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                architecture_spec=architecture_spec,
                training_spec=training_spec,
                metadata=best_metadata,
                history=history,
                vocab=vocab,
                split_ids=split_ids,
                adherence_config=adherence_config,
            )

        if metrics_logger is not None:
            epoch_payload: dict[str, Any] = {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "train/loss_epoch": float(train_stats["loss"]),
                "val/loss_epoch": float(val_stats["loss"]),
                "optimizer/lr": float(current_lr),
                "checkpoint/last_path": str((checkpoints_dir / "last.ckpt").resolve()),
                "checkpoint/best_updated": int(best_updated),
            }
            if "density_proxy_abs_error" in train_stats:
                epoch_payload["train/density_proxy_abs_error"] = float(train_stats["density_proxy_abs_error"])
                epoch_payload["val/density_proxy_abs_error"] = float(val_stats["density_proxy_abs_error"])
                epoch_payload["train/difficulty_proxy_drift"] = float(train_stats["difficulty_proxy_drift"])
                epoch_payload["val/difficulty_proxy_drift"] = float(val_stats["difficulty_proxy_drift"])
            metrics_logger(epoch_payload)

    if wandb_runtime.run is not None:
        wandb_runtime.run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
