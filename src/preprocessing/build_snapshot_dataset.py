from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import random
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .beat_aligned_dataset import run_pipeline, setup_logging
from .prepare_training_data import _quick_osu_screen_for_constant_taiko


DEFAULT_SOURCE_UNPACKED_ROOT = Path("C:/taiko-transformer-cache/unpacked")
DEFAULT_SNAPSHOTS_ROOT = Path("C:/taiko-transformer-cache/snapshots")
DEFAULT_TARGET_SET_COUNT = 2000
DEFAULT_SEED = 42
DEFAULT_MAX_AUDIO_MB = 5.0
_AUDIO_SIZE_UNIT_BYTES = 1024 * 1024


@dataclass
class CandidateSetRecord:
    folder_id: str
    folder_path: str
    audio_file: str
    audio_path: str
    audio_size_bytes: int
    n_osu_files: int
    n_complete_chart_triples: int


@dataclass
class RejectionRecord:
    folder_id: str
    folder_path: str
    reason: str
    detail: str


def _audio_files_for_folder(folder_path: Path) -> list[Path]:
    return sorted(list(folder_path.glob("*.mp3")) + list(folder_path.glob("*.ogg")))


def _count_complete_chart_triples(parsed_dir: Path) -> int:
    if not parsed_dir.exists() or not parsed_dir.is_dir():
        return 0

    notes_map = {p.name[:-11] for p in parsed_dir.glob("*.notes.json")}
    timing_map = {p.name[:-12] for p in parsed_dir.glob("*.timing.json")}
    metadata_map = {p.name[:-14] for p in parsed_dir.glob("*.metadata.json")}
    return int(len(notes_map & timing_map & metadata_map))


def evaluate_set_folder(folder_path: Path, *, max_audio_bytes: int) -> tuple[CandidateSetRecord | None, RejectionRecord | None]:
    folder_path = Path(folder_path)
    folder_id = folder_path.name
    audio_files = _audio_files_for_folder(folder_path)

    if len(audio_files) != 1:
        detail = f"expected exactly 1 audio file, found {len(audio_files)}"
        return None, RejectionRecord(folder_id=folder_id, folder_path=str(folder_path), reason="audio_count_error", detail=detail)

    audio_path = audio_files[0]
    audio_size_bytes = int(audio_path.stat().st_size)
    if audio_size_bytes > int(max_audio_bytes):
        detail = f"audio file {audio_path.name} is {audio_size_bytes} bytes"
        return None, RejectionRecord(folder_id=folder_id, folder_path=str(folder_path), reason="audio_too_large", detail=detail)

    osu_paths = sorted(folder_path.glob("*.osu"))
    if not osu_paths:
        return None, RejectionRecord(folder_id=folder_id, folder_path=str(folder_path), reason="no_osu_files", detail="no .osu files found")

    for osu_path in osu_paths:
        is_eligible, reason = _quick_osu_screen_for_constant_taiko(osu_path)
        if is_eligible:
            continue
        normalized_reason = "non_constant_bpm" if str(reason).startswith("non_constant_bpm") or str(reason) == "no_uninherited_bpm_points" else "non_taiko_mode"
        detail = f"{osu_path.name}: {reason}"
        return None, RejectionRecord(folder_id=folder_id, folder_path=str(folder_path), reason=normalized_reason, detail=detail)

    parsed_dir = folder_path / "parsed"
    if not parsed_dir.exists() or not parsed_dir.is_dir():
        return None, RejectionRecord(folder_id=folder_id, folder_path=str(folder_path), reason="no_parsed_folder", detail="parsed/ directory not found")

    n_complete_chart_triples = _count_complete_chart_triples(parsed_dir)
    if n_complete_chart_triples <= 0:
        return None, RejectionRecord(
            folder_id=folder_id,
            folder_path=str(folder_path),
            reason="no_complete_chart_triples",
            detail="no complete parsed chart triples found",
        )

    return (
        CandidateSetRecord(
            folder_id=folder_id,
            folder_path=str(folder_path),
            audio_file=audio_path.name,
            audio_path=str(audio_path),
            audio_size_bytes=audio_size_bytes,
            n_osu_files=len(osu_paths),
            n_complete_chart_triples=n_complete_chart_triples,
        ),
        None,
    )


def scan_candidate_sets(source_unpacked_root: str | Path, *, max_audio_mb: float = DEFAULT_MAX_AUDIO_MB) -> tuple[list[CandidateSetRecord], list[RejectionRecord]]:
    source_unpacked_root = Path(source_unpacked_root).resolve()
    if not source_unpacked_root.exists():
        raise FileNotFoundError(f"Source unpacked root not found: {source_unpacked_root}")
    if not source_unpacked_root.is_dir():
        raise NotADirectoryError(f"Source unpacked root is not a directory: {source_unpacked_root}")

    max_audio_bytes = int(float(max_audio_mb) * _AUDIO_SIZE_UNIT_BYTES)
    candidates: list[CandidateSetRecord] = []
    rejections: list[RejectionRecord] = []

    for folder_path in sorted([path for path in source_unpacked_root.iterdir() if path.is_dir()], key=lambda path: path.name):
        candidate, rejection = evaluate_set_folder(folder_path, max_audio_bytes=max_audio_bytes)
        if candidate is not None:
            candidates.append(candidate)
        elif rejection is not None:
            rejections.append(rejection)

    return candidates, rejections


def choose_snapshot_sets(candidates: list[CandidateSetRecord], *, target_set_count: int, seed: int) -> list[CandidateSetRecord]:
    ordered = sorted(candidates, key=lambda item: item.folder_id)
    if len(ordered) < int(target_set_count):
        raise RuntimeError(
            f"Requested {int(target_set_count)} sets but only {len(ordered)} eligible sets were found."
        )

    rng = random.Random(int(seed))
    rng.shuffle(ordered)
    return ordered[: int(target_set_count)]


def _default_snapshot_root(target_set_count: int, seed: int) -> Path:
    return DEFAULT_SNAPSHOTS_ROOT / f"taiko_only_static_bpm_{int(target_set_count)}_seed{int(seed)}"


def _prepare_snapshot_root(snapshot_root: Path, *, overwrite: bool) -> None:
    if snapshot_root.exists():
        if not overwrite:
            raise FileExistsError(f"Snapshot root already exists: {snapshot_root}")
        shutil.rmtree(snapshot_root)
    snapshot_root.mkdir(parents=True, exist_ok=True)


def _copy_selected_sets(selected_sets: list[CandidateSetRecord], destination_unpacked_root: Path) -> None:
    destination_unpacked_root.mkdir(parents=True, exist_ok=True)
    for record in selected_sets:
        source = Path(record.folder_path)
        destination = destination_unpacked_root / record.folder_id
        shutil.copytree(source, destination, dirs_exist_ok=False)


def _write_selection_manifest(
    snapshot_root: Path,
    selected_sets: list[CandidateSetRecord],
    *,
    seed: int,
    target_set_count: int,
) -> Path:
    rows = []
    for record in selected_sets:
        row = asdict(record)
        row["seed"] = int(seed)
        row["target_set_count"] = int(target_set_count)
        rows.append(row)
    manifest_path = snapshot_root / "selection_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return manifest_path


def _write_rejection_report(snapshot_root: Path, rejections: list[RejectionRecord]) -> Path:
    report_path = snapshot_root / "rejection_report.csv"
    pd.DataFrame([asdict(item) for item in rejections]).to_csv(report_path, index=False, encoding="utf-8-sig")
    return report_path


def build_snapshot_dataset(
    *,
    source_unpacked_root: str | Path = DEFAULT_SOURCE_UNPACKED_ROOT,
    snapshot_root: str | Path | None = None,
    target_set_count: int = DEFAULT_TARGET_SET_COUNT,
    seed: int = DEFAULT_SEED,
    max_audio_mb: float = DEFAULT_MAX_AUDIO_MB,
    overwrite: bool = False,
    keep_only_max_notes_per_song: bool = False,
) -> dict[str, Any]:
    resolved_target_count = max(1, int(target_set_count))
    resolved_seed = int(seed)
    resolved_snapshot_root = Path(snapshot_root).resolve() if snapshot_root is not None else _default_snapshot_root(resolved_target_count, resolved_seed)
    source_unpacked_root = Path(source_unpacked_root).resolve()

    candidates, rejections = scan_candidate_sets(source_unpacked_root, max_audio_mb=max_audio_mb)
    selected_sets = choose_snapshot_sets(candidates, target_set_count=resolved_target_count, seed=resolved_seed)

    _prepare_snapshot_root(resolved_snapshot_root, overwrite=bool(overwrite))
    destination_unpacked_root = resolved_snapshot_root / "unpacked"
    _copy_selected_sets(selected_sets, destination_unpacked_root)

    selection_manifest_path = _write_selection_manifest(
        resolved_snapshot_root,
        selected_sets,
        seed=resolved_seed,
        target_set_count=resolved_target_count,
    )
    rejection_report_path = _write_rejection_report(resolved_snapshot_root, rejections)

    index_dir = resolved_snapshot_root / "chart_index"
    dataset_dir = resolved_snapshot_root / "beat_aligned_dataset"
    run_pipeline(
        unpacked_root=destination_unpacked_root,
        index_dir=index_dir,
        dataset_dir=dataset_dir,
        keep_only_max_notes_per_song=bool(keep_only_max_notes_per_song),
        overwrite_dataset_outputs=False,
    )

    rejection_breakdown = Counter(item.reason for item in rejections)
    summary = {
        "source_unpacked_root": str(source_unpacked_root),
        "snapshot_root": str(resolved_snapshot_root),
        "selection_manifest_csv": str(selection_manifest_path),
        "rejection_report_csv": str(rejection_report_path),
        "scanned_set_count": int(len(candidates) + len(rejections)),
        "eligible_set_count": int(len(candidates)),
        "selected_set_count": int(len(selected_sets)),
        "rejection_breakdown": dict(sorted(rejection_breakdown.items())),
        "target_set_count": resolved_target_count,
        "seed": resolved_seed,
    }

    print(f"Scanned set count   : {summary['scanned_set_count']}")
    print(f"Eligible set count  : {summary['eligible_set_count']}")
    print(f"Selected set count  : {summary['selected_set_count']}")
    if summary["rejection_breakdown"]:
        print(
            "Rejection breakdown : "
            + ", ".join(f"{key}={value}" for key, value in summary["rejection_breakdown"].items())
        )
    else:
        print("Rejection breakdown : none")
    print(f"Snapshot root       : {resolved_snapshot_root}")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a taiko-only constant-BPM snapshot from an unpacked dataset.")
    parser.add_argument(
        "--source-unpacked-root",
        default=str(DEFAULT_SOURCE_UNPACKED_ROOT),
        help="Root directory containing already-unpacked beatmap-set folders.",
    )
    parser.add_argument(
        "--snapshot-root",
        default=None,
        help="Destination snapshot root. Defaults to C:/taiko-transformer-cache/snapshots/taiko_only_static_bpm_<count>_seed<seed>.",
    )
    parser.add_argument("--target-set-count", type=int, default=DEFAULT_TARGET_SET_COUNT, help="Number of eligible set folders to include.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed used after stable sorting for reproducible selection.")
    parser.add_argument("--max-audio-mb", type=float, default=DEFAULT_MAX_AUDIO_MB, help="Maximum allowed audio file size per set in MiB.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing snapshot root.")
    parser.add_argument(
        "--keep-only-max-notes-per-song",
        action="store_true",
        help="Pass through the existing song-level max-note filter during dataset build.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    build_snapshot_dataset(
        source_unpacked_root=args.source_unpacked_root,
        snapshot_root=args.snapshot_root,
        target_set_count=args.target_set_count,
        seed=args.seed,
        max_audio_mb=args.max_audio_mb,
        overwrite=args.overwrite,
        keep_only_max_notes_per_song=args.keep_only_max_notes_per_song,
    )
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
