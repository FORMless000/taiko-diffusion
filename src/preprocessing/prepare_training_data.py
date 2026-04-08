from __future__ import annotations

import argparse
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Iterable, Sequence

from .beat_aligned_dataset import run_pipeline, setup_logging
from .osutaiko_parser import parse_osu_file_to_jsons
from .unpack_osz import UnpackSummary, unpack_osz_files


@dataclass
class TrainingDataArtifacts:
    data_root: Path
    unpacked_root: Path
    index_dir: Path
    dataset_dir: Path
    audio_dir: Path
    token_dir: Path
    chart_metadata_csv: Path
    sequence_metadata_csv: Path


def _safe_int(text: str, default: int = 0) -> int:
    try:
        return int(str(text).strip())
    except Exception:
        return default


def _safe_float(text: str, default: float = 0.0) -> float:
    try:
        return float(str(text).strip())
    except Exception:
        return default


def _quick_osu_screen_for_constant_taiko(osu_path: Path) -> tuple[bool, str]:
    """
    Lightweight pre-screen to avoid expensive full parse for clearly ineligible charts.

    Returns:
        (is_eligible, reason_if_skipped)
    """
    text = osu_path.read_text(encoding="utf-8")
    mode_value = None
    in_general = False
    in_timing_points = False
    unique_uninherited_mpb: set[float] = set()

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue

        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            in_general = section == "General"
            in_timing_points = section == "TimingPoints"
            continue

        if in_general and ":" in line:
            key, value = line.split(":", 1)
            if key.strip() == "Mode":
                mode_value = _safe_int(value, 0)
            continue

        if in_timing_points:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 8:
                continue
            uninherited = _safe_int(parts[6], 1)
            if uninherited != 1:
                continue
            ms_per_beat = round(_safe_float(parts[1], 0.0), 10)
            if ms_per_beat > 0:
                unique_uninherited_mpb.add(ms_per_beat)

    if mode_value != 1:
        return False, f"non_taiko_mode_{mode_value}"
    if not unique_uninherited_mpb:
        return False, "no_uninherited_bpm_points"
    if len(unique_uninherited_mpb) != 1:
        return False, f"non_constant_bpm_{sorted(unique_uninherited_mpb)[:8]}"
    return True, ""


def resolve_osz_input_paths(osz_inputs: Sequence[str | Path]) -> list[str]:
    resolved: list[str] = []

    for raw_input in osz_inputs:
        text = str(raw_input)
        path = Path(text)

        if path.is_dir():
            resolved.extend(sorted(str(p) for p in path.glob("*.osz")))
            continue

        matches = sorted(glob(text))
        if matches:
            resolved.extend(matches)
            continue

        if path.exists() and path.suffix.lower() == ".osz":
            resolved.append(str(path))

    deduped = sorted({str(Path(p).resolve()) for p in resolved})
    if not deduped:
        raise FileNotFoundError("No .osz files matched the provided inputs.")
    return deduped


def parse_unpacked_beatmaps(unpacked_dirs: Iterable[Path], overwrite_parsed: bool = False) -> int:
    parsed_count = 0
    skipped_count = 0
    skipped_non_taiko = 0
    skipped_non_constant_bpm = 0
    skipped_no_bpm_points = 0

    for unpacked_dir in unpacked_dirs:
        parsed_dir = unpacked_dir / "parsed"
        parsed_dir.mkdir(parents=True, exist_ok=True)

        for osu_path in sorted(unpacked_dir.glob("*.osu")):
            stem = osu_path.stem
            expected_outputs = [
                parsed_dir / f"{stem}.metadata.json",
                parsed_dir / f"{stem}.timing.json",
                parsed_dir / f"{stem}.notes.json",
            ]

            if not overwrite_parsed and all(path.exists() for path in expected_outputs):
                continue

            try:
                is_eligible, reason = _quick_osu_screen_for_constant_taiko(osu_path)
                if not is_eligible:
                    skipped_count += 1
                    if reason.startswith("non_taiko_mode"):
                        skipped_non_taiko += 1
                    elif reason.startswith("non_constant_bpm"):
                        skipped_non_constant_bpm += 1
                    elif reason == "no_uninherited_bpm_points":
                        skipped_no_bpm_points += 1
                    print(f"[INFO] Fast-skip chart before full parse: {osu_path} ({reason})")
                    continue

                parse_osu_file_to_jsons(
                    osu_path=osu_path,
                    out_dir=parsed_dir,
                    include_bpm_events=True,
                )
                parsed_count += 1
            except (ValueError, UnicodeDecodeError, OSError) as exc:
                skipped_count += 1
                print(f"[WARN] Skipping unparseable/non-taiko chart: {osu_path} ({exc})")
                continue

    if skipped_count > 0:
        print(f"Skipped {skipped_count} .osu file(s) during parse due to parse/format errors.")
        print(
            "Fast-skip breakdown | "
            f"non-taiko: {skipped_non_taiko} | "
            f"non-constant-bpm: {skipped_non_constant_bpm} | "
            f"no-bpm-points: {skipped_no_bpm_points}"
        )

    return parsed_count


def prepare_training_data(
    osz_inputs: Sequence[str | Path],
    data_root: str | Path,
    *,
    overwrite_unpack: bool = False,
    overwrite_parsed: bool = False,
    overwrite_dataset_outputs: bool = False,
    reject_offgrid_notes: bool = True,
    offgrid_tolerance_ms: float = 5.0,
    keep_only_max_notes_per_song: bool = False,
) -> TrainingDataArtifacts:
    data_root = Path(data_root).resolve()
    unpacked_root = data_root / "unpacked"
    index_dir = data_root / "chart_index"
    dataset_dir = data_root / "beat_aligned_dataset"

    resolved_inputs = resolve_osz_input_paths(osz_inputs)

    unpack_result = unpack_osz_files(
        source_paths=resolved_inputs,
        destination_root=unpacked_root,
        overwrite=overwrite_unpack,
        keep_only_chart_and_audio=True,
        progress_desc="Unpacking .osz files (total)",
        return_summary=True,
    )
    if not isinstance(unpack_result, UnpackSummary):
        raise RuntimeError("Expected unpack summary metadata but received a legacy unpack result.")

    print(
        "Unpack summary | "
        f"total: {unpack_result.total_files} | "
        f"unpacked: {unpack_result.unpacked_ok} | "
        f"skipped existing: {unpack_result.skipped_existing} | "
        f"failed corrupt: {unpack_result.failed_corrupt}"
    )
    if unpack_result.failed_files:
        print(f"Skipped {len(unpack_result.failed_files)} corrupted/unreadable archive(s).")

    unpacked_dirs = unpack_result.extracted_dirs

    deduped_dirs = sorted({path.resolve() for path in unpacked_dirs})
    if not deduped_dirs:
        raise RuntimeError(
            "No valid unpacked beatmap directories are available. "
            "All matched .osz archives failed to unpack or were unavailable."
        )

    parse_unpacked_beatmaps(deduped_dirs, overwrite_parsed=overwrite_parsed)

    run_pipeline(
        unpacked_root=unpacked_root,
        index_dir=index_dir,
        dataset_dir=dataset_dir,
        overwrite_dataset_outputs=overwrite_dataset_outputs,
        reject_offgrid_notes=reject_offgrid_notes,
        offgrid_tolerance_ms=offgrid_tolerance_ms,
        keep_only_max_notes_per_song=keep_only_max_notes_per_song,
    )

    return TrainingDataArtifacts(
        data_root=data_root,
        unpacked_root=unpacked_root,
        index_dir=index_dir,
        dataset_dir=dataset_dir,
        audio_dir=dataset_dir / "audio_npz",
        token_dir=dataset_dir / "token_json",
        chart_metadata_csv=index_dir / "chart_build_summary.csv",
        sequence_metadata_csv=dataset_dir / "sequence_metadata.csv",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare training data from raw .osz files.")
    parser.add_argument("osz_inputs", nargs="+", help="Raw .osz files, directories, or glob patterns.")
    parser.add_argument("--data-root", required=True, help="Root directory for unpacked and dataset artifacts.")
    parser.add_argument("--overwrite-unpack", action="store_true", help="Re-extract beatmap archives if already unpacked.")
    parser.add_argument("--overwrite-parsed", action="store_true", help="Rebuild parsed JSONs even if they already exist.")
    parser.add_argument(
        "--overwrite-dataset-outputs",
        action="store_true",
        help="Rebuild beat-aligned dataset outputs even when per-chart files already exist.",
    )
    parser.add_argument("--allow-offgrid-notes", action="store_true", help="Do not reject off-grid notes during dataset build.")
    parser.add_argument("--offgrid-tolerance-ms", type=float, default=5.0, help="Maximum allowed off-grid deviation in milliseconds.")
    parser.add_argument("--keep-only-max-notes-per-song", action="store_true", help="Keep only the chart with the highest model note count per song.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    setup_logging()
    prepare_training_data(
        osz_inputs=args.osz_inputs,
        data_root=args.data_root,
        overwrite_unpack=args.overwrite_unpack,
        overwrite_parsed=args.overwrite_parsed,
        overwrite_dataset_outputs=args.overwrite_dataset_outputs,
        reject_offgrid_notes=not args.allow_offgrid_notes,
        offgrid_tolerance_ms=args.offgrid_tolerance_ms,
        keep_only_max_notes_per_song=args.keep_only_max_notes_per_song,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
