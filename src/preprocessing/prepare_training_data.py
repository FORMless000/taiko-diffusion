from __future__ import annotations

import argparse
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Iterable, Sequence

from .beat_aligned_dataset import run_pipeline, setup_logging
from .osutaiko_parser import parse_osu_file_to_jsons
from .unpack_osz import unpack_osz_files


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

            parse_osu_file_to_jsons(
                osu_path=osu_path,
                out_dir=parsed_dir,
                include_bpm_events=True,
            )
            parsed_count += 1

    return parsed_count


def prepare_training_data(
    osz_inputs: Sequence[str | Path],
    data_root: str | Path,
    *,
    overwrite_unpack: bool = False,
    overwrite_parsed: bool = False,
    reject_offgrid_notes: bool = True,
    offgrid_tolerance_ms: float = 5.0,
    keep_only_max_notes_per_song: bool = False,
) -> TrainingDataArtifacts:
    data_root = Path(data_root).resolve()
    unpacked_root = data_root / "unpacked"
    index_dir = data_root / "chart_index"
    dataset_dir = data_root / "beat_aligned_dataset"

    resolved_inputs = resolve_osz_input_paths(osz_inputs)

    unpacked_dirs: list[Path] = []
    for input_spec in resolved_inputs:
        unpacked_dirs.extend(
            unpack_osz_files(
                source_glob=input_spec,
                destination_root=unpacked_root,
                overwrite=overwrite_unpack,
                keep_only_chart_and_audio=True,
            )
        )

    deduped_dirs = sorted({path.resolve() for path in unpacked_dirs})
    parse_unpacked_beatmaps(deduped_dirs, overwrite_parsed=overwrite_parsed)

    run_pipeline(
        unpacked_root=unpacked_root,
        index_dir=index_dir,
        dataset_dir=dataset_dir,
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
        reject_offgrid_notes=not args.allow_offgrid_notes,
        offgrid_tolerance_ms=args.offgrid_tolerance_ms,
        keep_only_max_notes_per_song=args.keep_only_max_notes_per_song,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
