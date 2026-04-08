"""Utilities for unpacking .osz beatmap archives."""

from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import Path
from shutil import rmtree
import zlib
from zipfile import BadZipFile, ZipFile
from typing import Sequence

from tqdm import tqdm

_KEEP_EXTENSIONS = {
    ".osu",
    ".mp3",
    ".ogg",
    ".wav",
    ".flac",
    ".m4a",
    ".aac",
    ".opus",
}


@dataclass
class UnpackSummary:
    total_files: int
    unpacked_ok: int
    skipped_existing: int
    failed_corrupt: int
    extracted_dirs: list[Path]
    failed_files: list[Path]


def _clean_to_chart_and_audio_only(directory: Path) -> None:
    """Remove non-chart/non-audio files from an extracted beatmap directory."""
    for file_path in directory.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() in _KEEP_EXTENSIONS:
            continue
        file_path.unlink()

    # Remove now-empty directories left after file cleanup.
    for path in sorted(directory.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                # Directory still has kept files.
                pass


def unpack_osz_files(
    source_glob: str | None = None,
    *,
    source_paths: Sequence[str | Path] | None = None,
    destination_root: str | Path,
    overwrite: bool = False,
    keep_only_chart_and_audio: bool = True,
    progress_desc: str = "Unpacking .osz files",
    return_summary: bool = False,
) -> list[Path] | UnpackSummary:
    """Unpack all .osz files matched by ``source_glob`` into ``destination_root``.

    Each archive is extracted into its own subfolder named after the .osz filename stem.
    """
    if source_paths is not None:
        source_files = sorted({Path(path).resolve() for path in source_paths})
    elif source_glob:
        source_files = sorted(Path(path).resolve() for path in glob(source_glob))
    else:
        raise ValueError("Provide either `source_glob` or `source_paths`.")

    destination_root = Path(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    extracted_dirs: list[Path] = []
    failed_files: list[Path] = []
    skipped_existing = 0
    unpacked_ok = 0
    failed_corrupt = 0

    for osz_path in tqdm(source_files, desc=progress_desc, unit="file", total=len(source_files)):
        target_dir = destination_root / osz_path.stem

        if target_dir.exists() and not overwrite:
            skipped_existing += 1
            extracted_dirs.append(target_dir)
            continue

        if target_dir.exists() and overwrite:
            rmtree(target_dir)

        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            with ZipFile(osz_path, "r") as archive:
                archive.extractall(target_dir)
            if keep_only_chart_and_audio:
                _clean_to_chart_and_audio_only(target_dir)
        except (BadZipFile, zlib.error, OSError, RuntimeError) as exc:
            failed_corrupt += 1
            failed_files.append(osz_path)
            if target_dir.exists():
                rmtree(target_dir, ignore_errors=True)
            print(f"[WARN] Skipping corrupted/unreadable archive: {osz_path} ({exc})")
            continue

        unpacked_ok += 1
        extracted_dirs.append(target_dir)

    summary = UnpackSummary(
        total_files=len(source_files),
        unpacked_ok=unpacked_ok,
        skipped_existing=skipped_existing,
        failed_corrupt=failed_corrupt,
        extracted_dirs=extracted_dirs,
        failed_files=failed_files,
    )
    if return_summary:
        return summary
    return summary.extracted_dirs


if __name__ == "__main__":
    extracted = unpack_osz_files(
        source_glob="sample_data/raw/*.osz",
        destination_root="sample_data/unpacked",
        overwrite=False,
        keep_only_chart_and_audio=True,
    )
    print(f"Processed {len(extracted)} archive(s).")
