"""Utilities for unpacking .osz beatmap archives."""

from __future__ import annotations

from glob import glob
from pathlib import Path
from shutil import rmtree
from zipfile import BadZipFile, ZipFile

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
    source_glob: str,
    destination_root: str | Path,
    overwrite: bool = False,
    keep_only_chart_and_audio: bool = True,
) -> list[Path]:
    """Unpack all .osz files matched by ``source_glob`` into ``destination_root``.

    Each archive is extracted into its own subfolder named after the .osz filename stem.
    """
    source_files = sorted(Path(path) for path in glob(source_glob))
    destination_root = Path(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    extracted_dirs: list[Path] = []

    for osz_path in tqdm(source_files, desc="Unpacking .osz files", unit="file"):
        target_dir = destination_root / osz_path.stem

        if target_dir.exists() and not overwrite:
            extracted_dirs.append(target_dir)
            continue

        if target_dir.exists() and overwrite:
            rmtree(target_dir)

        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            with ZipFile(osz_path, "r") as archive:
                archive.extractall(target_dir)
        except BadZipFile as exc:
            raise RuntimeError(f"Invalid .osz archive: {osz_path}") from exc

        if keep_only_chart_and_audio:
            _clean_to_chart_and_audio_only(target_dir)

        extracted_dirs.append(target_dir)

    return extracted_dirs


if __name__ == "__main__":
    extracted = unpack_osz_files(
        source_glob="sample_data/raw/*.osz",
        destination_root="sample_data/unpacked",
        overwrite=False,
        keep_only_chart_and_audio=True,
    )
    print(f"Processed {len(extracted)} archive(s).")
