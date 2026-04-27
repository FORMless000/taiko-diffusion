from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from src.preprocessing.build_snapshot_dataset import DEFAULT_MAX_AUDIO_MB, evaluate_set_folder


DEFAULT_SOURCE_UNPACKED_ROOT = Path("C:/taiko-transformer-cache/unpacked")
DEFAULT_SNAPSHOT_MANIFEST = Path("C:/taiko-transformer-cache/snapshots/taiko_only_static_bpm_1000_seed42/selection_manifest.csv")


def load_snapshot_folder_ids(manifest_path: Path) -> set[str]:
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Snapshot manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {str(row.get("folder_id", "")).strip() for row in reader if str(row.get("folder_id", "")).strip()}


def build_osz_from_unpacked_folder(folder_path: Path, out_path: Path) -> Path:
    folder_path = Path(folder_path).resolve()
    out_path = Path(out_path).resolve()
    audio_files = sorted(list(folder_path.glob("*.mp3")) + list(folder_path.glob("*.ogg")))
    osu_files = sorted(folder_path.glob("*.osu"))
    if len(audio_files) != 1:
        raise RuntimeError(f"Expected exactly one audio file in {folder_path}, found {len(audio_files)}")
    if not osu_files:
        raise RuntimeError(f"No .osu files found in {folder_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(out_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(audio_files[0], arcname=audio_files[0].name)
        for osu_path in osu_files:
            archive.write(osu_path, arcname=osu_path.name)
    return out_path


def choose_eligible_non_snapshot_sets(
    *,
    source_unpacked_root: Path,
    snapshot_folder_ids: set[str],
    target_count: int,
    seed: int,
    max_audio_mb: float,
) -> list[dict]:
    folders = [path for path in Path(source_unpacked_root).resolve().iterdir() if path.is_dir() and path.name not in snapshot_folder_ids]
    rng = random.Random(int(seed))
    rng.shuffle(folders)

    max_audio_bytes = int(float(max_audio_mb) * 1024 * 1024)
    selected: list[dict] = []
    for folder_path in folders:
        candidate, rejection = evaluate_set_folder(folder_path, max_audio_bytes=max_audio_bytes)
        if candidate is None:
            continue
        selected.append(
            {
                "folder_id": candidate.folder_id,
                "folder_path": candidate.folder_path,
                "audio_file": candidate.audio_file,
                "audio_size_bytes": int(candidate.audio_size_bytes),
                "n_osu_files": int(candidate.n_osu_files),
                "n_complete_chart_triples": int(candidate.n_complete_chart_triples),
            }
        )
        if len(selected) >= int(target_count):
            break

    if len(selected) < int(target_count):
        raise RuntimeError(
            f"Only found {len(selected)} eligible non-snapshot sets, but {int(target_count)} were requested."
        )
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select eligible non-snapshot taiko sets and package them as .osz files.")
    parser.add_argument("--source-unpacked-root", default=str(DEFAULT_SOURCE_UNPACKED_ROOT), help="Root containing unpacked beatmap set folders.")
    parser.add_argument("--snapshot-manifest", default=str(DEFAULT_SNAPSHOT_MANIFEST), help="Snapshot selection_manifest.csv used for exclusion.")
    parser.add_argument("--out-dir", required=True, help="Directory to write selected .osz files and selection metadata.")
    parser.add_argument("--count", type=int, default=5, help="Number of non-snapshot eligible sets to select.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for selection.")
    parser.add_argument("--max-audio-mb", type=float, default=DEFAULT_MAX_AUDIO_MB, help="Maximum audio size in MB, matching snapshot eligibility.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_folder_ids = load_snapshot_folder_ids(Path(args.snapshot_manifest))
    selected = choose_eligible_non_snapshot_sets(
        source_unpacked_root=Path(args.source_unpacked_root),
        snapshot_folder_ids=snapshot_folder_ids,
        target_count=max(1, int(args.count)),
        seed=int(args.seed),
        max_audio_mb=float(args.max_audio_mb),
    )

    packaged_rows = []
    for item in selected:
        folder_id = str(item["folder_id"])
        source_folder = Path(item["folder_path"]).resolve()
        osz_path = out_dir / f"{folder_id}.osz"
        build_osz_from_unpacked_folder(source_folder, osz_path)
        row = dict(item)
        row["packaged_osz_path"] = str(osz_path)
        packaged_rows.append(row)
        print(f"[selected] {folder_id} -> {osz_path}")

    manifest_path = out_dir / "selected_sets.json"
    manifest_path.write_text(json.dumps(packaged_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[wrote] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
