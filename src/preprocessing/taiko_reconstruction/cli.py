from __future__ import annotations

import argparse
from pathlib import Path

from .writer import guess_related_path, reconstruct_osu


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct osu!taiko .osu from generated JSON files.")
    parser.add_argument("notes", type=Path, help="Path to .notes.json")
    parser.add_argument("-t", "--timing", type=Path, default=None, help="Path to .timing.json")
    parser.add_argument("-m", "--metadata", type=Path, default=None, help="Path to .metadata.json")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .osu path")
    args = parser.parse_args()

    notes_path = args.notes
    timing_path = args.timing or guess_related_path(notes_path, ".timing.json")
    metadata_path = args.metadata or guess_related_path(notes_path, ".metadata.json")

    output_path = args.output
    if output_path is None:
        if notes_path.name.endswith(".notes.json"):
            output_name = notes_path.name.replace(".notes.json", ".reconstructed.osu")
        else:
            output_name = notes_path.stem + ".reconstructed.osu"
        output_path = notes_path.with_name(output_name)

    reconstruct_osu(
        notes_path=notes_path,
        out_path=output_path,
        timing_path=timing_path,
        metadata_path=metadata_path,
    )
    print(f"Wrote {output_path}")
