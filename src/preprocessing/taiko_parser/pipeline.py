from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

from .hitobjects import NoteEvent, append_bpm_change_events, parse_hit_objects_taiko
from .sections import parse_key_value_section, safe_int, safe_float, split_sections
from .timing import TimingPoint, parse_raw_timing_points, snap_raw_timing_points


def parse_osu_taiko(
    osu_text: str,
    source_name: str = "map.osu",
    include_bpm_events: bool = False,
) -> Tuple[dict, dict, dict]:
    sections = split_sections(osu_text)

    general = parse_key_value_section(sections.get("General", []))
    metadata = parse_key_value_section(sections.get("Metadata", []))
    difficulty = parse_key_value_section(sections.get("Difficulty", []))

    mode = safe_int(general.get("Mode", "0"))
    if mode != 1:
        raise ValueError(f"Expected Mode: 1 (taiko), got {mode}")

    slider_multiplier = safe_float(difficulty.get("SliderMultiplier", "1.4"), 1.4)
    slider_tick_rate = safe_float(difficulty.get("SliderTickRate", "1"), 1.0)

    raw_timing_points = parse_raw_timing_points(sections.get("TimingPoints", []))
    if not raw_timing_points:
        raise ValueError("No timing points found")

    snapped_timing_points = snap_raw_timing_points(raw_timing_points)

    notes = parse_hit_objects_taiko(
        sections.get("HitObjects", []),
        raw_timing_points,
        slider_multiplier=slider_multiplier,
    )

    if include_bpm_events:
        notes = append_bpm_change_events(
            notes,
            snapped_timing_points,
            raw_timing_points,
            slider_multiplier,
        )

    metadata_obj = {
        "format": 2,
        "source_osu": source_name,
        "general": general,
        "metadata": metadata,
        "difficulty": difficulty,
    }

    timing_obj = {
        "format": 2,
        "source_osu": source_name,
        "slider_multiplier": slider_multiplier,
        "slider_tick_rate": slider_tick_rate,
        "timing_points": [asdict(tp) for tp in snapped_timing_points],
    }

    notes_obj = {
        "format": 2,
        "mode": 1,
        "source_osu": source_name,
        "notes": [asdict(n) for n in notes],
    }

    return metadata_obj, timing_obj, notes_obj


def parse_osu_file_to_jsons(
    osu_path: Path,
    out_dir: Path,
    include_bpm_events: bool = False,
) -> None:
    text = osu_path.read_text(encoding="utf-8")
    meta, timing, notes = parse_osu_taiko(
        text,
        source_name=osu_path.name,
        include_bpm_events=include_bpm_events,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = osu_path.stem

    (out_dir / f"{stem}.metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / f"{stem}.timing.json").write_text(
        json.dumps(timing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / f"{stem}.notes.json").write_text(
        json.dumps(notes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_unpacked_taiko_charts(
    unpacked_root: Path,
    *,
    include_bpm_events: bool = False,
    overwrite: bool = False,
) -> dict[str, int]:
    unpacked_root = Path(unpacked_root)
    if not unpacked_root.exists():
        raise FileNotFoundError(f"Unpacked root not found: {unpacked_root}")

    parsed_count = 0
    skipped_count = 0
    error_count = 0

    for folder_path in sorted(p for p in unpacked_root.iterdir() if p.is_dir()):
        parsed_dir = folder_path / "parsed"
        parsed_dir.mkdir(parents=True, exist_ok=True)

        for osu_path in sorted(folder_path.glob("*.osu")):
            notes_path = parsed_dir / f"{osu_path.stem}.notes.json"
            timing_path = parsed_dir / f"{osu_path.stem}.timing.json"
            metadata_path = parsed_dir / f"{osu_path.stem}.metadata.json"

            if not overwrite and notes_path.exists() and timing_path.exists() and metadata_path.exists():
                skipped_count += 1
                continue

            try:
                parse_osu_file_to_jsons(
                    osu_path=osu_path,
                    out_dir=parsed_dir,
                    include_bpm_events=include_bpm_events,
                )
                parsed_count += 1
            except ValueError as exc:
                if "Expected Mode: 1 (taiko)" in str(exc):
                    skipped_count += 1
                    continue
                error_count += 1
                logging.warning("Failed to parse %s: %s", osu_path, exc)
            except Exception as exc:
                error_count += 1
                logging.warning("Failed to parse %s: %s", osu_path, exc)

    return {
        "parsed_count": parsed_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
    }
