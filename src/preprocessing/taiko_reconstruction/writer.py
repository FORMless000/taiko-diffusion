from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .defaults import DEFAULT_DIFFICULTY, DEFAULT_GENERAL, DEFAULT_METADATA
from .hitobjects import build_hitobjects
from .timing import (
    build_timing_from_reference,
    infer_timing_from_notes,
    load_json,
    serialize_timing_point,
)


def make_general(metadata_json: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if metadata_json and "general" in metadata_json:
        out = dict(DEFAULT_GENERAL)
        out.update({k: str(v) for k, v in metadata_json["general"].items()})
        out["Mode"] = "1"
        return out
    return dict(DEFAULT_GENERAL)


def make_metadata(metadata_json: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if metadata_json and "metadata" in metadata_json:
        out = dict(DEFAULT_METADATA)
        out.update({k: str(v) for k, v in metadata_json["metadata"].items()})
        return out
    return dict(DEFAULT_METADATA)


def make_difficulty(
    metadata_json: Optional[Dict[str, Any]],
    timing_json: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    out = dict(DEFAULT_DIFFICULTY)
    if metadata_json and "difficulty" in metadata_json:
        out.update({k: str(v) for k, v in metadata_json["difficulty"].items()})
    if timing_json:
        if "slider_multiplier" in timing_json:
            out["SliderMultiplier"] = str(timing_json["slider_multiplier"])
        if "slider_tick_rate" in timing_json:
            out["SliderTickRate"] = str(timing_json["slider_tick_rate"])
    return out


def make_osu_text(
    general: Dict[str, str],
    metadata: Dict[str, str],
    difficulty: Dict[str, str],
    timing_points: List[Dict[str, Any]],
    hitobjects: List[str],
) -> str:
    lines: List[str] = ["osu file format v14", ""]

    lines.append("[General]")
    for k, v in general.items():
        lines.append(f"{k}:{v}")
    lines.append("")

    lines.append("[Editor]")
    lines.append("DistanceSpacing:1")
    lines.append("BeatDivisor:4")
    lines.append("GridSize:32")
    lines.append("TimelineZoom:1")
    lines.append("")

    lines.append("[Metadata]")
    for k, v in metadata.items():
        lines.append(f"{k}:{v}")
    lines.append("")

    lines.append("[Difficulty]")
    for k, v in difficulty.items():
        lines.append(f"{k}:{v}")
    lines.append("")

    lines.append("[Events]")
    lines.append("//Background and Video events")
    lines.append("//Break Periods")
    lines.append("")

    lines.append("[TimingPoints]")
    for tp in timing_points:
        lines.append(serialize_timing_point(tp))
    lines.append("")

    lines.append("[HitObjects]")
    lines.extend(hitobjects)
    lines.append("")

    return "\n".join(lines)


def reconstruct_osu(
    notes_path: Path,
    out_path: Path,
    timing_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
) -> None:
    notes_json = load_json(notes_path)
    if not notes_json:
        raise ValueError(f"Could not read notes file: {notes_path}")

    timing_json = load_json(timing_path)
    metadata_json = load_json(metadata_path)

    general = make_general(metadata_json)
    metadata = make_metadata(metadata_json)
    difficulty = make_difficulty(metadata_json, timing_json)

    slider_multiplier = float(difficulty.get("SliderMultiplier", "1.4"))

    if timing_json is not None:
        timing_points, slider_multiplier = build_timing_from_reference(timing_json)
        difficulty["SliderMultiplier"] = str(slider_multiplier)
    else:
        timing_points = infer_timing_from_notes(notes_json, slider_multiplier)

    hitobjects = build_hitobjects(notes_json)
    osu_text = make_osu_text(general, metadata, difficulty, timing_points, hitobjects)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(osu_text, encoding="utf-8")


def guess_related_path(notes_path: Path, suffix: str) -> Optional[Path]:
    name = notes_path.name
    if ".notes.json" in name:
        candidate = notes_path.with_name(name.replace(".notes.json", suffix))
        if candidate.exists():
            return candidate
    return None
