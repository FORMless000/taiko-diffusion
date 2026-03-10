#!/usr/bin/env python3
"""
Reconstruct an osu!taiko `.osu` file from the preprocessed JSON artifacts.

This script rebuilds a taiko beatmap from the generated note, timing, and
metadata JSON files. Only the notes file is required. If the timing file is
missing, the script infers timing from the note sequence. If explicit
"bpmchange" events are present in the notes, they are used as uninherited
timing points; inherited timing points are then inferred from baked absolute
scroll speed and volume changes. If metadata is missing, placeholder General /
Metadata / Difficulty fields are used. Reconstructed object times are rounded
to integer milliseconds.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_GENERAL = {
    "AudioFilename": "audio.mp3",
    "AudioLeadIn": "0",
    "PreviewTime": "-1",
    "Countdown": "0",
    "SampleSet": "Normal",
    "StackLeniency": "0.7",
    "Mode": "1",
    "LetterboxInBreaks": "0",
    "WidescreenStoryboard": "0",
}

DEFAULT_METADATA = {
    "Title": "Untitled",
    "TitleUnicode": "Untitled",
    "Artist": "Unknown Artist",
    "ArtistUnicode": "Unknown Artist",
    "Creator": "AutoReconstructed",
    "Version": "Reconstructed",
    "Source": "",
    "Tags": "reconstructed taiko",
    "BeatmapID": "0",
    "BeatmapSetID": "-1",
}

DEFAULT_DIFFICULTY = {
    "HPDrainRate": "5",
    "CircleSize": "5",
    "OverallDifficulty": "5",
    "ApproachRate": "5",
    "SliderMultiplier": "1.4",
    "SliderTickRate": "1",
}

PLACEHOLDER_UNINHERITED_MPB = 500.0
TAIKO_XY = (256, 192)


def load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def round_ms(value: float | int) -> int:
    return int(round(float(value)))


def note_time(note: Dict[str, Any]) -> int:
    return round_ms(note.get("time", note.get("raw_time", 0)))


def note_volume(note: Dict[str, Any]) -> int:
    return clamp(int(round(note.get("volume", 100))), 0, 100)


def note_sv(note: Dict[str, Any]) -> float:
    return float(note.get("sv", 1.0))


def is_bpm_change_event(note: Dict[str, Any]) -> bool:
    return str(note.get("type", "")).lower() == "bpmchange"


def note_bpm(note: Dict[str, Any]) -> float:
    return float(note.get("bpm", 120.0))


def note_meter(note: Dict[str, Any]) -> int:
    return int(note.get("meter", 4))


def hitsound_from_type(note_type: str) -> int:
    mapping = {
        "don": 0,
        "kat": 8,
        "bigdon": 4,
        "bigkat": 12,
    }
    return mapping.get(note_type, 0)


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


def sort_notes(notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    type_priority = {
        "bpmchange": 0,
        "don": 1,
        "kat": 2,
        "bigdon": 3,
        "bigkat": 4,
        "sliderstart": 5,
        "drumroll": 6,
        "sliderend": 7,
    }
    return sorted(
        notes,
        key=lambda n: (
            note_time(n),
            type_priority.get(str(n.get("type", "")).lower(), 999),
        ),
    )


def serialize_timing_point(tp: Dict[str, Any]) -> str:
    offset = round_ms(tp["offset"])
    ms_per_beat = float(tp["ms_per_beat"])
    meter = int(tp.get("meter", 4))
    sample_set = int(tp.get("sample_set", 1))
    sample_index = int(tp.get("sample_index", 0))
    volume = clamp(int(tp.get("volume", 100)), 0, 100)
    uninherited = int(tp.get("uninherited", 1))
    effects = int(tp.get("effects", 0))
    return f"{offset},{ms_per_beat:.15g},{meter},{sample_set},{sample_index},{volume},{uninherited},{effects}"


def build_timing_from_reference(
    timing_json: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], float]:
    slider_multiplier = float(timing_json.get("slider_multiplier", 1.4))
    timing_points = list(timing_json.get("timing_points", []))
    if not timing_points:
        timing_points = [{
            "offset": 0,
            "ms_per_beat": PLACEHOLDER_UNINHERITED_MPB,
            "meter": 4,
            "sample_set": 1,
            "sample_index": 0,
            "volume": 100,
            "uninherited": 1,
            "effects": 0,
        }]

    normalized = []
    for tp in timing_points:
        normalized.append({
            "offset": round_ms(tp.get("offset", tp.get("raw_offset", 0))),
            "ms_per_beat": float(tp["ms_per_beat"]),
            "meter": int(tp.get("meter", 4)),
            "sample_set": int(tp.get("sample_set", 1)),
            "sample_index": int(tp.get("sample_index", 0)),
            "volume": clamp(int(tp.get("volume", 100)), 0, 100),
            "uninherited": int(tp.get("uninherited", 1)),
            "effects": int(tp.get("effects", 0)),
        })
    normalized.sort(key=lambda x: (x["offset"], x["uninherited"]))
    return normalized, slider_multiplier


def infer_timing_from_notes(
    notes_json: Dict[str, Any],
    slider_multiplier: float,
) -> List[Dict[str, Any]]:
    notes = sort_notes(list(notes_json.get("notes", [])))
    if not notes:
        return [{
            "offset": 0,
            "ms_per_beat": PLACEHOLDER_UNINHERITED_MPB,
            "meter": 4,
            "sample_set": 1,
            "sample_index": 0,
            "volume": 100,
            "uninherited": 1,
            "effects": 0,
        }]

    bpm_events = [n for n in notes if is_bpm_change_event(n)]
    playable_notes = [n for n in notes if not is_bpm_change_event(n)]

    timing_points: List[Dict[str, Any]] = []

    if bpm_events:
        for ev in bpm_events:
            bpm = max(note_bpm(ev), 1e-6)
            mpb = 60000.0 / bpm
            timing_points.append({
                "offset": note_time(ev),
                "ms_per_beat": mpb,
                "meter": note_meter(ev),
                "sample_set": 1,
                "sample_index": 0,
                "volume": 100,
                "uninherited": 1,
                "effects": 0,
            })
    else:
        first_time = note_time(notes[0])
        base_volume = note_volume(playable_notes[0]) if playable_notes else 100
        timing_points.append({
            "offset": first_time,
            "ms_per_beat": PLACEHOLDER_UNINHERITED_MPB,
            "meter": 4,
            "sample_set": 1,
            "sample_index": 0,
            "volume": base_volume,
            "uninherited": 1,
            "effects": 0,
        })

    timing_points.sort(key=lambda x: (x["offset"], x["uninherited"]))

    def current_uninherited_at(time_ms: int) -> Dict[str, Any]:
        current = timing_points[0]
        for tp in timing_points:
            if tp["uninherited"] == 1 and tp["offset"] <= time_ms:
                current = tp
            elif tp["offset"] > time_ms:
                break
        return current

    prev_factor: Optional[float] = None
    prev_vol: Optional[int] = None

    for note in playable_notes:
        t = note_time(note)
        abs_sv = note_sv(note)
        vol = note_volume(note)

        base_tp = current_uninherited_at(t)
        bpm = 60000.0 / float(base_tp["ms_per_beat"]) if float(base_tp["ms_per_beat"]) > 0 else 120.0
        denom = slider_multiplier * bpm

        inherited_factor = 1.0
        if denom > 0:
            inherited_factor = abs_sv / denom
        inherited_factor = max(0.01, min(10.0, inherited_factor))
        inherited_mpb = -100.0 / inherited_factor

        changed = False
        if prev_factor is None or not math.isclose(inherited_factor, prev_factor, rel_tol=1e-6, abs_tol=1e-6):
            changed = True
        if prev_vol is None or vol != prev_vol:
            changed = True

        if changed:
            timing_points.append({
                "offset": t,
                "ms_per_beat": inherited_mpb,
                "meter": 4,
                "sample_set": 1,
                "sample_index": 0,
                "volume": vol,
                "uninherited": 0,
                "effects": 0,
            })
            prev_factor = inherited_factor
            prev_vol = vol

    timing_points.sort(key=lambda x: (x["offset"], x["uninherited"]))

    deduped: List[Dict[str, Any]] = []
    for tp in timing_points:
        if deduped:
            prev = deduped[-1]
            same = (
                prev["offset"] == tp["offset"]
                and math.isclose(prev["ms_per_beat"], tp["ms_per_beat"], rel_tol=1e-9, abs_tol=1e-9)
                and prev["volume"] == tp["volume"]
                and prev["uninherited"] == tp["uninherited"]
                and prev["meter"] == tp["meter"]
            )
            if same:
                continue
        deduped.append(tp)

    return deduped


def build_hitobjects(notes_json: Dict[str, Any]) -> List[str]:
    notes = sort_notes(list(notes_json.get("notes", [])))
    lines: List[str] = []

    i = 0
    while i < len(notes):
        note = notes[i]
        ntype = str(note.get("type", "")).lower()

        if ntype == "bpmchange":
            i += 1
            continue

        t = note_time(note)

        if ntype in {"don", "kat", "bigdon", "bigkat"}:
            hitsound = hitsound_from_type(ntype)
            lines.append(f"{TAIKO_XY[0]},{TAIKO_XY[1]},{t},1,{hitsound},0:0:0:0:")
            i += 1
            continue

        if ntype == "sliderstart":
            end_time = t
            j = i + 1
            while j < len(notes):
                other = notes[j]
                otype = str(other.get("type", "")).lower()
                if otype == "bpmchange":
                    j += 1
                    continue
                if otype == "sliderend":
                    end_time = note_time(other)
                    break
                j += 1

            if end_time < t:
                end_time = t

            lines.append(f"{TAIKO_XY[0]},{TAIKO_XY[1]},{t},8,0,{end_time},0:0:0:0:")
            i = j + 1 if j < len(notes) else i + 1
            continue

        if ntype == "drumroll":
            end_time = t
            j = i + 1
            while j < len(notes):
                other = notes[j]
                otype = str(other.get("type", "")).lower()
                if otype == "bpmchange":
                    j += 1
                    continue
                if otype == "sliderend":
                    end_time = note_time(other)
                    break
                j += 1

            if end_time < t:
                end_time = t

            lines.append(f"{TAIKO_XY[0]},{TAIKO_XY[1]},{t},8,0,{end_time},0:0:0:0:")
            i = j + 1 if j < len(notes) else i + 1
            continue

        if ntype == "sliderend":
            i += 1
            continue

        i += 1

    return lines


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
    """
    Reconstruct a single `.osu` file from generated taiko note / timing /
    metadata JSON files.

    The notes file is required. If timing reference data is missing, a minimal
    timing stream is inferred from note timestamps. If explicit "bpmchange"
    events are present in the notes file, they are used as uninherited timing
    points, and inherited timing points are inferred from baked absolute scroll
    speed and volume changes. If metadata is missing, placeholder values are
    used. All reconstructed event and timing-point offsets are rounded to
    integer milliseconds before writing the output beatmap.
    """
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


if __name__ == "__main__":
    main()