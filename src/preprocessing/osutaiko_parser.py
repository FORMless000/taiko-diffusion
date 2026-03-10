"""
Taiko beatmap preprocessing utilities.

This module parses osu!taiko `.osu` beatmaps into three simplified artifacts:

(1) a note sequence file used for model training, where each note stores its
taiko type, snapped timestamp, absolute baked scroll speed, and volume;

(2) a timing-point reference file that preserves timing-point information
essentially as-is, while also storing snapped offsets for later reconstruction
and analysis;

(3) a metadata reference file containing the General,
Metadata, and Difficulty blocks needed to reconstruct beatmaps later.

During training, only the note sequence is intended to be consumed directly,
while the timing and metadata files are kept as auxiliary references.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class TimingPoint:
    offset: float
    raw_offset: int
    ms_per_beat: float
    meter: int
    sample_set: int
    sample_index: int
    volume: int
    uninherited: int
    effects: int


@dataclass
class NoteEvent:
    type: str
    time: float
    raw_time: int
    sv: float
    volume: int
    bpm: Optional[float] = None
    meter: Optional[int] = None


def parse_key_value_section(lines: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def split_sections(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1]
            sections.setdefault(current, [])
            continue

        if current is not None:
            sections[current].append(line)

    return sections


def safe_int(x: str, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def parse_raw_timing_points(lines: List[str]) -> List[dict]:
    points: List[dict] = []

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("//"):
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue

        points.append(
            {
                "offset": round(safe_float(parts[0])),
                "ms_per_beat": safe_float(parts[1]),
                "meter": safe_int(parts[2], 4),
                "sample_set": safe_int(parts[3], 1),
                "sample_index": safe_int(parts[4], 0),
                "volume": safe_int(parts[5], 100),
                "uninherited": safe_int(parts[6], 1),
                "effects": safe_int(parts[7], 0),
            }
        )

    points.sort(key=lambda x: x["offset"])
    return points


def get_uninherited_points(raw_timing_points: List[dict]) -> List[dict]:
    return [tp for tp in raw_timing_points if tp["uninherited"] == 1]


def previous_uninherited_at(time_ms: int, raw_timing_points: List[dict]) -> dict:
    uninherited = get_uninherited_points(raw_timing_points)
    current = uninherited[0]
    for tp in uninherited:
        if tp["offset"] <= time_ms:
            current = tp
        else:
            break
    return current


def active_timing_point_at(time_ms: int, raw_timing_points: List[dict]) -> dict:
    current = raw_timing_points[0]
    for tp in raw_timing_points:
        if tp["offset"] <= time_ms:
            current = tp
        else:
            break
    return current


def active_inherited_factor_at(time_ms: int, raw_timing_points: List[dict]) -> float:
    tp = active_timing_point_at(time_ms, raw_timing_points)
    if tp["uninherited"] == 1 or tp["ms_per_beat"] >= 0:
        return 1.0
    return max(0.01, min(10.0, -100.0 / tp["ms_per_beat"]))


def bpm_at(time_ms: int, raw_timing_points: List[dict]) -> float:
    tp = previous_uninherited_at(time_ms, raw_timing_points)
    if tp["ms_per_beat"] <= 0:
        return 0.0
    return 60000.0 / tp["ms_per_beat"]


def absolute_scroll_speed_at(
    time_ms: int,
    raw_timing_points: List[dict],
    slider_multiplier: float,
) -> float:
    base_sv = slider_multiplier
    bpm = bpm_at(time_ms, raw_timing_points)
    inherited_factor = active_inherited_factor_at(time_ms, raw_timing_points)
    return base_sv * bpm * inherited_factor


def volume_at(time_ms: int, raw_timing_points: List[dict]) -> int:
    return active_timing_point_at(time_ms, raw_timing_points)["volume"]


def snap_time_to_grid(
    time_ms: float,
    raw_timing_points: List[dict],
    *,
    max_divisor: int = 48,
    tolerance_ms: int = 2,
) -> float:
    base_tp = previous_uninherited_at(time_ms, raw_timing_points)
    base_offset = base_tp["offset"]
    beat_len = base_tp["ms_per_beat"]

    if beat_len <= 0:
        return time_ms

    best_time = float(time_ms)
    best_error = float("inf")
    beats = (time_ms - base_offset) / beat_len

    for divisor in range(1, max_divisor + 1):
        snapped_beats = round(beats * divisor) / divisor
        candidate = base_offset + snapped_beats * beat_len
        error = abs(candidate - time_ms)

        if error < best_error:
            best_error = error
            best_time = candidate

    if best_error <= tolerance_ms:
        return best_time

    return float(time_ms)


def snap_raw_timing_points(raw_timing_points: List[dict]) -> List[TimingPoint]:
    snapped: List[TimingPoint] = []

    for i, tp in enumerate(raw_timing_points):
        raw_offset = int(tp["offset"])

        if i == 0:
            snapped_offset = raw_offset
        else:
            snapped_offset = snap_time_to_grid(raw_offset, raw_timing_points)

        snapped.append(
            TimingPoint(
                offset=snapped_offset,
                raw_offset=raw_offset,
                ms_per_beat=tp["ms_per_beat"],
                meter=tp["meter"],
                sample_set=tp["sample_set"],
                sample_index=tp["sample_index"],
                volume=tp["volume"],
                uninherited=tp["uninherited"],
                effects=tp["effects"],
            )
        )

    snapped.sort(key=lambda x: x.offset)
    return snapped


OBJ_CIRCLE = 1
OBJ_SLIDER = 2
OBJ_NEW_COMBO = 4
OBJ_SPINNER = 8
OBJ_HOLD = 128

HS_WHISTLE = 2
HS_FINISH = 4
HS_CLAP = 8


def taiko_circle_type(hit_sound: int) -> str:
    is_kat = bool(hit_sound & (HS_WHISTLE | HS_CLAP))
    is_big = bool(hit_sound & HS_FINISH)

    if is_kat and is_big:
        return "bigkat"
    if is_kat:
        return "kat"
    if is_big:
        return "bigdon"
    return "don"


def slider_duration_ms(
    start_time_ms: int,
    pixel_length: float,
    repeats: int,
    raw_timing_points: List[dict],
    slider_multiplier: float,
) -> int:
    uninherited_tp = previous_uninherited_at(start_time_ms, raw_timing_points)
    beat_length = uninherited_tp["ms_per_beat"]
    inherited_factor = active_inherited_factor_at(start_time_ms, raw_timing_points)

    if slider_multiplier <= 0:
        slider_multiplier = 1.0

    per_span = (pixel_length / (slider_multiplier * 100.0 * inherited_factor)) * beat_length
    total = per_span * max(repeats, 1)
    return round(total)


def parse_hit_objects_taiko(
    lines: List[str],
    raw_timing_points: List[dict],
    slider_multiplier: float,
) -> List[NoteEvent]:
    notes: List[NoteEvent] = []

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("//"):
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue

        raw_time = safe_int(parts[2])
        time_ms = raw_time if len(notes) == 0 else snap_time_to_grid(raw_time, raw_timing_points)

        obj_type = safe_int(parts[3])
        hit_sound = safe_int(parts[4])

        current_sv = absolute_scroll_speed_at(time_ms, raw_timing_points, slider_multiplier)
        current_volume = volume_at(time_ms, raw_timing_points)

        if obj_type & OBJ_CIRCLE:
            notes.append(
                NoteEvent(
                    type=taiko_circle_type(hit_sound),
                    time=time_ms,
                    raw_time=raw_time,
                    sv=current_sv,
                    volume=current_volume,
                )
            )
            continue

        if obj_type & OBJ_SLIDER:
            repeats = safe_int(parts[6], 1) if len(parts) > 6 else 1
            pixel_length = safe_float(parts[7], 0.0) if len(parts) > 7 else 0.0

            duration = slider_duration_ms(
                start_time_ms=time_ms,
                pixel_length=pixel_length,
                repeats=repeats,
                raw_timing_points=raw_timing_points,
                slider_multiplier=slider_multiplier,
            )
            raw_end_time = raw_time + duration
            end_time = snap_time_to_grid(raw_end_time, raw_timing_points)

            notes.append(
                NoteEvent(
                    type="sliderstart",
                    time=time_ms,
                    raw_time=raw_time,
                    sv=current_sv,
                    volume=current_volume,
                )
            )
            notes.append(
                NoteEvent(
                    type="sliderend",
                    time=end_time,
                    raw_time=raw_end_time,
                    sv=absolute_scroll_speed_at(end_time, raw_timing_points, slider_multiplier),
                    volume=volume_at(end_time, raw_timing_points),
                )
            )
            continue

        if obj_type & OBJ_SPINNER:
            raw_end_time = safe_int(parts[5]) if len(parts) > 5 else raw_time
            end_time = snap_time_to_grid(raw_end_time, raw_timing_points)

            notes.append(
                NoteEvent(
                    type="drumroll",
                    time=time_ms,
                    raw_time=raw_time,
                    sv=current_sv,
                    volume=current_volume,
                )
            )
            notes.append(
                NoteEvent(
                    type="sliderend",
                    time=end_time,
                    raw_time=raw_end_time,
                    sv=absolute_scroll_speed_at(end_time, raw_timing_points, slider_multiplier),
                    volume=volume_at(end_time, raw_timing_points),
                )
            )
            continue

        if obj_type & OBJ_HOLD:
            continue

    notes.sort(key=lambda n: (n.time, n.type))
    return notes


def bpm_from_ms_per_beat(ms_per_beat: float) -> float:
    if ms_per_beat <= 0:
        return 0.0
    return 60000.0 / ms_per_beat


def append_bpm_change_events(
    notes: List[NoteEvent],
    snapped_timing_points: List[TimingPoint],
    raw_timing_points: List[dict],
    slider_multiplier: float,
) -> List[NoteEvent]:
    merged = list(notes)

    for tp in snapped_timing_points:
        if tp.uninherited != 1:
            continue

        event_sv = absolute_scroll_speed_at(tp.offset, raw_timing_points, slider_multiplier)
        event_volume = volume_at(tp.offset, raw_timing_points)

        merged.append(
            NoteEvent(
                type="bpmchange",
                time=tp.offset,
                raw_time=tp.raw_offset,
                sv=event_sv,
                volume=event_volume,
                bpm=bpm_from_ms_per_beat(tp.ms_per_beat),
                meter=tp.meter,
            )
        )

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

    merged.sort(key=lambda n: (n.time, type_priority.get(n.type, 999), n.raw_time))
    return merged


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
    """
    Parse a single osu!taiko beatmap into three output objects: metadata,
    timing-point reference data, and a training-oriented note sequence.

    The generated note sequence is a taiko-only representation that keeps one
    event stream of notes and long-note endpoints, with absolute scroll speed
    already baked into each event and timestamps optionally snapped to the
    nearest beat subdivision when within tolerance. Optionally, uninherited
    timing points can also be appended into the note stream as explicit
    "bpmchange" events carrying the snapped timing, BPM, and beat signature.
    Timing points are exported separately as reference data so the original
    chart structure can be reconstructed later, while metadata is also stored
    separately for the same reason.
    """
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
