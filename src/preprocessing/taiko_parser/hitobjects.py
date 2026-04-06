from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .sections import safe_float, safe_int
from .timing import (
    TimingPoint,
    absolute_scroll_speed_at,
    bpm_from_ms_per_beat,
    snap_time_to_grid,
    previous_uninherited_at,
    active_inherited_factor_at,
    volume_at,
)


@dataclass
class NoteEvent:
    type: str
    time: float
    raw_time: int
    sv: float
    volume: int
    bpm: Optional[float] = None
    meter: Optional[int] = None


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
