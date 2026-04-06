from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .sections import safe_float, safe_int


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


def bpm_from_ms_per_beat(ms_per_beat: float) -> float:
    if ms_per_beat <= 0:
        return 0.0
    return 60000.0 / ms_per_beat
