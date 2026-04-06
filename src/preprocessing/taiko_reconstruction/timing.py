from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .defaults import PLACEHOLDER_UNINHERITED_MPB


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
