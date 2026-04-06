from __future__ import annotations

from typing import Any, Dict, List

from .defaults import TAIKO_XY
from .timing import is_bpm_change_event, note_time, sort_notes


def hitsound_from_type(note_type: str) -> int:
    mapping = {
        "don": 0,
        "kat": 8,
        "bigdon": 4,
        "bigkat": 12,
    }
    return mapping.get(note_type, 0)


def build_hitobjects(notes_json: Dict[str, Any]) -> List[str]:
    notes = sort_notes(list(notes_json.get("notes", [])))
    lines: List[str] = []

    i = 0
    while i < len(notes):
        note = notes[i]
        ntype = str(note.get("type", "")).lower()

        if is_bpm_change_event(note):
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
