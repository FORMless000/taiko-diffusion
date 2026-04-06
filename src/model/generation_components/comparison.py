from __future__ import annotations

import json
import math
from pathlib import Path


def compare_song_output_with_notes_json(song_output, gt_json_path, max_sequences=10):
    gt_json_path = Path(gt_json_path)
    if not gt_json_path.exists():
        raise FileNotFoundError(f"File not found: {gt_json_path}")

    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    if "notes" not in gt_data:
        raise ValueError("This file is not a raw notes.json with top-level key 'notes'.")

    raw_notes = gt_data["notes"]

    bpm_events = [x for x in raw_notes if x.get("type") == "bpmchange"]
    if len(bpm_events) == 0:
        raise ValueError("No bpmchange found in notes.json, cannot infer timing.")

    first_bpm = bpm_events[0]
    offset_ms = float(first_bpm["time"])
    bpm = float(first_bpm["bpm"])
    meter = int(first_bpm["meter"]) if first_bpm["meter"] is not None else 4

    if bpm <= 0:
        raise ValueError("Invalid bpm in bpmchange.")

    beat_duration_ms = 60000.0 / bpm
    tick_ms = beat_duration_ms / 48.0
    seq_ticks = 192
    seq_duration_ms = tick_ms * seq_ticks

    type_to_token = {
        "don": "DON",
        "kat": "KAT",
        "bigdon": "BIGDON",
        "bigkat": "BIGKAT",
        "drumroll": "DRUMROLL",
        "sliderstart": "SLIDERSTART",
        "sliderend": "SLIDEREND",
    }

    event_notes = []
    for note in raw_notes:
        note_type = note.get("type")
        if note_type in type_to_token:
            event_notes.append(
                {
                    "time": float(note["time"]),
                    "token": type_to_token[note_type],
                }
            )

    seq_to_events = {}

    for ev in event_notes:
        rel_ms = ev["time"] - offset_ms
        seq_idx = int(math.floor((rel_ms + 1e-6) / seq_duration_ms))
        seq_idx = max(seq_idx, 0)

        seq_start_ms = offset_ms + seq_idx * seq_duration_ms
        rel_in_seq_ms = ev["time"] - seq_start_ms
        pos_tick = int(round(rel_in_seq_ms / tick_ms))
        pos_tick = max(0, min(seq_ticks - 1, pos_tick))

        seq_to_events.setdefault(seq_idx, []).append((pos_tick, ev["token"]))

    gt_song_output = []
    max_seq_idx_from_gt = max(seq_to_events.keys()) if seq_to_events else -1
    total_gt_sequences = max_seq_idx_from_gt + 1

    for seq_idx in range(total_gt_sequences):
        events = seq_to_events.get(seq_idx, [])
        events = sorted(events, key=lambda x: (x[0], x[1]))

        tokens = []
        cursor = 0

        for pos_tick, token in events:
            gap = pos_tick - cursor
            if gap > 0:
                tokens.append(f"TS_{gap}")
            tokens.append(token)
            cursor = pos_tick

        gt_song_output.append(
            {
                "seq_idx": seq_idx,
                "start_frame": seq_idx * 192,
                "end_frame": seq_idx * 192 + 191,
                "tokens": tokens,
            }
        )

    n = min(len(song_output), len(gt_song_output), max_sequences)

    lines = []
    lines.append(f"offset_ms={offset_ms}, bpm={bpm}, meter={meter}")
    lines.append("=" * 80)

    for i in range(n):
        gt_tokens = gt_song_output[i]["tokens"]
        pred_tokens = song_output[i]["pred_tokens"]

        gt_str = " ".join(gt_tokens)
        pred_str = " ".join(pred_tokens)

        lines.append(f"Sequence {i}")
        lines.append(f"GT  : {gt_str}")
        lines.append(f"PRED: {pred_str}")
        lines.append("-" * 80)

    return "\n".join(lines)
