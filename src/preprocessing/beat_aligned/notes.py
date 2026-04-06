from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from .common import require_file
from .config import ALLOWED_EVENT_TYPES, FRAMES_PER_BEAT, FRAMES_PER_SEQUENCE, MODEL_EVENT_TYPES, NotesInfo


def load_note_events(notes_path: Path) -> List[Dict[str, Any]]:
    require_file(notes_path, "notes.json")

    with open(notes_path, "r", encoding="utf-8") as f:
        notes_data = json.load(f)

    if isinstance(notes_data, list):
        events = notes_data
    elif isinstance(notes_data, dict):
        if "notes" in notes_data:
            events = notes_data["notes"]
        elif "events" in notes_data:
            events = notes_data["events"]
        elif "hit_objects" in notes_data:
            events = notes_data["hit_objects"]
        else:
            raise ValueError(f"Unknown notes.json structure. Keys: {list(notes_data.keys())}")
    else:
        raise ValueError("Unsupported notes.json structure")

    if not events:
        raise ValueError("No events found in notes.json")

    return events


def compute_notes_info(
    events: Sequence[Dict[str, Any]],
    offset_ms: float,
    beat_duration_ms: float,
    total_frames: int,
) -> Tuple[NotesInfo, pd.DataFrame, pd.DataFrame]:
    events_df = pd.DataFrame(events).copy()
    if "type" not in events_df.columns or "time" not in events_df.columns:
        raise ValueError(f"Required event fields missing. Columns: {list(events_df.columns)}")

    events_df["time"] = events_df["time"].astype(float)
    events_df["beat_position"] = (events_df["time"] - offset_ms) / beat_duration_ms
    events_df["frame_position"] = events_df["beat_position"] * FRAMES_PER_BEAT
    events_df["frame_index_rounded"] = events_df["frame_position"].round().astype(int)

    event_type_counts = events_df["type"].value_counts().to_dict()
    unknown_event_types = sorted(set(events_df["type"]) - ALLOWED_EVENT_TYPES)

    model_df = events_df[events_df["type"].isin(MODEL_EVENT_TYPES)].copy()
    model_df = model_df.sort_values(["frame_index_rounded", "time"]).reset_index(drop=True)
    frame_duration_ms = beat_duration_ms / FRAMES_PER_BEAT
    model_df["nearest_frame_time_ms"] = offset_ms + model_df["frame_index_rounded"] * frame_duration_ms
    model_df["offgrid_abs_error_ms"] = (model_df["time"] - model_df["nearest_frame_time_ms"]).abs()

    outside_df = model_df[
        (model_df["frame_index_rounded"] < 0)
        | (model_df["frame_index_rounded"] >= total_frames)
    ]

    frame_counts = model_df["frame_index_rounded"].value_counts()
    collision_frames = frame_counts[frame_counts > 1]

    min_model_frame = int(model_df["frame_index_rounded"].min()) if not model_df.empty else None
    max_model_frame = int(model_df["frame_index_rounded"].max()) if not model_df.empty else None
    n_at_frame0 = int((model_df["frame_index_rounded"] == 0).sum()) if not model_df.empty else 0
    n_at_last_frame = int((model_df["frame_index_rounded"] == total_frames - 1).sum()) if not model_df.empty else 0

    notes_info = NotesInfo(
        total_events=int(len(events_df)),
        model_events=int(len(model_df)),
        unknown_event_types=unknown_event_types,
        min_model_frame=min_model_frame,
        max_model_frame=max_model_frame,
        outside_event_count=int(len(outside_df)),
        collision_frame_count=int(len(collision_frames)),
        collision_event_total=int(collision_frames.sum()) if not collision_frames.empty else 0,
        n_at_frame0=n_at_frame0,
        n_at_last_frame=n_at_last_frame,
        event_type_counts={str(k): int(v) for k, v in event_type_counts.items()},
    )
    return notes_info, events_df, model_df


def build_per_sequence_event_tokens(model_df: pd.DataFrame, total_sequences: int) -> List[Dict[str, Any]]:
    token_data: List[Dict[str, Any]] = []

    for seq_idx in range(total_sequences):
        seq_start_frame = seq_idx * FRAMES_PER_SEQUENCE
        seq_end_frame = seq_start_frame + FRAMES_PER_SEQUENCE - 1

        seq_events = model_df[
            (model_df["frame_index_rounded"] >= seq_start_frame)
            & (model_df["frame_index_rounded"] <= seq_end_frame)
        ].copy()
        seq_events["local_frame"] = seq_events["frame_index_rounded"] - seq_start_frame
        seq_events = seq_events.sort_values(["local_frame", "time"]).reset_index(drop=True)

        tokens: List[str] = []
        prev_local_frame = 0
        for _, row in seq_events.iterrows():
            local_frame = int(row["local_frame"])
            event_type = str(row["type"]).upper()
            time_shift = local_frame - prev_local_frame
            if time_shift > 0:
                tokens.append(f"TS_{time_shift}")
            tokens.append(event_type)
            prev_local_frame = local_frame

        token_data.append(
            {
                "seq_idx": seq_idx,
                "start_frame": seq_start_frame,
                "end_frame": seq_end_frame,
                "n_events": int(len(seq_events)),
                "n_tokens": int(len(tokens)),
                "tokens": tokens,
            }
        )

    return token_data
