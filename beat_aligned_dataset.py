#!/usr/bin/env python3
"""
Beat-aligned dataset builder for Taiko chart generation.

This script performs the following workflow:
1. Traverse the unpacked dataset folder and generate an internal chart mapping table.
2. Parse timing information from each timing.json.
3. Compute beat-grid information.
4. Parse and analyze note events.
5. Build a beat-aligned frame timeline.
6. Build a raw mel spectrogram from the audio file.
7. Interpolate the raw mel spectrogram onto the beat-aligned frame timeline.
8. Segment the aligned mel spectrogram into 4-beat sequences.
9. Build per-sequence event tokens.

Outputs are written into two top-level folders under the data root:
- chart_index
- beat_aligned_dataset

The script is designed for GitHub use:
- clear English comments
- modular functions
- conservative logging
- error handling at both chart and pipeline levels
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd


# ============================================================================
# Default paths and global configuration
# ============================================================================

DEFAULT_DATA_ROOT = Path(r"D:\Study Abroad\course\DSCI498\Project\data")
DEFAULT_UNPACKED_ROOT = DEFAULT_DATA_ROOT / "unpacked"
DEFAULT_INDEX_DIR = DEFAULT_DATA_ROOT / "chart_index"
DEFAULT_DATASET_DIR = DEFAULT_DATA_ROOT / "beat_aligned_dataset"

FRAMES_PER_BEAT = 48
SEQUENCE_BEATS = 4
FRAMES_PER_SEQUENCE = FRAMES_PER_BEAT * SEQUENCE_BEATS

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

MODEL_EVENT_TYPES = {
    "don",
    "kat",
    "bigdon",
    "bigkat",
    "drumroll",
    "sliderstart",
    "sliderend",
}

ALLOWED_EVENT_TYPES = MODEL_EVENT_TYPES | {"bpmchange"}


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class TimingInfo:
    offset_ms: float
    beat_duration_ms: float
    bpm: float
    meter: int
    n_bpm_points: int
    n_timing_points: int


@dataclass
class BeatGridInfo:
    total_beats: int
    total_frames: int
    total_sequences: int
    frame_duration_ms: float
    last_beat_time_ms: float
    last_frame_time_ms: float
    remaining_tail_ms: float
    frame_overshoot_ms: float


@dataclass
class NotesInfo:
    total_events: int
    model_events: int
    unknown_event_types: List[str]
    min_model_frame: Optional[int]
    max_model_frame: Optional[int]
    outside_event_count: int
    collision_frame_count: int
    collision_event_total: int
    n_at_frame0: int
    n_at_last_frame: int
    event_type_counts: Dict[str, int]


# ============================================================================
# Utility helpers
# ============================================================================


def setup_logging() -> None:
    """Configure concise logging for command-line execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )



def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)



def sanitize_filename(text: str, max_length: int = 150) -> str:
    """Convert an arbitrary chart name into a filesystem-safe stem."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    if not safe:
        safe = "chart"
    return safe[:max_length]



def chart_uid(folder_id: Any, chart_base: str) -> str:
    """Build a stable chart identifier for saved outputs."""
    return f"{folder_id}_{sanitize_filename(chart_base)}"



def safe_json_dump(obj: Any, path: Path) -> None:
    """Write JSON with UTF-8 encoding and readable indentation."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)



def require_file(path: Path, label: str) -> None:
    """Raise a clear error if a required file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


# ============================================================================
# Step 1: build mapping table by traversing the unpacked folder
# ============================================================================


def build_chart_mapping_table(unpacked_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Traverse the unpacked dataset directory and build a chart-level mapping table.

    Expected folder structure:
        unpacked/<folder_id>/
            <audio>.mp3 or <audio>.ogg
            parsed/
                <chart_base>.notes.json
                <chart_base>.timing.json
                <chart_base>.metadata.json

    Returns
    -------
    mapping_df:
        One row per complete chart triple.
    issues_df:
        Folder-level issues such as missing audio, multiple audio files,
        or incomplete chart triples.
    """
    rows: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []

    if not unpacked_root.exists():
        raise FileNotFoundError(f"Unpacked root not found: {unpacked_root}")

    folder_paths = sorted([p for p in unpacked_root.iterdir() if p.is_dir()])
    logging.info("Scanning unpacked folders: %d", len(folder_paths))

    for folder_path in folder_paths:
        parsed_path = folder_path / "parsed"
        folder_id = folder_path.name

        audio_files = sorted(list(folder_path.glob("*.mp3")) + list(folder_path.glob("*.ogg")))
        if len(audio_files) != 1:
            issues.append(
                {
                    "folder_id": folder_id,
                    "folder_path": str(folder_path),
                    "issue_type": "audio_count_error",
                    "issue_detail": f"Expected 1 audio file, found {len(audio_files)}",
                }
            )
            continue

        audio_path = audio_files[0]

        if not parsed_path.exists():
            issues.append(
                {
                    "folder_id": folder_id,
                    "folder_path": str(folder_path),
                    "issue_type": "missing_parsed_folder",
                    "issue_detail": "parsed/ folder not found",
                }
            )
            continue

        notes_files = sorted(parsed_path.glob("*.notes.json"))
        timing_files = sorted(parsed_path.glob("*.timing.json"))
        metadata_files = sorted(parsed_path.glob("*.metadata.json"))

        notes_map = {p.name[:-11]: p for p in notes_files}      # strip '.notes.json'
        timing_map = {p.name[:-12]: p for p in timing_files}    # strip '.timing.json'
        metadata_map = {p.name[:-14]: p for p in metadata_files}  # strip '.metadata.json'

        chart_bases = sorted(set(notes_map) | set(timing_map) | set(metadata_map))

        if not chart_bases:
            issues.append(
                {
                    "folder_id": folder_id,
                    "folder_path": str(folder_path),
                    "issue_type": "no_parsed_charts",
                    "issue_detail": "No parsed chart files found",
                }
            )
            continue

        for base in chart_bases:
            notes_path = notes_map.get(base)
            timing_path = timing_map.get(base)
            metadata_path = metadata_map.get(base)

            if not (notes_path and timing_path and metadata_path):
                issues.append(
                    {
                        "folder_id": folder_id,
                        "folder_path": str(folder_path),
                        "issue_type": "incomplete_chart_triple",
                        "issue_detail": (
                            f"chart_base={base}; "
                            f"notes={notes_path is not None}; "
                            f"timing={timing_path is not None}; "
                            f"metadata={metadata_path is not None}"
                        ),
                    }
                )
                continue

            rows.append(
                {
                    "folder_id": folder_id,
                    "folder_path": str(folder_path),
                    "audio_file": audio_path.name,
                    "audio_path": str(audio_path),
                    "chart_base": base,
                    "notes_path": str(notes_path),
                    "timing_path": str(timing_path),
                    "metadata_path": str(metadata_path),
                }
            )

    mapping_df = pd.DataFrame(rows)
    issues_df = pd.DataFrame(issues)
    return mapping_df, issues_df


# ============================================================================
# Step 2: timing information
# ============================================================================


def get_timing_info(timing_path: Path) -> TimingInfo:
    """
    Parse a timing.json file and extract constant-BPM timing information.

    Raises
    ------
    ValueError
        If timing points are missing or if the chart contains BPM changes.
    """
    require_file(timing_path, "timing.json")

    with open(timing_path, "r", encoding="utf-8") as f:
        timing_data = json.load(f)

    timing_points = timing_data.get("timing_points", [])
    if not timing_points:
        raise ValueError("No timing_points found")

    bpm_points = [tp for tp in timing_points if int(tp.get("uninherited", 0)) == 1]
    if not bpm_points:
        raise ValueError("No BPM timing points found (uninherited=1)")

    bpm_points = sorted(bpm_points, key=lambda x: float(x["offset"]))
    unique_ms_per_beat = sorted({round(float(tp["ms_per_beat"]), 10) for tp in bpm_points})

    if len(unique_ms_per_beat) != 1:
        raise ValueError(f"Non-constant BPM detected: {unique_ms_per_beat}")

    beat_duration_ms = float(bpm_points[0]["ms_per_beat"])
    offset_ms = float(bpm_points[0]["offset"])
    meter = int(bpm_points[0].get("meter", 4))
    bpm = 60000.0 / beat_duration_ms

    return TimingInfo(
        offset_ms=offset_ms,
        beat_duration_ms=beat_duration_ms,
        bpm=bpm,
        meter=meter,
        n_bpm_points=len(bpm_points),
        n_timing_points=len(timing_points),
    )


# ============================================================================
# Audio helper
# ============================================================================


def get_audio_info(audio_path: Path) -> Dict[str, Any]:
    """Load audio and return waveform plus basic audio metadata."""
    require_file(audio_path, "audio file")

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    n_samples = len(y)
    audio_duration_sec = n_samples / sr
    audio_duration_ms = audio_duration_sec * 1000.0

    return {
        "waveform": y,
        "sample_rate": sr,
        "n_samples": n_samples,
        "audio_duration_sec": audio_duration_sec,
        "audio_duration_ms": audio_duration_ms,
    }


# ============================================================================
# Step 3: beat grid information
# ============================================================================


def compute_beat_grid_info(
    offset_ms: float,
    beat_duration_ms: float,
    audio_duration_ms: float,
) -> Tuple[BeatGridInfo, np.ndarray]:
    """
    Compute beat times and beat-grid summary values for one chart.

    The beat grid starts from the first BPM timing point offset and expands
    forward until the audio duration is reached.
    """
    beat_times_ms: List[float] = []
    t = offset_ms
    while t < audio_duration_ms:
        beat_times_ms.append(t)
        t += beat_duration_ms

    if not beat_times_ms:
        raise ValueError("No beat times generated; check offset and audio duration")

    beat_times_ms_arr = np.array(beat_times_ms, dtype=np.float64)
    total_beats = len(beat_times_ms_arr)
    total_frames = total_beats * FRAMES_PER_BEAT
    total_sequences = total_frames // FRAMES_PER_SEQUENCE
    frame_duration_ms = beat_duration_ms / FRAMES_PER_BEAT
    last_frame_time_ms = offset_ms + (total_frames - 1) * frame_duration_ms

    info = BeatGridInfo(
        total_beats=total_beats,
        total_frames=total_frames,
        total_sequences=total_sequences,
        frame_duration_ms=frame_duration_ms,
        last_beat_time_ms=float(beat_times_ms_arr[-1]),
        last_frame_time_ms=float(last_frame_time_ms),
        remaining_tail_ms=float(audio_duration_ms - beat_times_ms_arr[-1]),
        frame_overshoot_ms=float(last_frame_time_ms - audio_duration_ms),
    )
    return info, beat_times_ms_arr


# ============================================================================
# Step 4: note information
# ============================================================================


def load_note_events(notes_path: Path) -> List[Dict[str, Any]]:
    """Load note events from notes.json, supporting common top-level formats."""
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
    """
    Convert note times into beat-relative frame coordinates and summarize them.

    Returns
    -------
    notes_info:
        Chart-level summary of event quality and counts.
    events_df:
        All events mapped to frame coordinates.
    model_df:
        Only modeling events mapped to frame coordinates.
    """
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


# ============================================================================
# Step 5: beat-aligned frame timeline
# ============================================================================


def build_beat_aligned_frame_timeline(
    offset_ms: float,
    beat_duration_ms: float,
    total_frames: int,
) -> np.ndarray:
    """
    Build the target frame timeline where 1 beat = 48 frames.

    The resulting array has length total_frames and stores the real time (ms)
    corresponding to each beat-aligned frame.
    """
    frame_duration_ms = beat_duration_ms / FRAMES_PER_BEAT
    frame_times_ms = offset_ms + np.arange(total_frames, dtype=np.float64) * frame_duration_ms
    return frame_times_ms


# ============================================================================
# Step 6: raw mel spectrogram
# ============================================================================


def build_raw_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a conventional mel spectrogram and its original time axis.

    Returns
    -------
    mel_spec_db:
        Shape = (n_original_frames, n_mels)
    orig_frame_times_ms:
        Shape = (n_original_frames,)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).T.astype(np.float32)

    orig_frame_times_sec = librosa.frames_to_time(
        np.arange(mel_spec_db.shape[0]),
        sr=sample_rate,
        hop_length=hop_length,
        n_fft=n_fft,
    )
    orig_frame_times_ms = orig_frame_times_sec * 1000.0
    return mel_spec_db, orig_frame_times_ms


# ============================================================================
# Step 7: interpolation onto beat-aligned timeline
# ============================================================================


def interpolate_raw_mel_to_beat_aligned_timeline(
    mel_spec_db: np.ndarray,
    orig_frame_times_ms: np.ndarray,
    beat_aligned_frame_times_ms: np.ndarray,
) -> np.ndarray:
    """
    Interpolate the raw mel spectrogram onto the beat-aligned frame timeline.

    Edge behavior is intentionally conservative:
    - values before the first raw frame use the first raw frame
    - values after the last raw frame use the last raw frame
    """
    if mel_spec_db.ndim != 2:
        raise ValueError(f"Expected 2D mel spectrogram, got shape {mel_spec_db.shape}")

    n_target_frames = len(beat_aligned_frame_times_ms)
    n_mels = mel_spec_db.shape[1]
    aligned = np.empty((n_target_frames, n_mels), dtype=np.float32)

    for mel_bin in range(n_mels):
        aligned[:, mel_bin] = np.interp(
            beat_aligned_frame_times_ms,
            orig_frame_times_ms,
            mel_spec_db[:, mel_bin],
            left=float(mel_spec_db[0, mel_bin]),
            right=float(mel_spec_db[-1, mel_bin]),
        )

    if np.isnan(aligned).any():
        raise ValueError("NaN detected after interpolation")

    return aligned


# ============================================================================
# Step 8: segment aligned mel into 4-beat sequences
# ============================================================================


def segment_aligned_mel_into_4beat_sequences(
    aligned_mel_db: np.ndarray,
    total_sequences: int,
) -> np.ndarray:
    """
    Segment the aligned mel spectrogram into 4-beat windows.

    Output shape:
        (n_sequences, 192, n_mels)
    """
    if aligned_mel_db.shape[0] < total_sequences * FRAMES_PER_SEQUENCE:
        raise ValueError(
            "Aligned mel spectrogram is shorter than the expected number of full sequences"
        )

    sequences = []
    for seq_idx in range(total_sequences):
        start_frame = seq_idx * FRAMES_PER_SEQUENCE
        end_frame = start_frame + FRAMES_PER_SEQUENCE
        segment = aligned_mel_db[start_frame:end_frame]
        if segment.shape[0] != FRAMES_PER_SEQUENCE:
            raise ValueError(
                f"Unexpected sequence length at seq_idx={seq_idx}: {segment.shape}"
            )
        sequences.append(segment)

    if not sequences:
        raise ValueError("No full 4-beat sequences were created")

    return np.stack(sequences, axis=0).astype(np.float32)


# ============================================================================
# Step 9: per-sequence event tokens
# ============================================================================


def build_per_sequence_event_tokens(model_df: pd.DataFrame, total_sequences: int) -> List[Dict[str, Any]]:
    """
    Build local token sequences for each 4-beat segment.

    Token rule:
        time shift is measured in beat-aligned frames within the current sequence
        and represented as TS_<delta_frames>.
    """
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


# ============================================================================
# Wrapper for one chart
# ============================================================================


def process_one_chart_row(row: pd.Series, dataset_dir: Path) -> Dict[str, Any]:
    """
    Run the full 9-step pipeline for one chart row from the mapping table.

    The function writes per-chart outputs to disk and returns a compact summary.
    """
    folder_id = row["folder_id"]
    chart_base = row["chart_base"]
    chart_id = chart_uid(folder_id, chart_base)

    audio_path = Path(row["audio_path"])
    notes_path = Path(row["notes_path"])
    timing_path = Path(row["timing_path"])
    metadata_path = Path(row["metadata_path"])

    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    timing_info = get_timing_info(timing_path)
    audio_info = get_audio_info(audio_path)
    beat_grid_info, _ = compute_beat_grid_info(
        timing_info.offset_ms,
        timing_info.beat_duration_ms,
        audio_info["audio_duration_ms"],
    )
    events = load_note_events(notes_path)
    notes_info, _events_df, model_df = compute_notes_info(
        events,
        timing_info.offset_ms,
        timing_info.beat_duration_ms,
        beat_grid_info.total_frames,
    )

    if notes_info.model_events == 0:
        raise ValueError("No modeling events found after filtering event types")
    if notes_info.outside_event_count > 0:
        raise ValueError(f"Found {notes_info.outside_event_count} note events outside frame grid")
    if notes_info.collision_frame_count > 0:
        raise ValueError(f"Found {notes_info.collision_frame_count} collision frames")
    if beat_grid_info.total_sequences == 0:
        raise ValueError("No full 4-beat sequences available")

    frame_times_ms = build_beat_aligned_frame_timeline(
        timing_info.offset_ms,
        timing_info.beat_duration_ms,
        beat_grid_info.total_frames,
    )
    mel_spec_db, orig_frame_times_ms = build_raw_mel_spectrogram(
        audio_info["waveform"],
        audio_info["sample_rate"],
    )
    aligned_mel_db = interpolate_raw_mel_to_beat_aligned_timeline(
        mel_spec_db,
        orig_frame_times_ms,
        frame_times_ms,
    )
    audio_sequences = segment_aligned_mel_into_4beat_sequences(
        aligned_mel_db,
        beat_grid_info.total_sequences,
    )
    token_data = build_per_sequence_event_tokens(model_df, beat_grid_info.total_sequences)

    audio_npz_dir = dataset_dir / "audio_npz"
    token_json_dir = dataset_dir / "token_json"
    ensure_dir(audio_npz_dir)
    ensure_dir(token_json_dir)

    np.savez_compressed(
        audio_npz_dir / f"{chart_id}.npz",
        audio_sequences=audio_sequences,
    )
    safe_json_dump(token_data, token_json_dir / f"{chart_id}.json")

    sequence_metadata = []
    for seq in token_data:
        sequence_metadata.append(
            {
                "chart_id": chart_id,
                "folder_id": folder_id,
                "chart_base": chart_base,
                "seq_idx": seq["seq_idx"],
                "start_frame": seq["start_frame"],
                "end_frame": seq["end_frame"],
                "n_events": seq["n_events"],
                "n_tokens": seq["n_tokens"],
                "audio_npz_path": str(audio_npz_dir / f"{chart_id}.npz"),
                "token_json_path": str(token_json_dir / f"{chart_id}.json"),
            }
        )

    summary = {
        "chart_id": chart_id,
        "folder_id": folder_id,
        "chart_base": chart_base,
        "title": metadata.get("title", ""),
        "artist": metadata.get("artist", ""),
        "difficulty": metadata.get("difficulty", ""),
        "mode": metadata.get("mode", ""),
        "status": "ok",
        "error_message": "",
        "audio_path": str(audio_path),
        "notes_path": str(notes_path),
        "timing_path": str(timing_path),
        "metadata_path": str(metadata_path),
        "offset_ms": timing_info.offset_ms,
        "beat_duration_ms": timing_info.beat_duration_ms,
        "bpm": timing_info.bpm,
        "meter": timing_info.meter,
        "audio_duration_ms": audio_info["audio_duration_ms"],
        "total_beats": beat_grid_info.total_beats,
        "total_frames": beat_grid_info.total_frames,
        "total_sequences": beat_grid_info.total_sequences,
        "frame_overshoot_ms": beat_grid_info.frame_overshoot_ms,
        "total_events": notes_info.total_events,
        "model_events": notes_info.model_events,
        "outside_event_count": notes_info.outside_event_count,
        "collision_frame_count": notes_info.collision_frame_count,
        "unknown_event_types": "|".join(notes_info.unknown_event_types),
        "audio_sequences_shape": str(tuple(audio_sequences.shape)),
    }
    return {
        "summary": summary,
        "sequence_metadata": sequence_metadata,
    }


# ============================================================================
# Main pipeline
# ============================================================================


def run_pipeline(
    unpacked_root: Path = DEFAULT_UNPACKED_ROOT,
    index_dir: Path = DEFAULT_INDEX_DIR,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> None:
    """
    End-to-end execution:
    1. Build internal mapping table.
    2. Save mapping and mapping issues.
    3. Process every chart row in the mapping table.
    4. Save chart-level and sequence-level summaries.
    """
    ensure_dir(index_dir)
    ensure_dir(dataset_dir)

    logging.info("Building internal chart mapping table...")
    mapping_df, issues_df = build_chart_mapping_table(unpacked_root)

    mapping_csv = index_dir / "audio_chart_mapping_generated.csv"
    issues_csv = index_dir / "mapping_issues.csv"
    mapping_df.to_csv(mapping_csv, index=False, encoding="utf-8-sig")
    issues_df.to_csv(issues_csv, index=False, encoding="utf-8-sig")

    logging.info("Mapping rows generated: %d", len(mapping_df))
    logging.info("Mapping issues found: %d", len(issues_df))

    if mapping_df.empty:
        raise RuntimeError("No valid chart mapping rows were created")

    chart_summaries: List[Dict[str, Any]] = []
    sequence_metadata_rows: List[Dict[str, Any]] = []

    total_rows = len(mapping_df)
    for i, (_, row) in enumerate(mapping_df.iterrows(), start=1):
        chart_label = f"folder_id={row['folder_id']} | chart={row['chart_base']}"
        if i == 1 or i == total_rows or i % 20 == 0:
            logging.info("Processing %d / %d | %s", i, total_rows, chart_label)

        try:
            result = process_one_chart_row(row, dataset_dir)
            chart_summaries.append(result["summary"])
            sequence_metadata_rows.extend(result["sequence_metadata"])
        except Exception as exc:
            logging.error("Failed on %s | %s", chart_label, exc)
            chart_summaries.append(
                {
                    "chart_id": chart_uid(row["folder_id"], row["chart_base"]),
                    "folder_id": row["folder_id"],
                    "chart_base": row["chart_base"],
                    "status": "error",
                    "error_message": str(exc),
                    "audio_path": row["audio_path"],
                    "notes_path": row["notes_path"],
                    "timing_path": row["timing_path"],
                    "metadata_path": row["metadata_path"],
                }
            )
            continue

    chart_summary_df = pd.DataFrame(chart_summaries)
    sequence_metadata_df = pd.DataFrame(sequence_metadata_rows)

    chart_summary_csv = index_dir / "chart_build_summary.csv"
    sequence_metadata_csv = dataset_dir / "sequence_metadata.csv"
    chart_summary_df.to_csv(chart_summary_csv, index=False, encoding="utf-8-sig")
    sequence_metadata_df.to_csv(sequence_metadata_csv, index=False, encoding="utf-8-sig")

    ok_count = int((chart_summary_df["status"] == "ok").sum()) if not chart_summary_df.empty else 0
    error_count = int((chart_summary_df["status"] == "error").sum()) if not chart_summary_df.empty else 0

    logging.info("Pipeline finished")
    logging.info("Charts succeeded: %d", ok_count)
    logging.info("Charts failed: %d", error_count)
    logging.info("Mapping CSV saved to: %s", mapping_csv)
    logging.info("Chart summary saved to: %s", chart_summary_csv)
    logging.info("Sequence metadata saved to: %s", sequence_metadata_csv)


if __name__ == "__main__":
    setup_logging()
    run_pipeline()
