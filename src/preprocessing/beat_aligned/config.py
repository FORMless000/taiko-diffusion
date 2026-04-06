from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_DATA_ROOT = Path("data")
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
