from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np

from .common import require_file
from .config import (
    BeatGridInfo,
    FRAMES_PER_BEAT,
    FRAMES_PER_SEQUENCE,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    TimingInfo,
)


def get_timing_info(timing_path: Path) -> TimingInfo:
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


def get_audio_info(audio_path: Path) -> Dict[str, Any]:
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


def compute_beat_grid_info(
    offset_ms: float,
    beat_duration_ms: float,
    audio_duration_ms: float,
) -> Tuple[BeatGridInfo, np.ndarray]:
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


def build_beat_aligned_frame_timeline(
    offset_ms: float,
    beat_duration_ms: float,
    total_frames: int,
) -> np.ndarray:
    frame_duration_ms = beat_duration_ms / FRAMES_PER_BEAT
    frame_times_ms = offset_ms + np.arange(total_frames, dtype=np.float64) * frame_duration_ms
    return frame_times_ms


def build_raw_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
) -> Tuple[np.ndarray, np.ndarray]:
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


def interpolate_raw_mel_to_beat_aligned_timeline(
    mel_spec_db: np.ndarray,
    orig_frame_times_ms: np.ndarray,
    beat_aligned_frame_times_ms: np.ndarray,
) -> np.ndarray:
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


def segment_aligned_mel_into_4beat_sequences(
    aligned_mel_db: np.ndarray,
    total_sequences: int,
) -> np.ndarray:
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
