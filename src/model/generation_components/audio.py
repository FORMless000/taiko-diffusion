from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from src.preprocessing.beat_aligned import (
    build_beat_aligned_frame_timeline,
    build_raw_mel_spectrogram,
    compute_beat_grid_info,
    get_audio_info,
    interpolate_raw_mel_to_beat_aligned_timeline,
    segment_aligned_mel_into_4beat_sequences,
)


class CachedAudioPreprocessor:
    def __init__(self, audio_cache_size: int = 8):
        self.audio_cache_size = max(1, int(audio_cache_size))
        self._audio_cache = OrderedDict()

    def build_timing_info(self, audio_path, offset_ms, bpm, meter=4):
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if bpm <= 0:
            raise ValueError("bpm must be positive.")
        if meter <= 0:
            raise ValueError("meter must be positive.")

        beat_duration_ms = 60000.0 / bpm
        return {
            "audio_path": str(audio_path),
            "offset_ms": float(offset_ms),
            "bpm": float(bpm),
            "meter": int(meter),
            "beat_duration_ms": float(beat_duration_ms),
        }

    def cache_get_or_compute(self, cache_key, compute_fn):
        if cache_key in self._audio_cache:
            value = self._audio_cache.pop(cache_key)
            self._audio_cache[cache_key] = value
            return value

        value = compute_fn()
        self._audio_cache[cache_key] = value
        if len(self._audio_cache) > self.audio_cache_size:
            self._audio_cache.popitem(last=False)
        return value

    def preprocess_audio(self, audio_path, offset_ms, bpm, meter=4):
        timing_info = self.build_timing_info(
            audio_path=audio_path,
            offset_ms=offset_ms,
            bpm=bpm,
            meter=meter,
        )

        audio_key = (
            str(Path(timing_info["audio_path"]).resolve()),
            round(timing_info["offset_ms"], 3),
            round(timing_info["bpm"], 6),
            int(timing_info["meter"]),
        )

        def _compute_audio_sequences():
            audio_info = get_audio_info(Path(timing_info["audio_path"]))
            waveform = audio_info["waveform"]
            sample_rate = audio_info["sample_rate"]
            audio_duration_ms = audio_info["audio_duration_ms"]

            beat_grid_info, _ = compute_beat_grid_info(
                offset_ms=timing_info["offset_ms"],
                beat_duration_ms=timing_info["beat_duration_ms"],
                audio_duration_ms=audio_duration_ms,
            )

            beat_aligned_frame_times_ms = build_beat_aligned_frame_timeline(
                offset_ms=timing_info["offset_ms"],
                beat_duration_ms=timing_info["beat_duration_ms"],
                total_frames=beat_grid_info.total_frames,
            )

            mel_spec_db, orig_frame_times_ms = build_raw_mel_spectrogram(
                waveform=waveform,
                sample_rate=sample_rate,
            )

            aligned_mel_db = interpolate_raw_mel_to_beat_aligned_timeline(
                mel_spec_db=mel_spec_db,
                orig_frame_times_ms=orig_frame_times_ms,
                beat_aligned_frame_times_ms=beat_aligned_frame_times_ms,
            )

            return segment_aligned_mel_into_4beat_sequences(
                aligned_mel_db=aligned_mel_db,
                total_sequences=beat_grid_info.total_sequences,
            )

        return self.cache_get_or_compute(audio_key, _compute_audio_sequences)
