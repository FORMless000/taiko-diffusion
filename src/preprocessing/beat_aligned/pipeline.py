from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .audio import (
    build_beat_aligned_frame_timeline,
    build_raw_mel_spectrogram,
    compute_beat_grid_info,
    get_audio_info,
    get_timing_info,
    interpolate_raw_mel_to_beat_aligned_timeline,
    segment_aligned_mel_into_4beat_sequences,
)
from .common import (
    chart_uid,
    compute_chart_density_nps,
    ensure_dir,
    get_chart_beatmap_id,
    load_chart_metadata,
    safe_json_dump,
)
from .mapping import build_chart_mapping_table, filter_mapping_keep_max_notes_per_song
from .notes import build_per_sequence_event_tokens, compute_notes_info, load_note_events


def process_one_chart_row(
    row: pd.Series,
    dataset_dir: Path,
    reject_offgrid_notes: bool = True,
    offgrid_tolerance_ms: float = 5.0,
) -> Dict[str, Any]:
    folder_id = row["folder_id"]
    chart_base = row["chart_base"]
    chart_id = chart_uid(folder_id, chart_base)

    audio_path = Path(row["audio_path"])
    notes_path = Path(row["notes_path"])
    timing_path = Path(row["timing_path"])
    metadata_path = Path(row["metadata_path"])

    metadata = load_chart_metadata(metadata_path)

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
    if reject_offgrid_notes:
        tolerance_with_epsilon = offgrid_tolerance_ms + 1e-9
        offgrid_df = model_df[model_df["offgrid_abs_error_ms"] > tolerance_with_epsilon]
        if not offgrid_df.empty:
            max_deviation_ms = float(offgrid_df["offgrid_abs_error_ms"].max())
            raise ValueError(
                f"Found {len(offgrid_df)} off-grid model notes (> {offgrid_tolerance_ms:.3f} ms); "
                f"max deviation={max_deviation_ms:.3f} ms"
            )
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

    metadata_block = metadata.get("metadata", {})
    if not isinstance(metadata_block, dict):
        metadata_block = {}

    beatmap_id = get_chart_beatmap_id(metadata, folder_id=folder_id)
    density_nps = compute_chart_density_nps(model_df)
    difficulty_value = density_nps

    sequence_metadata = []
    for seq in token_data:
        sequence_metadata.append(
            {
                "chart_id": chart_id,
                "folder_id": folder_id,
                "chart_base": chart_base,
                "beatmap_id": beatmap_id,
                "bpm": timing_info.bpm,
                "density_nps": density_nps,
                "difficulty_value": difficulty_value,
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
        "beatmap_id": beatmap_id,
        "difficulty_value": difficulty_value,
        "density_nps": density_nps,
        "title": metadata.get("title", "") or metadata_block.get("Title", ""),
        "artist": metadata.get("artist", "") or metadata_block.get("Artist", ""),
        "difficulty": metadata.get("difficulty", "") or metadata_block.get("Version", ""),
        "mode": metadata.get("mode", "") or metadata.get("general", {}).get("Mode", ""),
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


def run_pipeline(
    unpacked_root: Path,
    index_dir: Path,
    dataset_dir: Path,
    reject_offgrid_notes: bool = True,
    offgrid_tolerance_ms: float = 5.0,
    keep_only_max_notes_per_song: bool = False,
) -> None:
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

    if keep_only_max_notes_per_song:
        mapping_df, dropped_count = filter_mapping_keep_max_notes_per_song(mapping_df)
        logging.info(
            "Song-level max-note filter enabled: kept %d rows, dropped %d rows",
            len(mapping_df),
            dropped_count,
        )
        if mapping_df.empty:
            raise RuntimeError("No charts remain after song-level max-note filtering")

    chart_summaries: List[Dict[str, Any]] = []
    sequence_metadata_rows: List[Dict[str, Any]] = []

    total_rows = len(mapping_df)
    for i, (_, row) in enumerate(mapping_df.iterrows(), start=1):
        chart_label = f"folder_id={row['folder_id']} | chart={row['chart_base']}"
        if i == 1 or i == total_rows or i % 20 == 0:
            logging.info("Processing %d / %d | %s", i, total_rows, chart_label)

        try:
            result = process_one_chart_row(
                row,
                dataset_dir,
                reject_offgrid_notes=reject_offgrid_notes,
                offgrid_tolerance_ms=offgrid_tolerance_ms,
            )
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
