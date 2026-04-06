from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .common import get_song_group_key, load_chart_metadata
from .notes import load_note_events
from .config import MODEL_EVENT_TYPES


def count_model_note_events_in_file(notes_path: Path) -> int:
    events = load_note_events(notes_path)
    return int(sum(1 for event in events if event.get("type") in MODEL_EVENT_TYPES))


def filter_mapping_keep_max_notes_per_song(mapping_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    rows = mapping_df.reset_index(drop=False).rename(columns={"index": "mapping_order"}).copy()
    rows["song_group_key"] = ""
    rows["model_note_count"] = -1

    for idx, row in rows.iterrows():
        metadata = load_chart_metadata(Path(row["metadata_path"]))
        rows.at[idx, "song_group_key"] = get_song_group_key(metadata)
        try:
            rows.at[idx, "model_note_count"] = count_model_note_events_in_file(Path(row["notes_path"]))
        except Exception as exc:
            logging.warning(
                "Failed to count model notes for folder_id=%s chart=%s: %s",
                row["folder_id"],
                row["chart_base"],
                exc,
            )

    kept_row_indices: List[int] = []
    for _, group in rows.groupby("song_group_key", sort=False):
        selected = group.sort_values(
            by=["model_note_count", "mapping_order"],
            ascending=[False, True],
        ).iloc[0]
        kept_row_indices.append(int(selected.name))

    kept_rows = rows.loc[kept_row_indices].sort_values("mapping_order").copy()
    filtered_df = kept_rows.drop(columns=["mapping_order", "song_group_key", "model_note_count"]).reset_index(drop=True)
    dropped_count = int(len(mapping_df) - len(filtered_df))
    return filtered_df, dropped_count


def build_chart_mapping_table(unpacked_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []

    if not unpacked_root.exists():
        raise FileNotFoundError(f"Unpacked root not found: {unpacked_root}")

    folder_paths = sorted([p for p in unpacked_root.iterdir() if p.is_dir()])
    logging.info("Scanning unpacked folders: %d", len(folder_paths))

    for folder_path in folder_paths:
        parsed_path = folder_path / "parsed"
        folder_id = folder_path.name

        audio_files = sorted(
            list(folder_path.glob("*.mp3"))
            + list(folder_path.glob("*.ogg"))
            + list(folder_path.glob("*.wav"))
            + list(folder_path.glob("*.flac"))
            + list(folder_path.glob("*.m4a"))
            + list(folder_path.glob("*.aac"))
            + list(folder_path.glob("*.opus"))
        )
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

        notes_map = {p.name[:-11]: p for p in notes_files}
        timing_map = {p.name[:-12]: p for p in timing_files}
        metadata_map = {p.name[:-14]: p for p in metadata_files}

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
