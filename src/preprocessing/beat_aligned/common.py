from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_filename(text: str, max_length: int = 150) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    if not safe:
        safe = "chart"
    return safe[:max_length]


def chart_uid(folder_id: Any, chart_base: str) -> str:
    return f"{folder_id}_{sanitize_filename(chart_base)}"


def safe_json_dump(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def normalize_song_text(text: Any) -> str:
    text_str = str(text or "").strip().lower()
    return re.sub(r"\s+", " ", text_str)


def load_chart_metadata(metadata_path: Path) -> Dict[str, Any]:
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if isinstance(metadata, dict):
        return metadata
    return {}


def get_chart_beatmap_id(metadata: Dict[str, Any], folder_id: Any) -> int:
    metadata_block = metadata.get("metadata", {})
    if not isinstance(metadata_block, dict):
        metadata_block = {}

    raw = metadata_block.get("BeatmapID")
    if raw is not None and str(raw).strip():
        try:
            return max(1, int(str(raw).strip()))
        except ValueError:
            pass

    try:
        return max(1, int(str(folder_id).strip()))
    except ValueError:
        return 1


def compute_chart_density_nps(model_df: pd.DataFrame) -> float:
    if model_df.empty:
        return 0.0

    times = model_df["time"].astype(float).to_numpy()
    first_ms = float(np.min(times))
    last_ms = float(np.max(times))
    duration_sec = (last_ms - first_ms) / 1000.0
    if duration_sec <= 0:
        return 0.0

    return float(len(model_df) / duration_sec)


def get_song_group_key(metadata: Dict[str, Any]) -> str:
    metadata_block = metadata.get("metadata", {})
    if not isinstance(metadata_block, dict):
        metadata_block = {}

    beatmap_set_id = metadata_block.get("BeatmapSetID")
    beatmap_set_id_norm = str(beatmap_set_id).strip() if beatmap_set_id is not None else ""
    if beatmap_set_id_norm:
        return f"setid:{beatmap_set_id_norm}"

    artist = metadata_block.get("ArtistUnicode") or metadata_block.get("Artist") or metadata.get("artist", "")
    title = metadata_block.get("TitleUnicode") or metadata_block.get("Title") or metadata.get("title", "")
    return f"artist_title:{normalize_song_text(artist)}|{normalize_song_text(title)}"
