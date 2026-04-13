from pathlib import Path
from functools import partial
from collections import OrderedDict
import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence


DIFFICULTY_MIN = 0.5
DIFFICULTY_MAX = 10.0
DENSITY_NPS_MIN = 1.0
DENSITY_NPS_MAX = 20.0
BEATMAP_ID_MIN = 1.0
BEATMAP_ID_MAX = 5_000_000.0
CONTEXT_LABEL_IGNORE_INDEX = -100


def _is_nonempty(value):
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip() != ""


def _safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        if default is None:
            return None
        return float(default)


def _first_nonempty(values, default=""):
    for value in values:
        if _is_nonempty(value):
            return value
    return default


def _normalize_minmax(value, vmin, vmax):
    if vmax <= vmin:
        return 0.0
    clipped = max(vmin, min(vmax, float(value)))
    return (clipped - vmin) / (vmax - vmin)


def infer_difficulty_value(raw_difficulty):
    if not _is_nonempty(raw_difficulty):
        return 5.0

    raw_text = str(raw_difficulty).strip().lower()
    numeric = _safe_float(raw_difficulty, None)
    if numeric is not None and np.isfinite(numeric):
        return float(max(DIFFICULTY_MIN, min(DIFFICULTY_MAX, numeric)))

    # Fallback mapping from common taiko names when numeric OD is unavailable.
    if "kantan" in raw_text or "easy" in raw_text:
        return 2.0
    if "futsuu" in raw_text or "normal" in raw_text:
        return 4.0
    if "muzukashii" in raw_text or "hard" in raw_text:
        return 6.0
    if "inner oni" in raw_text:
        return 9.0
    if "ura oni" in raw_text:
        return 9.5
    if "oni" in raw_text or "insane" in raw_text:
        return 8.0
    return 5.0


def infer_beatmap_id_value(chart_id, explicit_beatmap_id=None):
    if _is_nonempty(explicit_beatmap_id):
        parsed = _safe_float(explicit_beatmap_id, None)
        if parsed is not None and np.isfinite(parsed):
            return float(max(BEATMAP_ID_MIN, parsed))

    chart_id_text = str(chart_id)
    m = re.match(r"^(\d+)", chart_id_text)
    if m:
        return float(max(BEATMAP_ID_MIN, _safe_float(m.group(1), BEATMAP_ID_MIN)))
    return BEATMAP_ID_MIN


def infer_density_nps(tokens, bpm):
    bpm_value = max(1.0, _safe_float(bpm, 120.0))
    event_count = sum(1 for tok in tokens if not str(tok).startswith("TS_"))

    # One training sequence is 4 beats.
    seq_duration_sec = 240.0 / bpm_value
    seq_duration_sec = max(1e-6, seq_duration_sec)

    density_nps = event_count / seq_duration_sec
    return float(max(0.0, min(DENSITY_NPS_MAX, density_nps)))


def preprocess_difficulty_value(difficulty_value):
    return _normalize_minmax(difficulty_value, DIFFICULTY_MIN, DIFFICULTY_MAX)


def preprocess_density_nps(density_nps):
    return _normalize_minmax(density_nps, DENSITY_NPS_MIN, DENSITY_NPS_MAX)


def preprocess_beatmap_id(beatmap_id):
    raw = max(BEATMAP_ID_MIN, _safe_float(beatmap_id, BEATMAP_ID_MIN))
    log_min = np.log1p(BEATMAP_ID_MIN)
    log_max = np.log1p(BEATMAP_ID_MAX)
    return _normalize_minmax(np.log1p(raw), log_min, log_max)


def build_chart_manifest(
    audio_dir,
    token_dir,
    chart_metadata_csv=None,
    sequence_metadata_csv=None,
    prefer_metadata=True,
):
    """
    Build chart-level manifest.

    Fast path:
      Build from chart/sequence metadata CSVs without opening every npz/json file.
    Fallback:
      Scan artifact directories and inspect files.
    """
    audio_dir = Path(audio_dir)
    token_dir = Path(token_dir)
    chart_metadata_path = Path(chart_metadata_csv) if chart_metadata_csv is not None else None
    sequence_metadata_path = Path(sequence_metadata_csv) if sequence_metadata_csv is not None else None

    if (
        prefer_metadata
        and chart_metadata_path is not None
        and sequence_metadata_path is not None
        and chart_metadata_path.exists()
        and sequence_metadata_path.exists()
    ):
        sequence_usecols = [
            "chart_id",
            "seq_idx",
            "audio_npz_path",
            "token_json_path",
            "shared_audio_npz_path",
            "offset_ms",
            "beat_duration_ms",
            "total_frames",
            "total_sequences",
        ]
        with open(sequence_metadata_path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline().strip().split(",")
        available_seq_cols = [c for c in sequence_usecols if c in header]

        if {"chart_id", "seq_idx"}.issubset(set(available_seq_cols)):
            seq_df = pd.read_csv(sequence_metadata_path, usecols=available_seq_cols)
            seq_df["chart_id"] = seq_df["chart_id"].astype(str)
            grouped = seq_df.groupby("chart_id", sort=False)
            seq_counts = grouped.size().rename("n_sequences_token")

            seq_manifest = seq_counts.to_frame().reset_index()
            if "audio_npz_path" in seq_df.columns:
                seq_manifest["npz_path"] = grouped["audio_npz_path"].agg(lambda s: _first_nonempty(s.tolist(), ""))
            else:
                seq_manifest["npz_path"] = ""
            if "token_json_path" in seq_df.columns:
                seq_manifest["json_path"] = grouped["token_json_path"].agg(lambda s: _first_nonempty(s.tolist(), ""))
            else:
                seq_manifest["json_path"] = ""
            if "shared_audio_npz_path" in seq_df.columns:
                seq_manifest["shared_audio_npz_path"] = grouped["shared_audio_npz_path"].agg(
                    lambda s: _first_nonempty(s.tolist(), "")
                )
            else:
                seq_manifest["shared_audio_npz_path"] = ""

            for col in ["offset_ms", "beat_duration_ms", "total_frames", "total_sequences"]:
                if col in seq_df.columns:
                    seq_manifest[col] = grouped[col].agg(lambda s: _first_nonempty(s.tolist(), np.nan))

            seq_manifest["n_sequences_audio"] = seq_manifest["n_sequences_token"].astype(int)
            seq_manifest["npz_path"] = seq_manifest["npz_path"].replace("", np.nan)
            seq_manifest["json_path"] = seq_manifest["json_path"].replace("", np.nan)
            seq_manifest["npz_path"] = seq_manifest.apply(
                lambda row: row["npz_path"]
                if _is_nonempty(row["npz_path"])
                else str(audio_dir / f"{row['chart_id']}.npz"),
                axis=1,
            )
            seq_manifest["json_path"] = seq_manifest.apply(
                lambda row: row["json_path"]
                if _is_nonempty(row["json_path"])
                else str(token_dir / f"{row['chart_id']}.json"),
                axis=1,
            )

            chart_meta_df = pd.read_csv(chart_metadata_path)
            if "chart_id" in chart_meta_df.columns:
                chart_meta_df["chart_id"] = chart_meta_df["chart_id"].astype(str)
                keep_cols = [
                    c
                    for c in [
                        "chart_id",
                        "difficulty",
                        "difficulty_value",
                        "bpm",
                        "beatmap_id",
                        "density_nps",
                        "shared_audio_npz_path",
                        "shared_audio_id",
                        "offset_ms",
                        "beat_duration_ms",
                        "total_frames",
                        "total_sequences",
                    ]
                    if c in chart_meta_df.columns
                ]
                merged = seq_manifest.merge(chart_meta_df[keep_cols], on="chart_id", how="left", suffixes=("", "_meta"))
                for col in ["shared_audio_npz_path", "offset_ms", "beat_duration_ms", "total_frames", "total_sequences"]:
                    col_meta = f"{col}_meta"
                    if col_meta in merged.columns:
                        merged[col] = merged[col].where(merged[col].notna(), merged[col_meta])
                        merged = merged.drop(columns=[col_meta])
                return merged.reset_index(drop=True)

    # Fallback path for older datasets without usable metadata files.
    audio_files = {p.stem: p for p in audio_dir.glob("*.npz")}
    token_files = {p.stem: p for p in token_dir.glob("*.json")}
    common_ids = sorted(set(audio_files) & set(token_files))

    rows = []
    for chart_id in common_ids:
        npz_path = audio_files[chart_id]
        json_path = token_files[chart_id]

        audio_arr = np.load(npz_path)["audio_sequences"]
        with open(json_path, "r", encoding="utf-8") as f:
            token_data = json.load(f)

        rows.append(
            {
                "chart_id": chart_id,
                "npz_path": str(npz_path),
                "json_path": str(json_path),
                "n_sequences_audio": int(audio_arr.shape[0]),
                "n_sequences_token": int(len(token_data)),
            }
        )

    manifest_df = pd.DataFrame(rows)

    if chart_metadata_csv is not None:
        meta_df = pd.read_csv(chart_metadata_csv)
        if "chart_id" in meta_df.columns:
            keep_cols = [
                c
                for c in [
                    "chart_id",
                    "difficulty",
                    "difficulty_value",
                    "bpm",
                    "beatmap_id",
                    "density_nps",
                    "shared_audio_npz_path",
                    "shared_audio_id",
                    "offset_ms",
                    "beat_duration_ms",
                    "total_frames",
                    "total_sequences",
                ]
                if c in meta_df.columns
            ]
            manifest_df = manifest_df.merge(meta_df[keep_cols], on="chart_id", how="left")

    return manifest_df


def split_chart_manifest(manifest_df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8

    chart_ids = manifest_df["chart_id"].tolist()

    train_ids, temp_ids = train_test_split(
        chart_ids,
        test_size=(1 - train_ratio),
        random_state=random_state,
        shuffle=True,
    )

    val_size_within_temp = val_ratio / (val_ratio + test_ratio)

    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=val_size_within_temp,
        random_state=random_state,
        shuffle=True,
    )

    return train_ids, val_ids, test_ids


def build_sequence_index(
    manifest_df,
    chart_id_list,
    sequence_metadata_csv=None,
    prefer_metadata=True,
):
    chart_id_set = {str(cid) for cid in chart_id_list}

    if prefer_metadata and sequence_metadata_csv is not None and Path(sequence_metadata_csv).exists():
        usecols_candidates = [
            "chart_id",
            "seq_idx",
            "audio_npz_path",
            "token_json_path",
            "shared_audio_npz_path",
            "offset_ms",
            "beat_duration_ms",
            "total_frames",
            "total_sequences",
            "difficulty",
            "difficulty_value",
            "bpm",
            "beatmap_id",
            "density_nps",
        ]
        with open(sequence_metadata_csv, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline().strip().split(",")
        usecols = [c for c in usecols_candidates if c in header]
        if {"chart_id", "seq_idx"}.issubset(set(usecols)):
            seq_df = pd.read_csv(sequence_metadata_csv, usecols=usecols)
            seq_df["chart_id"] = seq_df["chart_id"].astype(str)
            seq_df = seq_df[seq_df["chart_id"].isin(chart_id_set)].copy()
            if not seq_df.empty:
                seq_df["seq_idx"] = seq_df["seq_idx"].astype(int)
                seq_df = seq_df.sort_values(["chart_id", "seq_idx"]).reset_index(drop=True)
                if "audio_npz_path" in seq_df.columns:
                    seq_df = seq_df.rename(columns={"audio_npz_path": "npz_path"})
                else:
                    seq_df["npz_path"] = ""
                if "token_json_path" in seq_df.columns:
                    seq_df = seq_df.rename(columns={"token_json_path": "json_path"})
                else:
                    seq_df["json_path"] = ""

                manifest_cols = [
                    c
                    for c in [
                        "chart_id",
                        "npz_path",
                        "json_path",
                        "shared_audio_npz_path",
                        "offset_ms",
                        "beat_duration_ms",
                        "total_frames",
                        "total_sequences",
                        "difficulty",
                        "difficulty_value",
                        "bpm",
                        "beatmap_id",
                        "density_nps",
                    ]
                    if c in manifest_df.columns
                ]
                manifest_subset = manifest_df[manifest_cols].drop_duplicates("chart_id").copy()
                manifest_subset["chart_id"] = manifest_subset["chart_id"].astype(str)
                merged = seq_df.merge(manifest_subset, on="chart_id", how="left", suffixes=("", "_manifest"))

                fill_cols = [
                    "npz_path",
                    "json_path",
                    "shared_audio_npz_path",
                    "offset_ms",
                    "beat_duration_ms",
                    "total_frames",
                    "total_sequences",
                    "difficulty",
                    "difficulty_value",
                    "bpm",
                    "beatmap_id",
                    "density_nps",
                ]
                for col in fill_cols:
                    col_manifest = f"{col}_manifest"
                    if col_manifest in merged.columns:
                        merged[col] = merged[col].where(merged[col].notna() & (merged[col].astype(str) != ""), merged[col_manifest])
                        merged = merged.drop(columns=[col_manifest])

                # Keep the exact schema expected by datasets.
                expected_cols = [
                    "chart_id",
                    "seq_idx",
                    "npz_path",
                    "json_path",
                    "shared_audio_npz_path",
                    "offset_ms",
                    "beat_duration_ms",
                    "total_frames",
                    "total_sequences",
                    "difficulty_value",
                    "difficulty",
                    "bpm",
                    "beatmap_id",
                    "density_nps",
                ]
                for col in expected_cols:
                    if col not in merged.columns:
                        merged[col] = ""
                return merged[expected_cols].reset_index(drop=True)

    split_df = manifest_df[manifest_df["chart_id"].astype(str).isin(chart_id_set)].copy()
    rows = []
    for _, row in split_df.iterrows():
        chart_id = row["chart_id"]
        npz_path = row["npz_path"]
        json_path = row["json_path"]
        n_seq = int(row["n_sequences_audio"])

        difficulty_raw = row["difficulty"] if "difficulty" in row.index else ""
        difficulty_value = row["difficulty_value"] if "difficulty_value" in row.index else difficulty_raw
        bpm = row["bpm"] if "bpm" in row.index else 120.0
        beatmap_id = row["beatmap_id"] if "beatmap_id" in row.index else ""
        density_nps = row["density_nps"] if "density_nps" in row.index else ""

        for seq_idx in range(n_seq):
            rows.append(
                {
                    "chart_id": chart_id,
                    "seq_idx": seq_idx,
                    "npz_path": npz_path,
                    "json_path": json_path,
                    "shared_audio_npz_path": row["shared_audio_npz_path"] if "shared_audio_npz_path" in row.index else "",
                    "offset_ms": row["offset_ms"] if "offset_ms" in row.index else "",
                    "beat_duration_ms": row["beat_duration_ms"] if "beat_duration_ms" in row.index else "",
                    "total_frames": row["total_frames"] if "total_frames" in row.index else "",
                    "total_sequences": row["total_sequences"] if "total_sequences" in row.index else "",
                    "difficulty_value": difficulty_value,
                    "difficulty": difficulty_raw,
                    "bpm": bpm,
                    "beatmap_id": beatmap_id,
                    "density_nps": density_nps,
                }
            )
    return pd.DataFrame(rows)


def load_audio_sequences_for_row(seq_row):
    npz_path = str(seq_row.get("npz_path", "")).strip()
    if npz_path:
        return np.load(npz_path)["audio_sequences"]

    shared_audio_npz_path = str(seq_row.get("shared_audio_npz_path", "")).strip()
    if not shared_audio_npz_path:
        raise ValueError("Missing both `npz_path` and `shared_audio_npz_path` for chart audio.")

    offset_ms = _safe_float(seq_row.get("offset_ms", None), None)
    beat_duration_ms = _safe_float(seq_row.get("beat_duration_ms", None), None)
    total_frames_raw = _safe_float(seq_row.get("total_frames", None), None)
    total_sequences_raw = _safe_float(seq_row.get("total_sequences", None), None)
    if offset_ms is None or beat_duration_ms is None or total_frames_raw is None or total_sequences_raw is None:
        raise ValueError("Shared audio rows must provide offset/beat_duration/total_frames/total_sequences.")

    total_frames = int(total_frames_raw)
    total_sequences = int(total_sequences_raw)
    if total_frames <= 0 or total_sequences <= 0:
        raise ValueError("Shared audio rows must have positive total_frames and total_sequences.")

    from src.preprocessing.beat_aligned_dataset import (
        build_beat_aligned_frame_timeline,
        interpolate_raw_mel_to_beat_aligned_timeline,
        segment_aligned_mel_into_4beat_sequences,
    )

    shared_payload = np.load(shared_audio_npz_path)
    mel_spec_db = shared_payload["mel_spec_db"]
    orig_frame_times_ms = shared_payload["orig_frame_times_ms"]
    frame_times_ms = build_beat_aligned_frame_timeline(
        offset_ms=float(offset_ms),
        beat_duration_ms=float(beat_duration_ms),
        total_frames=total_frames,
    )
    aligned_mel_db = interpolate_raw_mel_to_beat_aligned_timeline(
        mel_spec_db=mel_spec_db,
        orig_frame_times_ms=orig_frame_times_ms,
        beat_aligned_frame_times_ms=frame_times_ms,
    )
    return segment_aligned_mel_into_4beat_sequences(
        aligned_mel_db=aligned_mel_db,
        total_sequences=total_sequences,
    )


def load_one_sample(seq_row):
    npz_path = seq_row["npz_path"]
    json_path = seq_row["json_path"]
    seq_idx = int(seq_row["seq_idx"])

    audio_arr = load_audio_sequences_for_row(seq_row)
    audio = audio_arr[seq_idx]

    with open(json_path, "r", encoding="utf-8") as f:
        token_data = json.load(f)

    item = token_data[seq_idx]
    tokens = item["tokens"]

    raw_diff_value = seq_row.get("difficulty_value", seq_row.get("difficulty", ""))
    difficulty_value = infer_difficulty_value(raw_diff_value)

    beatmap_id_raw = infer_beatmap_id_value(
        chart_id=seq_row.get("chart_id", ""),
        explicit_beatmap_id=seq_row.get("beatmap_id", None),
    )

    explicit_density = seq_row.get("density_nps", "")
    if _is_nonempty(explicit_density):
        density_nps = _safe_float(explicit_density, 6.0)
        if not np.isfinite(density_nps):
            density_nps = 6.0
    else:
        density_nps = infer_density_nps(tokens=tokens, bpm=seq_row.get("bpm", 120.0))

    return {
        "chart_id": seq_row["chart_id"],
        "seq_idx": seq_idx,
        "audio": audio,
        "tokens": tokens,
        "n_tokens": len(tokens),
        "difficulty_value": difficulty_value,
        "density_nps": density_nps,
        "beatmap_id": beatmap_id_raw,
        "difficulty_value_norm": preprocess_difficulty_value(difficulty_value),
        "density_value_norm": preprocess_density_nps(density_nps),
        "beatmap_id_value_norm": preprocess_beatmap_id(beatmap_id_raw),
    }


def build_vocab_from_all_splits(train_seq_index, val_seq_index, test_seq_index):
    token_set = set()

    all_json_paths = set(
        train_seq_index["json_path"].tolist()
        + val_seq_index["json_path"].tolist()
        + test_seq_index["json_path"].tolist()
    )

    for json_path in all_json_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            token_data = json.load(f)

        for item in token_data:
            for tok in item["tokens"]:
                token_set.add(tok)

    special_tokens = ["PAD", "BOS", "EOS"]
    event_tokens = sorted([t for t in token_set if not t.startswith("TS_")])
    ts_tokens = sorted(
        [t for t in token_set if t.startswith("TS_")],
        key=lambda x: int(x.split("_")[1]),
    )

    vocab_list = special_tokens + event_tokens + ts_tokens
    token_to_id = {tok: i for i, tok in enumerate(vocab_list)}
    id_to_token = {i: tok for tok, i in token_to_id.items()}
    return vocab_list, token_to_id, id_to_token


def encode_tokens(tokens, token_to_id):
    bos = token_to_id["BOS"]
    eos = token_to_id["EOS"]

    token_ids = [token_to_id[t] for t in tokens]
    input_ids = [bos] + token_ids
    labels = token_ids + [eos]
    return input_ids, labels


def is_context_architecture_name(name):
    return str(name).strip() == "taiko_context_transformer"


class TaikoDataset(Dataset):
    def __init__(self, seq_index_df, token_to_id):
        self.seq_index_df = seq_index_df.reset_index(drop=True)
        self.token_to_id = token_to_id

    def __len__(self):
        return len(self.seq_index_df)

    def __getitem__(self, idx):
        row = self.seq_index_df.iloc[idx]
        sample = load_one_sample(row)
        input_ids, labels = encode_tokens(sample["tokens"], self.token_to_id)

        return {
            "audio": torch.tensor(sample["audio"], dtype=torch.float32),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "difficulty_value": torch.tensor(sample["difficulty_value_norm"], dtype=torch.float32),
            "density_value": torch.tensor(sample["density_value_norm"], dtype=torch.float32),
            "beatmap_id_value": torch.tensor(sample["beatmap_id_value_norm"], dtype=torch.float32),
        }


class TaikoContextDataset(Dataset):
    def __init__(
        self,
        seq_index_df,
        token_to_id,
        history_max_tokens=1024,
        retrieval_top_k=2,
        retrieval_max_tokens_per_window=64,
        retrieval_exclude_last_n_windows=2,
        use_motif_retrieval=True,
        max_cached_charts=4,
    ):
        self.seq_index_df = seq_index_df.reset_index(drop=True)
        self.token_to_id = token_to_id
        self.history_max_tokens = max(1, int(history_max_tokens))
        self.retrieval_top_k = max(0, int(retrieval_top_k))
        self.retrieval_max_tokens_per_window = max(1, int(retrieval_max_tokens_per_window))
        self.retrieval_exclude_last_n_windows = max(0, int(retrieval_exclude_last_n_windows))
        self.use_motif_retrieval = bool(use_motif_retrieval)
        self.max_cached_charts = max(1, int(max_cached_charts))
        self.bos_id = int(token_to_id["BOS"])
        self.eos_id = int(token_to_id["EOS"])
        self.ts_token_ids = {int(idx) for tok, idx in token_to_id.items() if str(tok).startswith("TS_")}

        self._chart_rows = {}
        for _, row in self.seq_index_df.iterrows():
            row_dict = row.to_dict()
            chart_id = row_dict["chart_id"]
            self._chart_rows.setdefault(chart_id, []).append(row_dict)

        for chart_id in self._chart_rows:
            self._chart_rows[chart_id].sort(key=lambda row: int(row["seq_idx"]))

        self._chart_cache = OrderedDict()

    def __len__(self):
        return len(self.seq_index_df)

    def _build_sequence_assets(self, seq_idx, audio_arr, token_data):
        tokens = token_data[seq_idx]["tokens"]
        token_ids = [self.token_to_id[t] for t in tokens]
        audio_key = np.asarray(audio_arr[seq_idx], dtype=np.float32).mean(axis=0)
        norm = float(np.linalg.norm(audio_key))
        if norm > 0.0:
            audio_key = audio_key / norm

        return {
            "seq_idx": seq_idx,
            "token_ids": token_ids,
            "audio_key": audio_key.astype(np.float32, copy=False),
        }

    def _conditioning_from_row(self, row, token_ids):
        raw_diff_value = row.get("difficulty_value", row.get("difficulty", ""))
        difficulty_value = infer_difficulty_value(raw_diff_value)
        beatmap_id_raw = infer_beatmap_id_value(
            chart_id=row.get("chart_id", ""),
            explicit_beatmap_id=row.get("beatmap_id", None),
        )
        explicit_density = row.get("density_nps", "")
        if _is_nonempty(explicit_density):
            density_nps = _safe_float(explicit_density, 6.0)
            if not np.isfinite(density_nps):
                density_nps = 6.0
        else:
            # Estimate from token ids when explicit value is unavailable.
            n_events = sum(1 for tid in token_ids if int(tid) not in self.ts_token_ids)
            bpm_value = max(1.0, _safe_float(row.get("bpm", 120.0), 120.0))
            seq_duration_sec = max(1e-6, 240.0 / bpm_value)
            density_nps = float(max(0.0, min(DENSITY_NPS_MAX, n_events / seq_duration_sec)))
        return {
            "difficulty_value_norm": preprocess_difficulty_value(difficulty_value),
            "density_value_norm": preprocess_density_nps(density_nps),
            "beatmap_id_value_norm": preprocess_beatmap_id(beatmap_id_raw),
        }

    def _cache_put(self, chart_id, payload):
        if chart_id in self._chart_cache:
            self._chart_cache.pop(chart_id, None)
        self._chart_cache[chart_id] = payload
        while len(self._chart_cache) > self.max_cached_charts:
            _, evicted_payload = self._chart_cache.popitem(last=False)
            del evicted_payload

    def _load_chart_samples(self, chart_id):
        if chart_id in self._chart_cache:
            payload = self._chart_cache.pop(chart_id)
            self._chart_cache[chart_id] = payload
            return payload

        rows = self._chart_rows[chart_id]
        audio_arr = load_audio_sequences_for_row(rows[0])
        with open(rows[0]["json_path"], "r", encoding="utf-8") as f:
            token_data = json.load(f)

        ordered = []
        by_seq = {}
        for row in rows:
            seq_idx = int(row["seq_idx"])
            sample = self._build_sequence_assets(seq_idx, audio_arr, token_data)
            ordered.append(sample)
            by_seq[seq_idx] = sample

        payload = {"ordered": ordered, "by_seq": by_seq, "audio_arr": audio_arr}
        self._cache_put(chart_id, payload)
        return payload

    def _serialize_window_token_ids(self, token_ids, limit=None):
        if limit is None:
            return list(token_ids) + [self.eos_id]

        limit = max(1, int(limit))
        payload = list(token_ids[: max(0, limit - 1)])
        payload.append(self.eos_id)
        return payload[:limit]

    def _build_recent_history_ids(self, ordered_samples, current_seq_idx):
        history_ids = []
        for sample in ordered_samples:
            if int(sample["seq_idx"]) >= int(current_seq_idx):
                break
            history_ids.extend(self._serialize_window_token_ids(sample["token_ids"]))

        if len(history_ids) > self.history_max_tokens:
            history_ids = history_ids[-self.history_max_tokens :]
        return history_ids

    def _build_retrieved_ids(self, ordered_samples, current_sample):
        if not self.use_motif_retrieval or self.retrieval_top_k <= 0:
            return []

        current_seq_idx = int(current_sample["seq_idx"])
        cutoff_seq_idx = current_seq_idx - self.retrieval_exclude_last_n_windows
        candidates = []

        for sample in ordered_samples:
            sample_seq_idx = int(sample["seq_idx"])
            if sample_seq_idx >= current_seq_idx:
                break
            if sample_seq_idx >= cutoff_seq_idx:
                continue

            similarity = float(np.dot(current_sample["audio_key"], sample["audio_key"]))
            candidates.append((similarity, sample_seq_idx, sample))

        if not candidates:
            return []

        candidates.sort(key=lambda item: (-item[0], item[1]))
        retrieved_ids = []
        for _, _, sample in candidates[: self.retrieval_top_k]:
            retrieved_ids.extend(
                self._serialize_window_token_ids(
                    sample["token_ids"],
                    limit=self.retrieval_max_tokens_per_window,
                )
            )
        return retrieved_ids

    def __getitem__(self, idx):
        row = self.seq_index_df.iloc[idx].to_dict()
        chart_payload = self._load_chart_samples(row["chart_id"])
        current_seq_idx = int(row["seq_idx"])
        current_sample = chart_payload["by_seq"][current_seq_idx]
        current_token_ids = list(current_sample["token_ids"])
        current_input_ids = [self.bos_id] + current_token_ids
        current_labels = current_token_ids + [self.eos_id]
        history_ids = self._build_recent_history_ids(chart_payload["ordered"], current_sample["seq_idx"])
        retrieved_ids = self._build_retrieved_ids(chart_payload["ordered"], current_sample)
        conditioning = self._conditioning_from_row(row, current_token_ids)

        input_ids = history_ids + retrieved_ids + current_input_ids
        labels = (
            [CONTEXT_LABEL_IGNORE_INDEX] * (len(history_ids) + len(retrieved_ids))
            + current_labels
        )
        segment_ids = (
            [0] * len(history_ids)
            + [1] * len(retrieved_ids)
            + [2] * len(current_input_ids)
        )

        return {
            "audio": torch.tensor(chart_payload["audio_arr"][current_seq_idx], dtype=torch.float32),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
            "difficulty_value": torch.tensor(conditioning["difficulty_value_norm"], dtype=torch.float32),
            "density_value": torch.tensor(conditioning["density_value_norm"], dtype=torch.float32),
            "beatmap_id_value": torch.tensor(conditioning["beatmap_id_value_norm"], dtype=torch.float32),
        }


def taiko_collate_fn(batch, pad_id=0):
    audio_list = [item["audio"] for item in batch]
    input_ids_list = [item["input_ids"] for item in batch]
    labels_list = [item["labels"] for item in batch]
    difficulty_value_list = [item["difficulty_value"] for item in batch]
    density_value_list = [item["density_value"] for item in batch]
    beatmap_id_value_list = [item["beatmap_id_value"] for item in batch]

    audio = torch.stack(audio_list, dim=0)
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=pad_id)
    decoder_attention_mask = (input_ids != pad_id).long()

    difficulty_values = torch.stack(difficulty_value_list, dim=0)
    density_values = torch.stack(density_value_list, dim=0)
    beatmap_id_values = torch.stack(beatmap_id_value_list, dim=0)

    return {
        "audio": audio,
        "input_ids": input_ids,
        "labels": labels,
        "decoder_attention_mask": decoder_attention_mask,
        "difficulty_values": difficulty_values,
        "density_values": density_values,
        "beatmap_id_values": beatmap_id_values,
    }


def taiko_context_collate_fn(batch, pad_id=0, ignore_index=CONTEXT_LABEL_IGNORE_INDEX):
    audio_list = [item["audio"] for item in batch]
    input_ids_list = [item["input_ids"] for item in batch]
    labels_list = [item["labels"] for item in batch]
    segment_ids_list = [item["segment_ids"] for item in batch]
    difficulty_value_list = [item["difficulty_value"] for item in batch]
    density_value_list = [item["density_value"] for item in batch]
    beatmap_id_value_list = [item["beatmap_id_value"] for item in batch]

    audio = torch.stack(audio_list, dim=0)
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=ignore_index)
    segment_ids = pad_sequence(segment_ids_list, batch_first=True, padding_value=0)
    decoder_attention_mask = (input_ids != pad_id).long()

    difficulty_values = torch.stack(difficulty_value_list, dim=0)
    density_values = torch.stack(density_value_list, dim=0)
    beatmap_id_values = torch.stack(beatmap_id_value_list, dim=0)

    return {
        "audio": audio,
        "input_ids": input_ids,
        "labels": labels,
        "segment_ids": segment_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "difficulty_values": difficulty_values,
        "density_values": density_values,
        "beatmap_id_values": beatmap_id_values,
    }


def build_dataset_for_spec(seq_index_df, token_to_id, architecture_spec):
    pad_id = int(token_to_id["PAD"])

    if is_context_architecture_name(getattr(architecture_spec, "name", architecture_spec)):
        dataset_context_kwargs = {}
        if hasattr(architecture_spec, "dataset_context_kwargs"):
            dataset_context_kwargs = architecture_spec.dataset_context_kwargs()
        dataset = TaikoContextDataset(
            seq_index_df=seq_index_df,
            token_to_id=token_to_id,
            history_max_tokens=getattr(architecture_spec, "history_max_tokens", 1024),
            retrieval_top_k=getattr(architecture_spec, "retrieval_top_k", 2),
            retrieval_max_tokens_per_window=getattr(architecture_spec, "retrieval_max_tokens_per_window", 64),
            retrieval_exclude_last_n_windows=getattr(architecture_spec, "retrieval_exclude_last_n_windows", 2),
            use_motif_retrieval=getattr(architecture_spec, "use_motif_retrieval", True),
            max_cached_charts=dataset_context_kwargs.get(
                "max_cached_charts",
                getattr(architecture_spec, "max_cached_charts", 4),
            ),
        )
        collate_fn = partial(
            taiko_context_collate_fn,
            pad_id=pad_id,
            ignore_index=CONTEXT_LABEL_IGNORE_INDEX,
        )
        return dataset, collate_fn, CONTEXT_LABEL_IGNORE_INDEX

    dataset = TaikoDataset(seq_index_df, token_to_id)
    collate_fn = partial(taiko_collate_fn, pad_id=pad_id)
    return dataset, collate_fn, pad_id
