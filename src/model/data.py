from pathlib import Path
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


def _safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        if default is None:
            return None
        return float(default)


def _normalize_minmax(value, vmin, vmax):
    if vmax <= vmin:
        return 0.0
    clipped = max(vmin, min(vmax, float(value)))
    return (clipped - vmin) / (vmax - vmin)


def infer_difficulty_value(raw_difficulty):
    if raw_difficulty is None:
        return 5.0

    raw_text = str(raw_difficulty).strip().lower()
    numeric = _safe_float(raw_difficulty, None)
    if numeric is not None:
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
    if explicit_beatmap_id is not None and str(explicit_beatmap_id).strip() != "":
        return float(max(BEATMAP_ID_MIN, _safe_float(explicit_beatmap_id, BEATMAP_ID_MIN)))

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


def build_chart_manifest(audio_dir, token_dir, chart_metadata_csv=None):
    audio_dir = Path(audio_dir)
    token_dir = Path(token_dir)

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
                for c in ["chart_id", "difficulty", "difficulty_value", "bpm", "beatmap_id", "density_nps"]
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


def build_sequence_index(manifest_df, chart_id_list):
    chart_id_set = set(chart_id_list)
    split_df = manifest_df[manifest_df["chart_id"].isin(chart_id_set)].copy()

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
                    "difficulty_value": difficulty_value,
                    "difficulty": difficulty_raw,
                    "bpm": bpm,
                    "beatmap_id": beatmap_id,
                    "density_nps": density_nps,
                }
            )

    seq_index_df = pd.DataFrame(rows)
    return seq_index_df


def load_one_sample(seq_row):
    npz_path = seq_row["npz_path"]
    json_path = seq_row["json_path"]
    seq_idx = int(seq_row["seq_idx"])

    audio_arr = np.load(npz_path)["audio_sequences"]
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
    if explicit_density is not None and str(explicit_density).strip() != "":
        density_nps = _safe_float(explicit_density, 6.0)
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


