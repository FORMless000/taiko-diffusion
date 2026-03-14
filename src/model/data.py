from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence


def build_chart_manifest(audio_dir, token_dir):
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

        for seq_idx in range(n_seq):
            rows.append(
                {
                    "chart_id": chart_id,
                    "seq_idx": seq_idx,
                    "npz_path": npz_path,
                    "json_path": json_path,
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

    return {
        "chart_id": seq_row["chart_id"],
        "seq_idx": seq_idx,
        "audio": audio,
        "tokens": tokens,
        "n_tokens": len(tokens),
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
        }


def taiko_collate_fn(batch, pad_id=0):
    audio_list = [item["audio"] for item in batch]
    input_ids_list = [item["input_ids"] for item in batch]
    labels_list = [item["labels"] for item in batch]

    audio = torch.stack(audio_list, dim=0)
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=pad_id)
    decoder_attention_mask = (input_ids != pad_id).long()

    return {
        "audio": audio,
        "input_ids": input_ids,
        "labels": labels,
        "decoder_attention_mask": decoder_attention_mask,
    }
