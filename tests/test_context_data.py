import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if torch is not None:
    from src.model.data import CONTEXT_LABEL_IGNORE_INDEX, TaikoContextDataset, build_chart_manifest, build_sequence_index


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestContextDataset(unittest.TestCase):
    def _write_chart(self, root: Path):
        audio_dir = root / "audio_npz"
        token_dir = root / "token_json"
        audio_dir.mkdir(parents=True, exist_ok=True)
        token_dir.mkdir(parents=True, exist_ok=True)

        chart_id = "2034220_context_chart"
        audio_sequences = np.zeros((4, 192, 128), dtype=np.float32)
        audio_sequences[0, :, 0] = 1.0
        audio_sequences[1, :, 1] = 1.0
        audio_sequences[2, :, 2] = 1.0
        audio_sequences[3, :, 0] = 1.0
        np.savez_compressed(audio_dir / f"{chart_id}.npz", audio_sequences=audio_sequences)

        token_payload = [
            {"seq_idx": 0, "tokens": ["DON", "TS_2", "KAT"]},
            {"seq_idx": 1, "tokens": ["KAT", "TS_2", "DON"]},
            {"seq_idx": 2, "tokens": ["BIGDON", "TS_2", "BIGKAT"]},
            {"seq_idx": 3, "tokens": ["DON", "TS_2", "KAT"]},
        ]
        (token_dir / f"{chart_id}.json").write_text(__import__("json").dumps(token_payload), encoding="utf-8")

        meta_csv = root / "chart_meta.csv"
        pd.DataFrame(
            [
                {
                    "chart_id": chart_id,
                    "difficulty_value": 6.5,
                    "bpm": 180.0,
                    "beatmap_id": 2034220,
                }
            ]
        ).to_csv(meta_csv, index=False)

        manifest = build_chart_manifest(audio_dir, token_dir, chart_metadata_csv=meta_csv)
        seq_index = build_sequence_index(manifest, [chart_id])
        return chart_id, seq_index

    def test_context_dataset_masks_prefix_and_adds_segments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, seq_index = self._write_chart(Path(tmpdir))
            token_to_id = {
                "PAD": 0,
                "BOS": 1,
                "EOS": 2,
                "DON": 3,
                "KAT": 4,
                "BIGDON": 5,
                "BIGKAT": 6,
                "TS_2": 7,
            }

            dataset = TaikoContextDataset(
                seq_index,
                token_to_id,
                history_max_tokens=32,
                retrieval_top_k=1,
                retrieval_max_tokens_per_window=8,
                retrieval_exclude_last_n_windows=1,
                use_motif_retrieval=True,
            )

            item = dataset[2]
            prefix_mask = item["labels"] == CONTEXT_LABEL_IGNORE_INDEX
            self.assertGreater(int(prefix_mask.sum().item()), 0)
            self.assertEqual(int(item["segment_ids"][-1].item()), 2)

    def test_retrieval_prefers_repeated_window_and_skips_recent_neighbor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, seq_index = self._write_chart(Path(tmpdir))
            token_to_id = {
                "PAD": 0,
                "BOS": 1,
                "EOS": 2,
                "DON": 3,
                "KAT": 4,
                "BIGDON": 5,
                "BIGKAT": 6,
                "TS_2": 7,
            }

            dataset = TaikoContextDataset(
                seq_index,
                token_to_id,
                history_max_tokens=32,
                retrieval_top_k=1,
                retrieval_max_tokens_per_window=8,
                retrieval_exclude_last_n_windows=1,
                use_motif_retrieval=True,
            )

            chart_payload = dataset._load_chart_samples("2034220_context_chart")
            current = chart_payload["by_seq"][3]
            retrieved_ids = dataset._build_retrieved_ids(chart_payload["ordered"], current)

            expected = dataset._serialize_window_token_ids([3, 7, 4], limit=8)
            self.assertEqual(retrieved_ids, expected)


if __name__ == "__main__":
    unittest.main()
