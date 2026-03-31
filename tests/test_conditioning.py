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
    from src.model.data import build_chart_manifest, build_sequence_index, TaikoDataset, taiko_collate_fn
    from src.model.model import TaikoTransformer


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestConditioningPipeline(unittest.TestCase):
    def test_dataset_collate_and_model_forward_with_continuous_conditioning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_dir = root / "audio_npz"
            token_dir = root / "token_json"
            audio_dir.mkdir(parents=True, exist_ok=True)
            token_dir.mkdir(parents=True, exist_ok=True)

            chart_id = "2034220_test_chart"
            np.savez_compressed(audio_dir / f"{chart_id}.npz", audio_sequences=np.random.randn(1, 192, 128).astype(np.float32))

            token_payload = [{"seq_idx": 0, "tokens": ["DON", "TS_4", "KAT"]}]
            (token_dir / f"{chart_id}.json").write_text(__import__("json").dumps(token_payload), encoding="utf-8")

            meta_csv = root / "chart_meta.csv"
            pd.DataFrame(
                [
                    {
                        "chart_id": chart_id,
                        "difficulty_value": 7.2,
                        "bpm": 180.0,
                        "beatmap_id": 2034220,
                    }
                ]
            ).to_csv(meta_csv, index=False)

            manifest = build_chart_manifest(audio_dir, token_dir, chart_metadata_csv=meta_csv)
            seq_index = build_sequence_index(manifest, [chart_id])

            token_to_id = {
                "PAD": 0,
                "BOS": 1,
                "EOS": 2,
                "DON": 3,
                "KAT": 4,
                "TS_4": 5,
            }

            dataset = TaikoDataset(seq_index, token_to_id)
            batch = taiko_collate_fn([dataset[0]])

            model = TaikoTransformer(vocab_size=len(token_to_id), d_model=32, nhead=4, num_encoder_layers=1, num_decoder_layers=1)
            logits = model(
                audio=batch["audio"],
                input_ids=batch["input_ids"],
                decoder_attention_mask=batch["decoder_attention_mask"],
                difficulty_values=batch["difficulty_values"],
                density_values=batch["density_values"],
                beatmap_id_values=batch["beatmap_id_values"],
            )

            self.assertEqual(logits.shape[0], 1)
            self.assertEqual(logits.shape[1], batch["input_ids"].shape[1])
            self.assertEqual(logits.shape[2], len(token_to_id))


if __name__ == "__main__":
    unittest.main()
