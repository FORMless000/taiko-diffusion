import json
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
    from src.model.checkpoints import load_checkpoint
    from src.model.train_cli import main as train_main


def _write_dummy_dataset(data_root: Path, num_charts: int = 10) -> None:
    audio_dir = data_root / "beat_aligned_dataset" / "audio_npz"
    token_dir = data_root / "beat_aligned_dataset" / "token_json"
    index_dir = data_root / "chart_index"
    audio_dir.mkdir(parents=True, exist_ok=True)
    token_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rng = np.random.default_rng(123)
    for idx in range(num_charts):
        chart_id = f"{1000 + idx}_chart_{idx}"
        audio = rng.standard_normal((1, 192, 128)).astype(np.float32)
        np.savez_compressed(audio_dir / f"{chart_id}.npz", audio_sequences=audio)
        token_payload = [{"seq_idx": 0, "tokens": ["DON", "TS_4", "KAT"]}]
        (token_dir / f"{chart_id}.json").write_text(json.dumps(token_payload), encoding="utf-8")
        rows.append(
            {
                "chart_id": chart_id,
                "difficulty": "Oni",
                "difficulty_value": 7.0,
                "bpm": 180.0,
                "beatmap_id": 1000 + idx,
                "density_nps": 5.0,
            }
        )

    pd.DataFrame(rows).to_csv(index_dir / "chart_build_summary.csv", index=False)


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestTrainCli(unittest.TestCase):
    def test_cli_smoke_run_writes_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            _write_dummy_dataset(data_root)

            rc = train_main(
                [
                    "--data-root",
                    str(data_root),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--lr",
                    "0.001",
                    "--device",
                    "cpu",
                    "--d-model",
                    "16",
                    "--nhead",
                    "4",
                    "--num-encoder-layers",
                    "1",
                    "--num-decoder-layers",
                    "1",
                    "--dim-feedforward",
                    "32",
                    "--max-len",
                    "32",
                ]
            )

            self.assertEqual(rc, 0)
            self.assertTrue((data_root / "training" / "splits.json").exists())
            self.assertTrue((data_root / "training" / "vocab.json").exists())
            self.assertTrue((data_root / "training" / "checkpoints" / "last.ckpt").exists())

            payload = load_checkpoint(data_root / "training" / "checkpoints" / "last.ckpt", map_location="cpu")
            self.assertEqual(payload["metadata"]["epoch"], 1)

    def test_cli_resume_reuses_saved_vocab_and_splits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            _write_dummy_dataset(data_root)

            first_args = [
                "--data-root",
                str(data_root),
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--lr",
                "0.001",
                "--device",
                "cpu",
                "--d-model",
                "16",
                "--nhead",
                "4",
                "--num-encoder-layers",
                "1",
                "--num-decoder-layers",
                "1",
                "--dim-feedforward",
                "32",
                "--max-len",
                "32",
            ]
            self.assertEqual(train_main(first_args), 0)

            splits_before = (data_root / "training" / "splits.json").read_text(encoding="utf-8")
            vocab_before = (data_root / "training" / "vocab.json").read_text(encoding="utf-8")
            checkpoint_path = data_root / "training" / "checkpoints" / "last.ckpt"

            second_args = [
                "--resume-checkpoint",
                str(checkpoint_path),
                "--epochs",
                "2",
                "--batch-size",
                "4",
                "--lr",
                "0.001",
                "--device",
                "cpu",
            ]
            self.assertEqual(train_main(second_args), 0)

            payload = load_checkpoint(checkpoint_path, map_location="cpu")
            self.assertEqual(payload["metadata"]["epoch"], 2)
            self.assertEqual((data_root / "training" / "splits.json").read_text(encoding="utf-8"), splits_before)
            self.assertEqual((data_root / "training" / "vocab.json").read_text(encoding="utf-8"), vocab_before)


if __name__ == "__main__":
    unittest.main()
