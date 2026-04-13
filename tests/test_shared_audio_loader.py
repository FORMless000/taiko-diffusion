import json
import sys
import tempfile
import unittest
from pathlib import Path
import types
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.model.data import TaikoContextDataset, build_chart_manifest, build_sequence_index, load_one_sample
except ImportError:
    TaikoContextDataset = None
    build_chart_manifest = None
    build_sequence_index = None
    load_one_sample = None


@unittest.skipIf(TaikoContextDataset is None, "torch is not installed in this environment")
class TestSharedAudioLoader(unittest.TestCase):
    def test_manifest_and_loader_support_shared_audio_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_dir = root / "beat_aligned_dataset" / "audio_npz"
            shared_dir = root / "beat_aligned_dataset" / "audio_shared_npz"
            token_dir = root / "beat_aligned_dataset" / "token_json"
            index_dir = root / "chart_index"
            audio_dir.mkdir(parents=True, exist_ok=True)
            shared_dir.mkdir(parents=True, exist_ok=True)
            token_dir.mkdir(parents=True, exist_ok=True)
            index_dir.mkdir(parents=True, exist_ok=True)

            chart_id = "1000_demo"
            shared_npz = shared_dir / "shared_demo.npz"
            np.savez_compressed(
                shared_npz,
                mel_spec_db=np.ones((4, 128), dtype=np.float32),
                orig_frame_times_ms=np.array([0.0, 200.0, 400.0, 600.0], dtype=np.float64),
                audio_duration_ms=np.array([1000.0], dtype=np.float64),
            )
            (token_dir / f"{chart_id}.json").write_text(
                json.dumps([{"seq_idx": 0, "tokens": ["DON", "TS_1", "KAT"]}]),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "chart_id": chart_id,
                        "difficulty": "Oni",
                        "difficulty_value": 7.0,
                        "bpm": 120.0,
                        "beatmap_id": 1000,
                        "density_nps": 3.0,
                        "shared_audio_id": "shared_demo",
                        "shared_audio_npz_path": str(shared_npz),
                        "offset_ms": 0.0,
                        "beat_duration_ms": 500.0,
                        "total_frames": 192,
                        "total_sequences": 1,
                    }
                ]
            ).to_csv(index_dir / "chart_build_summary.csv", index=False, encoding="utf-8-sig")

            manifest = build_chart_manifest(
                audio_dir=audio_dir,
                token_dir=token_dir,
                chart_metadata_csv=index_dir / "chart_build_summary.csv",
            )
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest.iloc[0]["chart_id"], chart_id)

            seq_index = build_sequence_index(manifest, [chart_id])
            fake_librosa = types.ModuleType("librosa")
            fake_librosa.load = lambda *args, **kwargs: (np.zeros(8, dtype=np.float32), 8)
            fake_librosa.feature = types.SimpleNamespace(
                melspectrogram=lambda **kwargs: np.ones((128, 4), dtype=np.float32)
            )
            fake_librosa.power_to_db = lambda x, ref=None: x
            fake_librosa.frames_to_time = lambda frames, sr, hop_length, n_fft: np.asarray(frames, dtype=np.float32)

            with patch.dict(sys.modules, {"librosa": fake_librosa}):
                sample = load_one_sample(seq_index.iloc[0])
                self.assertEqual(tuple(sample["audio"].shape), (192, 128))

                token_to_id = {"PAD": 0, "BOS": 1, "EOS": 2, "DON": 3, "KAT": 4, "TS_1": 5}
                context_dataset = TaikoContextDataset(seq_index_df=seq_index, token_to_id=token_to_id)
                item = context_dataset[0]
                self.assertEqual(tuple(item["audio"].shape), (192, 128))


if __name__ == "__main__":
    unittest.main()
