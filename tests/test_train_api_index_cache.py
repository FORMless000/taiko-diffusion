import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if torch is not None:
    import src.model.train_api as train_api_module
    from src.model.specs import ArchitectureSpec, TrainingSpec
    from src.model.train_api import build_training_artifacts, create_dataset_bundle


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestTrainApiIndexCache(unittest.TestCase):
    def _write_min_dataset(self, root: Path) -> None:
        audio_dir = root / "beat_aligned_dataset" / "audio_npz"
        token_dir = root / "beat_aligned_dataset" / "token_json"
        chart_index_dir = root / "chart_index"
        audio_dir.mkdir(parents=True, exist_ok=True)
        token_dir.mkdir(parents=True, exist_ok=True)
        chart_index_dir.mkdir(parents=True, exist_ok=True)

        chart_id = "1000_chart_a"
        audio = np.random.default_rng(123).standard_normal((1, 192, 128)).astype(np.float32)
        np.savez_compressed(audio_dir / f"{chart_id}.npz", audio_sequences=audio)
        token_payload = [{"seq_idx": 0, "tokens": ["DON", "TS_4", "KAT"]}]
        (token_dir / f"{chart_id}.json").write_text(json.dumps(token_payload), encoding="utf-8")

        pd.DataFrame(
            [
                {
                    "chart_id": chart_id,
                    "difficulty": "Oni",
                    "difficulty_value": 7.0,
                    "bpm": 180.0,
                    "beatmap_id": 1000,
                    "density_nps": 5.0,
                    "status": "ok",
                    "total_sequences": 1,
                    "offset_ms": 0.0,
                    "beat_duration_ms": 500.0,
                    "total_frames": 192,
                }
            ]
        ).to_csv(chart_index_dir / "chart_build_summary.csv", index=False, encoding="utf-8-sig")

        pd.DataFrame(
            [
                {
                    "chart_id": chart_id,
                    "seq_idx": 0,
                    "audio_npz_path": str(audio_dir / f"{chart_id}.npz"),
                    "token_json_path": str(token_dir / f"{chart_id}.json"),
                    "offset_ms": 0.0,
                    "beat_duration_ms": 500.0,
                    "total_frames": 192,
                    "total_sequences": 1,
                    "difficulty_value": 7.0,
                    "bpm": 180.0,
                    "beatmap_id": 1000,
                    "density_nps": 5.0,
                }
            ]
        ).to_csv(root / "beat_aligned_dataset" / "sequence_metadata.csv", index=False, encoding="utf-8-sig")

    def test_index_cache_miss_then_hit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_min_dataset(root)
            artifacts = build_training_artifacts(root)
            cache_dir = root / "training" / "index_cache_test"
            spec = TrainingSpec(epochs=1, batch_size=1, device="cpu", num_workers=0)
            arch = ArchitectureSpec(name="taiko_context_transformer", d_model=16, nhead=4, max_cached_charts=2)

            bundle = create_dataset_bundle(
                artifacts,
                spec,
                arch,
                use_index_cache=True,
                index_cache_dir=cache_dir,
                log_startup=False,
            )
            self.assertEqual(len(bundle.train_seq_index), 1)
            self.assertEqual(getattr(bundle.train_loader.dataset, "max_cached_charts", None), 2)
            self.assertEqual(int(bundle.train_loader.num_workers), 0)
            self.assertFalse(bool(bundle.train_loader.pin_memory))
            cache_entries = [p for p in cache_dir.glob("*") if p.is_dir()]
            self.assertEqual(len(cache_entries), 1)

            with patch("src.model.train_api.build_chart_manifest", side_effect=AssertionError("cache hit should skip manifest build")):
                cached_bundle = create_dataset_bundle(
                    artifacts,
                    spec,
                    arch,
                    use_index_cache=True,
                    index_cache_dir=cache_dir,
                    log_startup=False,
                )
            self.assertEqual(len(cached_bundle.train_seq_index), 1)
            self.assertEqual(getattr(cached_bundle.train_loader.dataset, "max_cached_charts", None), 2)

    def test_index_cache_invalidates_when_sequence_metadata_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_min_dataset(root)
            artifacts = build_training_artifacts(root)
            cache_dir = root / "training" / "index_cache_test"
            spec = TrainingSpec(epochs=1, batch_size=1, device="cpu", num_workers=0)
            arch = ArchitectureSpec(name="taiko_context_transformer", d_model=16, nhead=4, max_cached_charts=2)

            create_dataset_bundle(
                artifacts,
                spec,
                arch,
                use_index_cache=True,
                index_cache_dir=cache_dir,
                log_startup=False,
            )
            initial_cache_count = len([p for p in cache_dir.glob("*") if p.is_dir()])

            seq_csv = root / "beat_aligned_dataset" / "sequence_metadata.csv"
            seq_csv.write_text(seq_csv.read_text(encoding="utf-8") + "\n", encoding="utf-8")

            with patch("src.model.train_api.build_chart_manifest", wraps=train_api_module.build_chart_manifest) as manifest_mock:
                create_dataset_bundle(
                    artifacts,
                    spec,
                    arch,
                    use_index_cache=True,
                    index_cache_dir=cache_dir,
                    log_startup=False,
                )
                self.assertGreaterEqual(manifest_mock.call_count, 1)

            updated_cache_count = len([p for p in cache_dir.glob("*") if p.is_dir()])
            self.assertGreaterEqual(updated_cache_count, initial_cache_count + 1)

    def test_create_dataset_bundle_max_cached_override_precedence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_min_dataset(root)
            artifacts = build_training_artifacts(root)
            spec = TrainingSpec(epochs=1, batch_size=1, device="cpu", num_workers=0)
            arch = ArchitectureSpec(name="taiko_context_transformer", d_model=16, nhead=4, max_cached_charts=2)

            bundle = create_dataset_bundle(
                artifacts,
                spec,
                arch,
                use_index_cache=False,
                log_startup=False,
                max_cached_charts=5,
            )
            self.assertEqual(getattr(bundle.train_loader.dataset, "max_cached_charts", None), 5)
            self.assertEqual(arch.max_cached_charts, 5)


if __name__ == "__main__":
    unittest.main()
