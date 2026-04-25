import random
import tempfile
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if torch is not None:
    from src.model.checkpoints import (
        CheckpointMetadata,
        capture_rng_states,
        export_diffusion_inference_bundle,
        load_checkpoint,
        load_inference_artifacts,
        restore_rng_states,
        save_checkpoint,
        save_inference_bundle,
    )
    from src.model.factory import build_model
    from src.model.specs import ArchitectureSpec, TrainingSpec


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestCheckpoints(unittest.TestCase):
    def test_roundtrip_restores_metadata_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ckpt_path = root / "last.ckpt"

            spec = ArchitectureSpec(
                d_model=16,
                nhead=4,
                num_encoder_layers=1,
                num_decoder_layers=1,
                dim_feedforward=32,
                max_len=32,
            )
            train_spec = TrainingSpec(epochs=3, batch_size=2, lr=1e-3, device="cpu")
            model = build_model(spec, vocab_size=8)
            optimizer = torch.optim.Adam(model.parameters(), lr=train_spec.lr)

            random.seed(123)
            np.random.seed(123)
            torch.manual_seed(123)
            saved_rng_state = capture_rng_states()

            restore_rng_states(saved_rng_state)
            expected_python = random.random()
            expected_numpy = float(np.random.rand())
            expected_torch = float(torch.rand(1).item())
            restore_rng_states(saved_rng_state)

            metadata = CheckpointMetadata(
                epoch=3,
                global_step=12,
                best_val_loss=0.456,
                data_root=str(root),
                artifact_paths={"audio_dir": "beat_aligned_dataset/audio_npz"},
            )
            history = {"train_loss": [1.0], "val_loss": [0.8], "lr": [1e-3]}
            vocab = {
                "vocab_list": ["PAD", "BOS", "EOS", "DON"],
                "token_to_id": {"PAD": 0, "BOS": 1, "EOS": 2, "DON": 3},
                "id_to_token": {0: "PAD", 1: "BOS", 2: "EOS", 3: "DON"},
            }
            split_ids = {"train": ["chart_a"], "val": ["chart_b"], "test": ["chart_c"]}

            save_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                architecture_spec=spec,
                training_spec=train_spec,
                metadata=metadata,
                history=history,
                vocab=vocab,
                split_ids=split_ids,
                adherence_config={"pad_id": 0, "ts_token_ids": []},
            )

            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)

            payload = load_checkpoint(ckpt_path, map_location="cpu")
            self.assertEqual(payload["metadata"]["epoch"], 3)
            self.assertEqual(payload["metadata"]["global_step"], 12)
            self.assertEqual(payload["split_ids"], split_ids)
            self.assertEqual(payload["vocab"]["token_to_id"], vocab["token_to_id"])

            optimizer_reloaded = torch.optim.Adam(model.parameters(), lr=99.0)
            optimizer_reloaded.load_state_dict(payload["optimizer_state_dict"])
            self.assertAlmostEqual(optimizer_reloaded.param_groups[0]["lr"], 1e-3)

            restore_rng_states(payload["rng_state"])
            self.assertAlmostEqual(random.random(), expected_python)
            self.assertAlmostEqual(float(np.random.rand()), expected_numpy)
            self.assertAlmostEqual(float(torch.rand(1).item()), expected_torch)

    def test_roundtrip_preserves_context_architecture_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ckpt_path = root / "context.ckpt"

            spec = ArchitectureSpec(
                name="taiko_context_transformer",
                d_model=16,
                nhead=4,
                num_encoder_layers=1,
                num_decoder_layers=1,
                dim_feedforward=32,
                max_len=128,
                history_max_tokens=48,
                retrieval_top_k=2,
                retrieval_max_tokens_per_window=24,
                retrieval_exclude_last_n_windows=1,
                use_motif_retrieval=True,
            )
            train_spec = TrainingSpec(epochs=3, batch_size=2, lr=1e-3, device="cpu")
            model = build_model(spec, vocab_size=8)
            optimizer = torch.optim.Adam(model.parameters(), lr=train_spec.lr)

            metadata = CheckpointMetadata(
                epoch=1,
                global_step=4,
                best_val_loss=0.123,
                data_root=str(root),
                artifact_paths={"audio_dir": "beat_aligned_dataset/audio_npz"},
            )

            save_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                architecture_spec=spec,
                training_spec=train_spec,
                metadata=metadata,
                history={"train_loss": [], "val_loss": [], "lr": []},
                vocab={
                    "vocab_list": ["PAD", "BOS", "EOS", "DON"],
                    "token_to_id": {"PAD": 0, "BOS": 1, "EOS": 2, "DON": 3},
                    "id_to_token": {0: "PAD", 1: "BOS", 2: "EOS", 3: "DON"},
                },
                split_ids={"train": ["chart_a"], "val": ["chart_b"], "test": ["chart_c"]},
                adherence_config={"pad_id": 0, "ts_token_ids": []},
            )

            payload = load_checkpoint(ckpt_path, map_location="cpu")
            restored = ArchitectureSpec.from_dict(payload["architecture_spec"])
            self.assertEqual(restored.name, "taiko_context_transformer")
            self.assertEqual(restored.history_max_tokens, 48)
            self.assertEqual(restored.retrieval_top_k, 2)

    def test_save_checkpoint_is_atomic_and_no_tmp_left(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ckpt_path = root / "atomic.ckpt"

            spec = ArchitectureSpec(
                d_model=16,
                nhead=4,
                num_encoder_layers=1,
                num_decoder_layers=1,
                dim_feedforward=32,
                max_len=32,
            )
            train_spec = TrainingSpec(epochs=1, batch_size=2, lr=1e-3, device="cpu")
            model = build_model(spec, vocab_size=8)
            optimizer = torch.optim.Adam(model.parameters(), lr=train_spec.lr)

            metadata = CheckpointMetadata(
                epoch=1,
                global_step=1,
                best_val_loss=1.0,
                data_root=str(root),
                artifact_paths={"audio_dir": "beat_aligned_dataset/audio_npz"},
            )
            save_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                architecture_spec=spec,
                training_spec=train_spec,
                metadata=metadata,
                history={"train_loss": [1.0], "val_loss": [1.0], "lr": [1e-3]},
                vocab={
                    "vocab_list": ["PAD", "BOS", "EOS", "DON"],
                    "token_to_id": {"PAD": 0, "BOS": 1, "EOS": 2, "DON": 3},
                    "id_to_token": {0: "PAD", 1: "BOS", 2: "EOS", 3: "DON"},
                },
                split_ids={"train": ["a"], "val": ["b"], "test": ["c"]},
                adherence_config={"pad_id": 0, "ts_token_ids": []},
            )

            self.assertTrue(ckpt_path.exists())
            payload = load_checkpoint(ckpt_path, map_location="cpu")
            self.assertEqual(payload["metadata"]["epoch"], 1)
            tmp_files = list(root.glob(".atomic.ckpt.*.tmp"))
            self.assertEqual(tmp_files, [])

    def test_save_inference_bundle_is_minimal_and_loadable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_path = root / "snapshots" / "step_001000.pt"

            spec = ArchitectureSpec(
                d_model=16,
                nhead=4,
                num_encoder_layers=1,
                num_decoder_layers=1,
                dim_feedforward=32,
                max_len=32,
            )
            model = build_model(spec, vocab_size=8)
            vocab = {
                "vocab_list": ["PAD", "BOS", "EOS", "DON"],
                "token_to_id": {"PAD": 0, "BOS": 1, "EOS": 2, "DON": 3},
                "id_to_token": {0: "PAD", 1: "BOS", 2: "EOS", 3: "DON"},
            }

            save_inference_bundle(
                bundle_path,
                model=model,
                architecture_spec=spec,
                vocab=vocab,
                global_step=1000,
                epoch=2,
                adherence_config={"pad_id": 0, "ts_token_ids": []},
                metadata={"source": "unit_test"},
            )

            raw_payload = load_checkpoint(bundle_path, map_location="cpu")
            self.assertEqual(raw_payload["artifact_type"], "inference_bundle")
            self.assertNotIn("optimizer_state_dict", raw_payload)
            self.assertNotIn("scheduler_state_dict", raw_payload)
            self.assertNotIn("rng_state", raw_payload)

            inference_payload = load_inference_artifacts(bundle_path, map_location="cpu")
            self.assertEqual(inference_payload["metadata"]["global_step"], 1000)
            self.assertEqual(inference_payload["metadata"]["epoch"], 2)
            self.assertEqual(inference_payload["vocab"]["token_to_id"], vocab["token_to_id"])

    def test_load_inference_artifacts_accepts_full_training_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ckpt_path = root / "last.ckpt"

            spec = ArchitectureSpec(
                d_model=16,
                nhead=4,
                num_encoder_layers=1,
                num_decoder_layers=1,
                dim_feedforward=32,
                max_len=32,
            )
            train_spec = TrainingSpec(epochs=1, batch_size=2, lr=1e-3, device="cpu")
            model = build_model(spec, vocab_size=8)
            optimizer = torch.optim.Adam(model.parameters(), lr=train_spec.lr)
            vocab = {
                "vocab_list": ["PAD", "BOS", "EOS", "DON"],
                "token_to_id": {"PAD": 0, "BOS": 1, "EOS": 2, "DON": 3},
                "id_to_token": {0: "PAD", 1: "BOS", 2: "EOS", 3: "DON"},
            }

            save_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                architecture_spec=spec,
                training_spec=train_spec,
                metadata=CheckpointMetadata(
                    epoch=1,
                    global_step=4,
                    best_val_loss=0.1,
                    data_root=str(root),
                    artifact_paths={"audio_dir": "beat_aligned_dataset/audio_npz"},
                ),
                history={"train_loss": [1.0], "val_loss": [0.5], "lr": [1e-3]},
                vocab=vocab,
                split_ids={"train": ["a"], "val": ["b"], "test": ["c"]},
                adherence_config={"pad_id": 0, "ts_token_ids": []},
            )

            inference_payload = load_inference_artifacts(ckpt_path, map_location="cpu")
            self.assertEqual(inference_payload["artifact_type"], "training_checkpoint")
            self.assertEqual(inference_payload["metadata"]["global_step"], 4)
            self.assertEqual(inference_payload["vocab"]["token_to_id"], vocab["token_to_id"])

    def test_export_diffusion_inference_bundle_from_raw_checkpoint_and_vocab(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_checkpoint_path = root / "last.ckpt"
            vocab_path = root / "vocab.pth"
            bundle_path = root / "inference_bundle.pt"

            spec = ArchitectureSpec(
                name="taiko_diffusion_refiner",
                input_dim=128,
                d_model=16,
                nhead=4,
                num_encoder_layers=1,
                num_decoder_layers=1,
                dim_feedforward=32,
                dropout=0.2,
                max_len=64,
            )
            model = build_model(spec, vocab_size=7)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            vocab = {
                "vocab_list": ["PAD", "BOS", "EOS", "MASK", "DON", "KAT", "TS_3"],
                "token_to_id": {"PAD": 0, "BOS": 1, "EOS": 2, "MASK": 3, "DON": 4, "KAT": 5, "TS_3": 6},
                "id_to_token": {0: "PAD", 1: "BOS", 2: "EOS", 3: "MASK", 4: "DON", 5: "KAT", 6: "TS_3"},
            }

            torch.save(
                {
                    "epoch": 7,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                raw_checkpoint_path,
            )
            torch.save(vocab, vocab_path)

            export_diffusion_inference_bundle(
                bundle_path,
                raw_checkpoint_path=raw_checkpoint_path,
                vocab_path=vocab_path,
                architecture_spec=spec,
            )

            payload = load_inference_artifacts(bundle_path, map_location="cpu")
            self.assertEqual(payload["artifact_type"], "inference_bundle")
            self.assertEqual(payload["metadata"]["epoch"], 7)
            self.assertEqual(payload["vocab"]["token_to_id"]["MASK"], 3)
            self.assertEqual(payload["architecture_spec"]["name"], "taiko_diffusion_refiner")


if __name__ == "__main__":
    unittest.main()
