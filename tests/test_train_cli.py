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
    from src.model.checkpoints import load_checkpoint
    from src.model.specs import ArchitectureSpec, TrainingSpec
    from src.model.train_cli import _build_architecture_spec, _build_training_spec, build_arg_parser, main as train_main


class _FakeWandbRun:
    def __init__(self):
        self.logged: list[dict[str, float | int | str]] = []
        self.defined_metrics: list[tuple[str, str | None]] = []
        self.finished = False

    def log(self, payload):
        self.logged.append(dict(payload))

    def define_metric(self, name, step_metric=None):
        self.defined_metrics.append((str(name), None if step_metric is None else str(step_metric)))

    def finish(self):
        self.finished = True


class _FakeWandbModule:
    def __init__(self):
        self.runs: list[_FakeWandbRun] = []
        self.login_calls: list[dict] = []

    def init(self, **kwargs):
        run = _FakeWandbRun()
        run.init_kwargs = kwargs
        self.runs.append(run)
        return run

    def login(self, **kwargs):
        self.login_calls.append(dict(kwargs))
        return True


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
    def test_build_architecture_spec_fresh_uses_fast_context_defaults(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--data-root", "unused", "--architecture-name", "taiko_context_transformer"])
        spec = _build_architecture_spec(args, checkpoint_payload=None)
        self.assertEqual(spec.history_max_tokens, 256)
        self.assertEqual(spec.retrieval_top_k, 1)
        self.assertEqual(spec.retrieval_max_tokens_per_window, 24)
        self.assertEqual(spec.retrieval_exclude_last_n_windows, 2)
        self.assertTrue(spec.use_motif_retrieval)

    def test_build_architecture_spec_resume_keeps_context_budget_without_override(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--data-root", "unused"])
        payload = {
            "architecture_spec": ArchitectureSpec(
                name="taiko_context_transformer",
                history_max_tokens=1024,
                retrieval_top_k=2,
                retrieval_max_tokens_per_window=64,
                retrieval_exclude_last_n_windows=3,
                use_motif_retrieval=True,
                max_cached_charts=5,
            ).to_dict()
        }
        spec = _build_architecture_spec(args, checkpoint_payload=payload)
        self.assertEqual(spec.history_max_tokens, 1024)
        self.assertEqual(spec.retrieval_top_k, 2)
        self.assertEqual(spec.retrieval_max_tokens_per_window, 64)
        self.assertEqual(spec.retrieval_exclude_last_n_windows, 3)
        self.assertTrue(spec.use_motif_retrieval)
        self.assertEqual(spec.max_cached_charts, 5)

    def test_build_architecture_spec_resume_applies_explicit_context_overrides(self):
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--data-root",
                "unused",
                "--history-max-tokens",
                "320",
                "--retrieval-top-k",
                "1",
                "--retrieval-max-tokens-per-window",
                "20",
                "--retrieval-exclude-last-n-windows",
                "1",
                "--no-use-motif-retrieval",
                "--max-cached-charts",
                "7",
            ]
        )
        payload = {
            "architecture_spec": ArchitectureSpec(
                name="taiko_context_transformer",
                history_max_tokens=1024,
                retrieval_top_k=2,
                retrieval_max_tokens_per_window=64,
                retrieval_exclude_last_n_windows=3,
                use_motif_retrieval=True,
                max_cached_charts=5,
            ).to_dict()
        }
        spec = _build_architecture_spec(args, checkpoint_payload=payload)
        self.assertEqual(spec.history_max_tokens, 320)
        self.assertEqual(spec.retrieval_top_k, 1)
        self.assertEqual(spec.retrieval_max_tokens_per_window, 20)
        self.assertEqual(spec.retrieval_exclude_last_n_windows, 1)
        self.assertFalse(spec.use_motif_retrieval)
        self.assertEqual(spec.max_cached_charts, 7)

    def test_build_training_spec_sets_auto_precision_for_fresh_run(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--data-root", "unused"])
        spec = _build_training_spec(args, checkpoint_payload=None)
        self.assertEqual(spec.precision, "auto")

    def test_build_training_spec_resume_keeps_precision_without_override(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--data-root", "unused"])
        payload = {
            "training_spec": TrainingSpec(
                epochs=3,
                batch_size=2,
                lr=1e-3,
                device="cpu",
                precision="fp16",
            ).to_dict()
        }
        spec = _build_training_spec(args, checkpoint_payload=payload)
        self.assertEqual(spec.precision, "fp16")

    def test_build_training_spec_resume_applies_precision_override(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--data-root", "unused", "--precision", "bf16"])
        payload = {
            "training_spec": TrainingSpec(
                epochs=3,
                batch_size=2,
                lr=1e-3,
                device="cpu",
                precision="fp16",
            ).to_dict()
        }
        spec = _build_training_spec(args, checkpoint_payload=payload)
        self.assertEqual(spec.precision, "bf16")

    def test_cli_passes_preprocessing_flags_when_raw_osz_is_provided(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "dataset"
            raw_dir = Path(tmpdir) / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "a.osz").write_text("placeholder", encoding="utf-8")

            with patch("src.preprocessing.prepare_training_data.prepare_training_data") as prepare_mock:
                with self.assertRaises(FileNotFoundError):
                    train_main(
                        [
                            str(raw_dir),
                            "--data-root",
                            str(data_root),
                            "--overwrite-unpack",
                            "--overwrite-parsed",
                            "--overwrite-dataset-outputs",
                            "--allow-offgrid-notes",
                            "--offgrid-tolerance-ms",
                            "7.5",
                            "--keep-only-max-notes-per-song",
                            "--epochs",
                            "1",
                            "--device",
                            "cpu",
                        ]
                    )

            prepare_mock.assert_called_once()
            kwargs = prepare_mock.call_args.kwargs
            self.assertEqual(kwargs["osz_inputs"], [str(raw_dir)])
            self.assertEqual(str(kwargs["data_root"]), str(data_root.resolve()))
            self.assertTrue(kwargs["overwrite_unpack"])
            self.assertTrue(kwargs["overwrite_parsed"])
            self.assertTrue(kwargs["overwrite_dataset_outputs"])
            self.assertFalse(kwargs["reject_offgrid_notes"])
            self.assertEqual(kwargs["offgrid_tolerance_ms"], 7.5)
            self.assertTrue(kwargs["keep_only_max_notes_per_song"])

    def test_cli_without_wandb_does_not_import_wandb_module(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            _write_dummy_dataset(data_root)
            checkpoints_dir = data_root / "repo_checkpoints" / "context"

            original_wandb = sys.modules.get("wandb")
            sys.modules["wandb"] = None
            try:
                rc = train_main(
                    [
                        "--data-root",
                        str(data_root),
                        "--checkpoints-dir",
                        str(checkpoints_dir),
                        "--epochs",
                        "1",
                        "--batch-size",
                        "2",
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
            finally:
                if original_wandb is None:
                    sys.modules.pop("wandb", None)
                else:
                    sys.modules["wandb"] = original_wandb

            self.assertEqual(rc, 0)
            self.assertTrue((checkpoints_dir / "last.ckpt").exists())

    def test_cli_smoke_run_writes_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            _write_dummy_dataset(data_root)

            checkpoints_dir = data_root / "repo_checkpoints" / "context"
            rc = train_main(
                [
                    "--data-root",
                    str(data_root),
                    "--checkpoints-dir",
                    str(checkpoints_dir),
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
            self.assertTrue((checkpoints_dir / "last.ckpt").exists())
            self.assertTrue((checkpoints_dir / "best.ckpt").exists())

            payload = load_checkpoint(checkpoints_dir / "last.ckpt", map_location="cpu")
            self.assertEqual(payload["metadata"]["epoch"], 1)

    def test_cli_resume_reuses_saved_vocab_and_splits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            _write_dummy_dataset(data_root)

            first_args = [
                "--data-root",
                str(data_root),
                "--checkpoints-dir",
                str(data_root / "repo_checkpoints" / "context"),
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
            checkpoint_path = data_root / "repo_checkpoints" / "context" / "last.ckpt"

            second_args = [
                "--resume-checkpoint",
                str(checkpoint_path),
                "--checkpoints-dir",
                str(data_root / "repo_checkpoints" / "context"),
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

    def test_cli_context_max_cached_charts_persists_in_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            _write_dummy_dataset(data_root)
            checkpoints_dir = data_root / "repo_checkpoints" / "context"

            rc = train_main(
                [
                    "--data-root",
                    str(data_root),
                    "--checkpoints-dir",
                    str(checkpoints_dir),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--lr",
                    "0.001",
                    "--device",
                    "cpu",
                    "--architecture-name",
                    "taiko_context_transformer",
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
                    "--max-cached-charts",
                    "3",
                ]
            )

            self.assertEqual(rc, 0)
            payload = load_checkpoint(checkpoints_dir / "last.ckpt", map_location="cpu")
            self.assertEqual(payload["architecture_spec"]["name"], "taiko_context_transformer")
            self.assertEqual(payload["architecture_spec"]["max_cached_charts"], 3)

    def test_cli_context_resume_keeps_or_overrides_max_cached_charts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            _write_dummy_dataset(data_root)
            checkpoints_dir = data_root / "repo_checkpoints" / "context"
            checkpoint_path = checkpoints_dir / "last.ckpt"

            first_run = [
                "--data-root",
                str(data_root),
                "--checkpoints-dir",
                str(checkpoints_dir),
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--lr",
                "0.001",
                "--device",
                "cpu",
                "--architecture-name",
                "taiko_context_transformer",
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
                "--max-cached-charts",
                "3",
            ]
            self.assertEqual(train_main(first_run), 0)
            self.assertEqual(load_checkpoint(checkpoint_path, map_location="cpu")["architecture_spec"]["max_cached_charts"], 3)

            resume_without_override = [
                "--resume-checkpoint",
                str(checkpoint_path),
                "--checkpoints-dir",
                str(checkpoints_dir),
                "--epochs",
                "2",
                "--batch-size",
                "4",
                "--device",
                "cpu",
            ]
            self.assertEqual(train_main(resume_without_override), 0)
            self.assertEqual(load_checkpoint(checkpoint_path, map_location="cpu")["architecture_spec"]["max_cached_charts"], 3)

            resume_with_override = [
                "--resume-checkpoint",
                str(checkpoint_path),
                "--checkpoints-dir",
                str(checkpoints_dir),
                "--epochs",
                "3",
                "--batch-size",
                "4",
                "--device",
                "cpu",
                "--max-cached-charts",
                "6",
            ]
            self.assertEqual(train_main(resume_with_override), 0)
            self.assertEqual(load_checkpoint(checkpoint_path, map_location="cpu")["architecture_spec"]["max_cached_charts"], 6)

    def test_cli_wandb_logs_batch_and_epoch_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            _write_dummy_dataset(data_root, num_charts=12)
            checkpoints_dir = data_root / "repo_checkpoints" / "context"

            fake_wandb = _FakeWandbModule()
            original_wandb = sys.modules.get("wandb")
            sys.modules["wandb"] = fake_wandb
            try:
                rc = train_main(
                    [
                        "--data-root",
                        str(data_root),
                        "--checkpoints-dir",
                        str(checkpoints_dir),
                        "--epochs",
                        "1",
                        "--batch-size",
                        "2",
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
                        "--wandb",
                        "--wandb-api-key",
                        "test-key",
                        "--wandb-notebook-name",
                        "test_train_cli.ipynb",
                        "--wandb-run-name",
                        "test",
                        "--wandb-log-every-batches",
                        "1",
                    ]
                )
            finally:
                if original_wandb is None:
                    sys.modules.pop("wandb", None)
                else:
                    sys.modules["wandb"] = original_wandb

            self.assertEqual(rc, 0)
            self.assertEqual(len(fake_wandb.runs), 1)
            run = fake_wandb.runs[0]
            self.assertTrue(run.finished)
            self.assertIn("project", run.init_kwargs)
            self.assertEqual(run.init_kwargs["project"], "taiko-transformer")
            self.assertEqual(run.init_kwargs["entity"], "yiy523-lehigh-university")
            self.assertEqual(len(fake_wandb.login_calls), 1)

            flattened_keys = {key for payload in run.logged for key in payload.keys()}
            self.assertIn("train/loss_batch", flattened_keys)
            self.assertIn("val/loss_batch", flattened_keys)
            self.assertIn("train/loss_epoch", flattened_keys)
            self.assertIn("val/loss_epoch", flattened_keys)
            self.assertIn("optimizer/lr", flattened_keys)
            self.assertIn("checkpoint/last_path", flattened_keys)
            self.assertIn("checkpoint/best_updated", flattened_keys)


if __name__ == "__main__":
    unittest.main()
