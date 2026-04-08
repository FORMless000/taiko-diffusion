import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _import_bad_module():
    fake_librosa = types.ModuleType("librosa")
    fake_librosa.load = lambda *args, **kwargs: (np.zeros(8, dtype=np.float32), 8)
    fake_librosa.feature = types.SimpleNamespace(
        melspectrogram=lambda **kwargs: np.ones((128, 4), dtype=np.float32)
    )
    fake_librosa.power_to_db = lambda x, ref=None: x
    fake_librosa.frames_to_time = lambda frames, sr, hop_length, n_fft: np.asarray(frames, dtype=np.float32)

    with patch.dict(sys.modules, {"librosa": fake_librosa}):
        module = importlib.import_module("src.preprocessing.beat_aligned_dataset")
        module = importlib.reload(module)
    return module


class TestTimingDiagnostics(unittest.TestCase):
    def test_get_timing_info_raises_structured_non_constant_bpm(self):
        bad = _import_bad_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            timing_path = root / "x.timing.json"
            timing_payload = {
                "timing_points": [
                    {"offset": 0, "ms_per_beat": 500.0, "uninherited": 1, "meter": 4},
                    {"offset": 1000, "ms_per_beat": 400.0, "uninherited": 1, "meter": 4},
                    {"offset": 1200, "ms_per_beat": -50.0, "uninherited": 0, "meter": 4},
                ]
            }
            timing_path.write_text(json.dumps(timing_payload), encoding="utf-8")

            with self.assertRaises(bad.ChartBuildError) as ctx:
                bad.get_timing_info(timing_path)

            self.assertEqual(ctx.exception.error_type, "non_constant_bpm")
            self.assertEqual(ctx.exception.diagnostics["n_bpm_points"], 2)
            self.assertEqual(ctx.exception.diagnostics["unique_uninherited_mpb_count"], 2)
            self.assertTrue("400.0" in ctx.exception.diagnostics["unique_uninherited_mpb_preview"])


class TestRunPipelineCaching(unittest.TestCase):
    def _build_mapping_df(self, root: Path):
        return pd.DataFrame(
            [
                {
                    "folder_id": "123",
                    "folder_path": str(root / "unpacked" / "123"),
                    "audio_file": "audio.mp3",
                    "audio_path": str(root / "unpacked" / "123" / "audio.mp3"),
                    "chart_base": "demo",
                    "notes_path": str(root / "unpacked" / "123" / "parsed" / "demo.notes.json"),
                    "timing_path": str(root / "unpacked" / "123" / "parsed" / "demo.timing.json"),
                    "metadata_path": str(root / "unpacked" / "123" / "parsed" / "demo.metadata.json"),
                }
            ]
        )

    def test_pipeline_skips_cached_chart_when_outputs_and_metadata_exist(self):
        bad = _import_bad_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            unpacked_root = root / "unpacked"
            index_dir = root / "chart_index"
            dataset_dir = root / "beat_aligned_dataset"
            unpacked_root.mkdir(parents=True, exist_ok=True)
            index_dir.mkdir(parents=True, exist_ok=True)
            (dataset_dir / "audio_npz").mkdir(parents=True, exist_ok=True)
            (dataset_dir / "token_json").mkdir(parents=True, exist_ok=True)

            mapping_df = self._build_mapping_df(root)
            chart_id = bad.chart_uid("123", "demo")
            (dataset_dir / "audio_npz" / f"{chart_id}.npz").write_bytes(b"x")
            (dataset_dir / "token_json" / f"{chart_id}.json").write_text("[]", encoding="utf-8")

            pd.DataFrame(
                [
                    {
                        "chart_id": chart_id,
                        "folder_id": "123",
                        "chart_base": "demo",
                        "status": "ok",
                        "bpm": 120.0,
                    }
                ]
            ).to_csv(index_dir / "chart_build_summary.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {
                        "chart_id": chart_id,
                        "seq_idx": 0,
                        "n_tokens": 2,
                    }
                ]
            ).to_csv(dataset_dir / "sequence_metadata.csv", index=False, encoding="utf-8-sig")

            with patch.object(bad, "build_chart_mapping_table", return_value=(mapping_df, pd.DataFrame())), patch.object(
                bad, "process_one_chart_row"
            ) as process_mock:
                bad.run_pipeline(
                    unpacked_root=unpacked_root,
                    index_dir=index_dir,
                    dataset_dir=dataset_dir,
                    overwrite_dataset_outputs=False,
                )
                process_mock.assert_not_called()

            chart_df = pd.read_csv(index_dir / "chart_build_summary.csv")
            seq_df = pd.read_csv(dataset_dir / "sequence_metadata.csv")
            self.assertEqual(len(chart_df), 1)
            self.assertEqual(len(seq_df), 1)
            self.assertEqual(chart_df.iloc[0]["chart_id"], chart_id)

    def test_pipeline_recomputes_when_sequence_metadata_missing(self):
        bad = _import_bad_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            unpacked_root = root / "unpacked"
            index_dir = root / "chart_index"
            dataset_dir = root / "beat_aligned_dataset"
            unpacked_root.mkdir(parents=True, exist_ok=True)
            index_dir.mkdir(parents=True, exist_ok=True)
            (dataset_dir / "audio_npz").mkdir(parents=True, exist_ok=True)
            (dataset_dir / "token_json").mkdir(parents=True, exist_ok=True)

            mapping_df = self._build_mapping_df(root)
            chart_id = bad.chart_uid("123", "demo")
            (dataset_dir / "audio_npz" / f"{chart_id}.npz").write_bytes(b"x")
            (dataset_dir / "token_json" / f"{chart_id}.json").write_text("[]", encoding="utf-8")

            pd.DataFrame([{"chart_id": chart_id, "status": "ok"}]).to_csv(
                index_dir / "chart_build_summary.csv", index=False, encoding="utf-8-sig"
            )

            process_result = {
                "summary": {
                    "chart_id": chart_id,
                    "folder_id": "123",
                    "chart_base": "demo",
                    "status": "ok",
                    "error_type": "",
                    "error_detail": "",
                    "error_message": "",
                },
                "sequence_metadata": [{"chart_id": chart_id, "seq_idx": 0, "n_tokens": 2}],
            }
            with patch.object(bad, "build_chart_mapping_table", return_value=(mapping_df, pd.DataFrame())), patch.object(
                bad, "process_one_chart_row", return_value=process_result
            ) as process_mock:
                bad.run_pipeline(
                    unpacked_root=unpacked_root,
                    index_dir=index_dir,
                    dataset_dir=dataset_dir,
                    overwrite_dataset_outputs=False,
                )
                process_mock.assert_called_once()

    def test_pipeline_recomputes_when_overwrite_enabled(self):
        bad = _import_bad_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            unpacked_root = root / "unpacked"
            index_dir = root / "chart_index"
            dataset_dir = root / "beat_aligned_dataset"
            unpacked_root.mkdir(parents=True, exist_ok=True)
            index_dir.mkdir(parents=True, exist_ok=True)
            (dataset_dir / "audio_npz").mkdir(parents=True, exist_ok=True)
            (dataset_dir / "token_json").mkdir(parents=True, exist_ok=True)

            mapping_df = self._build_mapping_df(root)
            chart_id = bad.chart_uid("123", "demo")
            (dataset_dir / "audio_npz" / f"{chart_id}.npz").write_bytes(b"x")
            (dataset_dir / "token_json" / f"{chart_id}.json").write_text("[]", encoding="utf-8")
            pd.DataFrame([{"chart_id": chart_id, "status": "ok"}]).to_csv(
                index_dir / "chart_build_summary.csv", index=False, encoding="utf-8-sig"
            )
            pd.DataFrame([{"chart_id": chart_id, "seq_idx": 0}]).to_csv(
                dataset_dir / "sequence_metadata.csv", index=False, encoding="utf-8-sig"
            )

            process_result = {
                "summary": {
                    "chart_id": chart_id,
                    "folder_id": "123",
                    "chart_base": "demo",
                    "status": "ok",
                    "error_type": "",
                    "error_detail": "",
                    "error_message": "",
                },
                "sequence_metadata": [{"chart_id": chart_id, "seq_idx": 0, "n_tokens": 2}],
            }
            with patch.object(bad, "build_chart_mapping_table", return_value=(mapping_df, pd.DataFrame())), patch.object(
                bad, "process_one_chart_row", return_value=process_result
            ) as process_mock:
                bad.run_pipeline(
                    unpacked_root=unpacked_root,
                    index_dir=index_dir,
                    dataset_dir=dataset_dir,
                    overwrite_dataset_outputs=True,
                )
                process_mock.assert_called_once()

    def test_pipeline_writes_structured_error_row(self):
        bad = _import_bad_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            unpacked_root = root / "unpacked"
            index_dir = root / "chart_index"
            dataset_dir = root / "beat_aligned_dataset"
            unpacked_root.mkdir(parents=True, exist_ok=True)
            index_dir.mkdir(parents=True, exist_ok=True)
            dataset_dir.mkdir(parents=True, exist_ok=True)

            mapping_df = self._build_mapping_df(root)
            diagnostics = {
                "n_bpm_points": 4,
                "unique_uninherited_mpb_count": 2,
                "unique_uninherited_mpb_preview": "300.0|600.0",
            }
            with patch.object(bad, "build_chart_mapping_table", return_value=(mapping_df, pd.DataFrame())), patch.object(
                bad,
                "process_one_chart_row",
                side_effect=bad.ChartBuildError(
                    "non_constant_bpm",
                    "Non-constant BPM detected: [300.0, 600.0]",
                    diagnostics=diagnostics,
                ),
            ):
                bad.run_pipeline(
                    unpacked_root=unpacked_root,
                    index_dir=index_dir,
                    dataset_dir=dataset_dir,
                )

            chart_df = pd.read_csv(index_dir / "chart_build_summary.csv")
            self.assertEqual(chart_df.iloc[0]["status"], "error")
            self.assertEqual(chart_df.iloc[0]["error_type"], "non_constant_bpm")
            self.assertEqual(int(chart_df.iloc[0]["n_bpm_points"]), 4)
            self.assertEqual(int(chart_df.iloc[0]["unique_uninherited_mpb_count"]), 2)


if __name__ == "__main__":
    unittest.main()
