import tempfile
import unittest
from pathlib import Path
import types
from unittest.mock import patch
from zipfile import ZipFile

import sys
import importlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.unpack_osz import UnpackSummary, unpack_osz_files


def _write_valid_osz(path: Path, chart_name: str = "demo.osu") -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr(chart_name, "osu file body")
        archive.writestr("audio.mp3", "fake")
        archive.writestr("bg.jpg", "drop me")


class TestUnpackOsz(unittest.TestCase):
    def test_skip_existing_when_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            dest = root / "dest"
            source.mkdir(parents=True, exist_ok=True)
            osz = source / "a.osz"
            _write_valid_osz(osz)

            first = unpack_osz_files(source_paths=[osz], destination_root=dest, return_summary=True)
            self.assertIsInstance(first, UnpackSummary)
            self.assertEqual(first.unpacked_ok, 1)
            self.assertEqual(first.skipped_existing, 0)

            second = unpack_osz_files(source_paths=[osz], destination_root=dest, return_summary=True)
            self.assertEqual(second.unpacked_ok, 0)
            self.assertEqual(second.skipped_existing, 1)

    def test_overwrite_reextracts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            dest = root / "dest"
            source.mkdir(parents=True, exist_ok=True)
            osz = source / "a.osz"
            _write_valid_osz(osz)

            unpack_osz_files(source_paths=[osz], destination_root=dest, return_summary=True)
            summary = unpack_osz_files(source_paths=[osz], destination_root=dest, overwrite=True, return_summary=True)
            self.assertEqual(summary.unpacked_ok, 1)
            self.assertEqual(summary.skipped_existing, 0)

    def test_corrupt_archive_is_skipped_and_partial_dir_removed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            dest = root / "dest"
            source.mkdir(parents=True, exist_ok=True)
            bad = source / "bad.osz"
            bad.write_bytes(b"not a zip")

            summary = unpack_osz_files(source_paths=[bad], destination_root=dest, return_summary=True)
            self.assertEqual(summary.failed_corrupt, 1)
            self.assertEqual(summary.unpacked_ok, 0)
            self.assertEqual(len(summary.extracted_dirs), 0)
            self.assertFalse((dest / "bad").exists())

    def test_single_tqdm_for_batch_unpack(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            dest = root / "dest"
            source.mkdir(parents=True, exist_ok=True)
            a = source / "a.osz"
            b = source / "b.osz"
            _write_valid_osz(a, chart_name="a.osu")
            _write_valid_osz(b, chart_name="b.osu")

            with patch("src.preprocessing.unpack_osz.tqdm") as tqdm_mock:
                tqdm_mock.side_effect = lambda iterable, **kwargs: iterable
                unpack_osz_files(source_paths=[a, b], destination_root=dest, return_summary=True)
                self.assertEqual(tqdm_mock.call_count, 1)


class TestPrepareTrainingDataFlow(unittest.TestCase):
    def _import_prepare_module(self):
        fake_bad = types.ModuleType("src.preprocessing.beat_aligned_dataset")
        fake_bad.run_pipeline = lambda *args, **kwargs: None
        fake_bad.setup_logging = lambda *args, **kwargs: None
        with patch.dict(sys.modules, {"src.preprocessing.beat_aligned_dataset": fake_bad}):
            module = importlib.import_module("src.preprocessing.prepare_training_data")
            module = importlib.reload(module)
        return module

    def test_prepare_calls_unpack_once_with_batch_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prepare_module = self._import_prepare_module()
            root = Path(tmpdir)
            raw_dir = root / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "a.osz").write_text("placeholder", encoding="utf-8")
            (raw_dir / "b.osz").write_text("placeholder", encoding="utf-8")

            unpacked_a = root / "dataset" / "unpacked" / "a"
            unpacked_b = root / "dataset" / "unpacked" / "b"
            with patch.object(prepare_module, "unpack_osz_files") as unpack_mock, patch.object(
                prepare_module, "parse_unpacked_beatmaps"
            ) as parse_mock, patch.object(prepare_module, "run_pipeline") as pipeline_mock:
                unpack_mock.return_value = UnpackSummary(
                    total_files=2,
                    unpacked_ok=1,
                    skipped_existing=1,
                    failed_corrupt=0,
                    extracted_dirs=[unpacked_a, unpacked_b],
                    failed_files=[],
                )

                prepare_module.prepare_training_data([raw_dir], data_root=root / "dataset")

                self.assertEqual(unpack_mock.call_count, 1)
                kwargs = unpack_mock.call_args.kwargs
                self.assertIn("source_paths", kwargs)
                self.assertEqual(len(kwargs["source_paths"]), 2)
                parse_mock.assert_called_once()
                pipeline_mock.assert_called_once()
                self.assertEqual(pipeline_mock.call_args.kwargs.get("overwrite_dataset_outputs"), False)

    def test_prepare_passes_overwrite_dataset_outputs_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prepare_module = self._import_prepare_module()
            root = Path(tmpdir)
            raw_dir = root / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "a.osz").write_text("placeholder", encoding="utf-8")

            unpacked_a = root / "dataset" / "unpacked" / "a"
            with patch.object(prepare_module, "unpack_osz_files") as unpack_mock, patch.object(
                prepare_module, "parse_unpacked_beatmaps"
            ) as parse_mock, patch.object(prepare_module, "run_pipeline") as pipeline_mock:
                unpack_mock.return_value = UnpackSummary(
                    total_files=1,
                    unpacked_ok=1,
                    skipped_existing=0,
                    failed_corrupt=0,
                    extracted_dirs=[unpacked_a],
                    failed_files=[],
                )

                prepare_module.prepare_training_data(
                    [raw_dir],
                    data_root=root / "dataset",
                    overwrite_dataset_outputs=True,
                )

                parse_mock.assert_called_once()
                self.assertEqual(pipeline_mock.call_args.kwargs.get("overwrite_dataset_outputs"), True)

    def test_prepare_raises_when_no_valid_unpacked_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prepare_module = self._import_prepare_module()
            root = Path(tmpdir)
            raw_dir = root / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "bad.osz").write_text("placeholder", encoding="utf-8")

            with patch.object(prepare_module, "unpack_osz_files") as unpack_mock, patch.object(
                prepare_module, "parse_unpacked_beatmaps"
            ) as parse_mock, patch.object(prepare_module, "run_pipeline") as pipeline_mock:
                unpack_mock.return_value = UnpackSummary(
                    total_files=1,
                    unpacked_ok=0,
                    skipped_existing=0,
                    failed_corrupt=1,
                    extracted_dirs=[],
                    failed_files=[raw_dir / "bad.osz"],
                )

                with self.assertRaises(RuntimeError):
                    prepare_module.prepare_training_data([raw_dir], data_root=root / "dataset")

                parse_mock.assert_not_called()
                pipeline_mock.assert_not_called()

    def test_parse_unpacked_fast_screens_nonconstant_and_non_taiko(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prepare_module = self._import_prepare_module()
            root = Path(tmpdir)
            unpacked_dir = root / "unpacked" / "12345"
            unpacked_dir.mkdir(parents=True, exist_ok=True)

            constant_taiko = (
                "[General]\n"
                "Mode: 1\n"
                "[TimingPoints]\n"
                "0,500,4,1,0,100,1,0\n"
            )
            nonconstant_taiko = (
                "[General]\n"
                "Mode: 1\n"
                "[TimingPoints]\n"
                "0,500,4,1,0,100,1,0\n"
                "1000,400,4,1,0,100,1,0\n"
            )
            non_taiko = (
                "[General]\n"
                "Mode: 0\n"
                "[TimingPoints]\n"
                "0,500,4,1,0,100,1,0\n"
            )
            no_bpm_points = (
                "[General]\n"
                "Mode: 1\n"
                "[TimingPoints]\n"
                "0,-50,4,1,0,100,0,0\n"
            )

            (unpacked_dir / "constant.osu").write_text(constant_taiko, encoding="utf-8")
            (unpacked_dir / "nonconstant.osu").write_text(nonconstant_taiko, encoding="utf-8")
            (unpacked_dir / "nontaiko.osu").write_text(non_taiko, encoding="utf-8")
            (unpacked_dir / "nobpm.osu").write_text(no_bpm_points, encoding="utf-8")

            with patch.object(prepare_module, "parse_osu_file_to_jsons") as parse_mock:
                parsed = prepare_module.parse_unpacked_beatmaps([unpacked_dir], overwrite_parsed=True)

            self.assertEqual(parsed, 1)
            self.assertEqual(parse_mock.call_count, 1)
            called_path = parse_mock.call_args.kwargs["osu_path"]
            self.assertEqual(called_path.name, "constant.osu")


if __name__ == "__main__":
    unittest.main()
