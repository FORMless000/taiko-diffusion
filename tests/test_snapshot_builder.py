import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestSnapshotBuilder(unittest.TestCase):
    def _import_snapshot_module(self):
        fake_bad = types.ModuleType("src.preprocessing.beat_aligned_dataset")
        fake_bad.run_pipeline = lambda *args, **kwargs: None
        fake_bad.setup_logging = lambda *args, **kwargs: None
        with patch.dict(sys.modules, {"src.preprocessing.beat_aligned_dataset": fake_bad}):
            module = importlib.import_module("src.preprocessing.build_snapshot_dataset")
            module = importlib.reload(module)
        return module

    def _write_parsed_triple(self, parsed_dir: Path, stem: str = "demo") -> None:
        parsed_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ["notes", "timing", "metadata"]:
            (parsed_dir / f"{stem}.{suffix}.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    def _write_constant_taiko_osu(self, path: Path) -> None:
        path.write_text(
            "[General]\n"
            "Mode: 1\n"
            "[TimingPoints]\n"
            "0,500,4,1,0,100,1,0\n",
            encoding="utf-8",
        )

    def _write_nonconstant_taiko_osu(self, path: Path) -> None:
        path.write_text(
            "[General]\n"
            "Mode: 1\n"
            "[TimingPoints]\n"
            "0,500,4,1,0,100,1,0\n"
            "1000,400,4,1,0,100,1,0\n",
            encoding="utf-8",
        )

    def _write_non_taiko_osu(self, path: Path) -> None:
        path.write_text(
            "[General]\n"
            "Mode: 0\n"
            "[TimingPoints]\n"
            "0,500,4,1,0,100,1,0\n",
            encoding="utf-8",
        )

    def _build_set(
        self,
        root: Path,
        folder_id: str,
        *,
        audio_size_bytes: int = 1024,
        osu_writers: list | None = None,
        with_parsed: bool = True,
        complete_triples: int = 1,
        extra_audio: int = 0,
    ) -> Path:
        set_dir = root / folder_id
        set_dir.mkdir(parents=True, exist_ok=True)
        (set_dir / "audio.mp3").write_bytes(b"a" * audio_size_bytes)
        for idx in range(extra_audio):
            (set_dir / f"extra_{idx}.ogg").write_bytes(b"b")

        writers = osu_writers or [self._write_constant_taiko_osu]
        for idx, writer in enumerate(writers):
            writer(set_dir / f"chart_{idx}.osu")

        if with_parsed:
            parsed_dir = set_dir / "parsed"
            for idx in range(complete_triples):
                self._write_parsed_triple(parsed_dir, stem=f"chart_{idx}")
        return set_dir

    def test_evaluate_set_folder_accepts_eligible_set(self):
        module = self._import_snapshot_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            set_dir = self._build_set(root, "1001")

            candidate, rejection = module.evaluate_set_folder(set_dir, max_audio_bytes=5 * 1024 * 1024)

            self.assertIsNotNone(candidate)
            self.assertIsNone(rejection)
            self.assertEqual(candidate.folder_id, "1001")
            self.assertEqual(candidate.n_osu_files, 1)
            self.assertEqual(candidate.n_complete_chart_triples, 1)

    def test_evaluate_set_folder_rejects_non_taiko_set(self):
        module = self._import_snapshot_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            set_dir = self._build_set(root, "1002", osu_writers=[self._write_constant_taiko_osu, self._write_non_taiko_osu])

            candidate, rejection = module.evaluate_set_folder(set_dir, max_audio_bytes=5 * 1024 * 1024)

            self.assertIsNone(candidate)
            self.assertIsNotNone(rejection)
            self.assertEqual(rejection.reason, "non_taiko_mode")

    def test_evaluate_set_folder_rejects_nonconstant_bpm_set(self):
        module = self._import_snapshot_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            set_dir = self._build_set(root, "1003", osu_writers=[self._write_nonconstant_taiko_osu])

            candidate, rejection = module.evaluate_set_folder(set_dir, max_audio_bytes=5 * 1024 * 1024)

            self.assertIsNone(candidate)
            self.assertIsNotNone(rejection)
            self.assertEqual(rejection.reason, "non_constant_bpm")

    def test_evaluate_set_folder_rejects_audio_and_parsed_issues(self):
        module = self._import_snapshot_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            audio_count_dir = self._build_set(root, "2001", extra_audio=1)
            _, rejection = module.evaluate_set_folder(audio_count_dir, max_audio_bytes=5 * 1024 * 1024)
            self.assertEqual(rejection.reason, "audio_count_error")

            too_large_dir = self._build_set(root, "2002", audio_size_bytes=(5 * 1024 * 1024) + 1)
            _, rejection = module.evaluate_set_folder(too_large_dir, max_audio_bytes=5 * 1024 * 1024)
            self.assertEqual(rejection.reason, "audio_too_large")

            missing_parsed_dir = self._build_set(root, "2003", with_parsed=False)
            _, rejection = module.evaluate_set_folder(missing_parsed_dir, max_audio_bytes=5 * 1024 * 1024)
            self.assertEqual(rejection.reason, "no_parsed_folder")

            incomplete_dir = self._build_set(root, "2004", with_parsed=True, complete_triples=0)
            parsed_dir = incomplete_dir / "parsed"
            parsed_dir.mkdir(parents=True, exist_ok=True)
            (parsed_dir / "chart_0.notes.json").write_text("{}", encoding="utf-8")
            _, rejection = module.evaluate_set_folder(incomplete_dir, max_audio_bytes=5 * 1024 * 1024)
            self.assertEqual(rejection.reason, "no_complete_chart_triples")

    def test_choose_snapshot_sets_is_seeded_and_reproducible(self):
        module = self._import_snapshot_module()
        candidates = [
            module.CandidateSetRecord(
                folder_id=str(idx),
                folder_path=f"/tmp/{idx}",
                audio_file="audio.mp3",
                audio_path=f"/tmp/{idx}/audio.mp3",
                audio_size_bytes=100,
                n_osu_files=1,
                n_complete_chart_triples=1,
            )
            for idx in range(10)
        ]

        first = module.choose_snapshot_sets(candidates, target_set_count=4, seed=42)
        second = module.choose_snapshot_sets(candidates, target_set_count=4, seed=42)
        third = module.choose_snapshot_sets(candidates, target_set_count=4, seed=7)

        self.assertEqual([item.folder_id for item in first], [item.folder_id for item in second])
        self.assertNotEqual([item.folder_id for item in first], [item.folder_id for item in third])

    def test_build_snapshot_dataset_copies_selected_sets_and_runs_pipeline(self):
        module = self._import_snapshot_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "unpacked"
            source.mkdir(parents=True, exist_ok=True)
            self._build_set(source, "3001")
            self._build_set(source, "3002")
            self._build_set(source, "3003", osu_writers=[self._write_non_taiko_osu])

            snapshot_root = root / "snapshot"

            def _fake_run_pipeline(*, unpacked_root, index_dir, dataset_dir, **kwargs):
                del kwargs
                index_dir.mkdir(parents=True, exist_ok=True)
                dataset_dir.mkdir(parents=True, exist_ok=True)
                (index_dir / "chart_build_summary.csv").write_text("chart_id,status\nx,ok\n", encoding="utf-8")
                (dataset_dir / "sequence_metadata.csv").write_text("chart_id,seq_idx\nx,0\n", encoding="utf-8")
                (dataset_dir / "audio_npz").mkdir(parents=True, exist_ok=True)
                (dataset_dir / "token_json").mkdir(parents=True, exist_ok=True)
                self.assertTrue(Path(unpacked_root).exists())

            with patch.object(module, "run_pipeline", side_effect=_fake_run_pipeline) as run_pipeline_mock:
                summary = module.build_snapshot_dataset(
                    source_unpacked_root=source,
                    snapshot_root=snapshot_root,
                    target_set_count=2,
                    seed=42,
                    max_audio_mb=5.0,
                    overwrite=False,
                    keep_only_max_notes_per_song=False,
                )

            self.assertEqual(summary["selected_set_count"], 2)
            self.assertTrue((snapshot_root / "selection_manifest.csv").exists())
            self.assertTrue((snapshot_root / "rejection_report.csv").exists())
            self.assertTrue((snapshot_root / "chart_index" / "chart_build_summary.csv").exists())
            self.assertTrue((snapshot_root / "beat_aligned_dataset" / "sequence_metadata.csv").exists())
            copied_ids = sorted(path.name for path in (snapshot_root / "unpacked").iterdir() if path.is_dir())
            self.assertEqual(len(copied_ids), 2)
            run_pipeline_mock.assert_called_once()

    def test_build_snapshot_dataset_raises_when_not_enough_eligible_sets(self):
        module = self._import_snapshot_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "unpacked"
            source.mkdir(parents=True, exist_ok=True)
            self._build_set(source, "4001")

            with self.assertRaises(RuntimeError):
                module.build_snapshot_dataset(
                    source_unpacked_root=source,
                    snapshot_root=root / "snapshot",
                    target_set_count=2,
                    seed=42,
                )


if __name__ == "__main__":
    unittest.main()
