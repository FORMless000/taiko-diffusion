import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.infer_from_metadata import _checkpoint_output_slug, validate_constant_bpm_timing_json
from src.inference.service import (
    GenerationMetadataInput,
    GenerationTimingInput,
    built_in_model_registry,
    build_chart_stem,
    build_metadata_json,
    build_timing_json,
    load_model_registry,
    package_osz,
)


class TestGenerationServiceHelpers(unittest.TestCase):
    def test_build_metadata_json_uses_web_defaults(self):
        metadata = GenerationMetadataInput(
            title="Song Title",
            artist="Song Artist",
            version="Oni",
        )
        chart_stem = build_chart_stem(metadata)
        payload = build_metadata_json(
            metadata,
            audio_filename="audio.mp3",
            chart_stem=chart_stem,
        )

        self.assertEqual(payload["general"]["AudioFilename"], "audio.mp3")
        self.assertEqual(payload["metadata"]["Creator"], "taiko-diffusion")
        self.assertEqual(payload["metadata"]["BeatmapSetID"], "-1")
        self.assertEqual(payload["difficulty"]["SliderMultiplier"], "1.4")
        self.assertEqual(payload["source_osu"], f"{chart_stem}.osu")

    def test_build_timing_json_validates_constant_bpm(self):
        timing_json = build_timing_json(
            GenerationTimingInput(bpm=180.0, offset_ms=250.0, meter=4),
            chart_stem="Artist - Title (Mapper) [Oni]",
        )
        validate_constant_bpm_timing_json(timing_json)

        timing_json["timing_points"].append(
            {
                "offset": 1000.0,
                "raw_offset": 1000,
                "ms_per_beat": 250.0,
                "meter": 4,
                "sample_set": 1,
                "sample_index": 0,
                "volume": 100,
                "uninherited": 1,
                "effects": 0,
            }
        )
        with self.assertRaises(ValueError):
            validate_constant_bpm_timing_json(timing_json)

    def test_package_osz_writes_audio_and_osu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "audio.mp3"
            osu_path = root / "demo.osu"
            out_path = root / "demo.osz"

            audio_path.write_bytes(b"fake mp3")
            osu_path.write_text("osu file format v14\n", encoding="utf-8")

            package_osz(audio_path, osu_path, out_path)

            self.assertTrue(out_path.exists())
            with ZipFile(out_path, "r") as archive:
                self.assertEqual(sorted(archive.namelist()), ["audio.mp3", "demo.osu"])

    def test_built_in_model_registry_seeds_expected_models(self):
        registry = built_in_model_registry(Path(__file__).resolve().parents[1])
        self.assertEqual(
            sorted(registry),
            [
                "sample_large_baseline",
                "sample_large_baseline_maxopt",
                "sample_large_context",
            ],
        )
        self.assertTrue(all(model.checkpoint_path.name == "last.ckpt" for model in registry.values()))

    def test_load_model_registry_resolves_relative_paths_and_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_path = root / "checkpoints" / "demo" / "last.ckpt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"ckpt")
            manifest_path = root / "models.json"
            manifest_path.write_text(
                """
                {
                  "models": [
                    {
                      "id": "demo_model",
                      "label": "Demo Model",
                      "checkpoint_path": "checkpoints/demo/last.ckpt",
                      "architecture_name": "taiko_transformer"
                    }
                  ]
                }
                """.strip(),
                encoding="utf-8",
            )

            registry = load_model_registry(manifest_path, repo_root=root)

            model = registry["demo_model"]
            self.assertEqual(model.checkpoint_path, checkpoint_path.resolve())
            self.assertTrue(model.enabled)
            self.assertEqual(model.default_sampling["top_k"], 8)
            self.assertEqual(model.input_fields[0]["id"], "audio_file")
            self.assertIn("osz", model.output_artifact_kinds)

    def test_load_model_registry_marks_missing_checkpoints_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "models.json"
            manifest_path.write_text(
                """
                [
                  {
                    "id": "missing_model",
                    "label": "Missing Model",
                    "checkpoint_path": "checkpoints/missing/last.ckpt",
                    "architecture_name": "taiko_transformer"
                  }
                ]
                """.strip(),
                encoding="utf-8",
            )

            registry = load_model_registry(manifest_path, repo_root=root)

            self.assertFalse(registry["missing_model"].enabled)

    def test_checkpoint_output_slug_uses_parent_directory(self):
        slug_a = _checkpoint_output_slug(
            Path("checkpoints/sample_large_baseline/last.ckpt"),
            "taiko_transformer",
        )
        slug_b = _checkpoint_output_slug(
            Path("checkpoints/sample_large_baseline_maxopt/last.ckpt"),
            "taiko_transformer",
        )
        self.assertNotEqual(slug_a, slug_b)


if __name__ == "__main__":
    unittest.main()
