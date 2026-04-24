import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.infer_from_metadata import _checkpoint_output_slug, validate_constant_bpm_timing_json
from src.inference.service import (
    GenerationConditioningInput,
    GenerationMetadataInput,
    GenerationRequest,
    GenerationTimingInput,
    GenerationService,
    ModelDescriptor,
    apply_refined_blocks_to_song_output,
    built_in_model_registry,
    build_chart_stem,
    build_metadata_json,
    build_timing_json,
    convert_song_output_to_refiner_blocks,
    load_model_registry,
    mask_note_tokens_for_refinement,
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

    def test_load_model_registry_requires_bootstrap_model_for_hybrid_refine(self):
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
                      "id": "diffusion_model",
                      "label": "Diffusion Model",
                      "checkpoint_path": "checkpoints/demo/last.ckpt",
                      "architecture_name": "taiko_diffusion_refiner",
                      "inference_kind": "hybrid_refine"
                    }
                  ]
                }
                """.strip(),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_model_registry(manifest_path, repo_root=root)

    def test_load_model_registry_marks_hybrid_disabled_when_bootstrap_checkpoint_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            diffusion_checkpoint = root / "checkpoints" / "diffusion" / "last.ckpt"
            diffusion_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            diffusion_checkpoint.write_bytes(b"ckpt")
            manifest_path = root / "models.json"
            manifest_path.write_text(
                """
                {
                  "models": [
                    {
                      "id": "bootstrap",
                      "label": "Bootstrap",
                      "checkpoint_path": "checkpoints/bootstrap/last.ckpt",
                      "architecture_name": "taiko_transformer"
                    },
                    {
                      "id": "diffusion_model",
                      "label": "Diffusion Model",
                      "checkpoint_path": "checkpoints/diffusion/last.ckpt",
                      "architecture_name": "taiko_diffusion_refiner",
                      "inference_kind": "hybrid_refine",
                      "bootstrap_model_id": "bootstrap"
                    }
                  ]
                }
                """.strip(),
                encoding="utf-8",
            )

            registry = load_model_registry(manifest_path, repo_root=root)
            self.assertFalse(registry["bootstrap"].enabled)
            self.assertFalse(registry["diffusion_model"].enabled)

    def test_song_output_refiner_block_round_trip_preserves_tokens(self):
        song_output = []
        sample_tokens = [
            ["DON", "TS_48", "KAT"],
            ["TS_24", "DON"],
            ["TS_12", "KAT", "TS_12", "DON"],
            ["KAT"],
            ["TS_60", "DON"],
            ["TS_6", "KAT", "TS_30", "DON"],
            ["TS_10", "BIGDON"],
            ["TS_20", "BIGKAT"],
        ]
        for seq_idx, pred_tokens in enumerate(sample_tokens):
            start_frame = seq_idx * 192
            song_output.append(
                {
                    "seq_idx": seq_idx,
                    "start_frame": start_frame,
                    "end_frame": start_frame + 191,
                    "pred_ids": [seq_idx],
                    "pred_tokens": pred_tokens,
                }
            )

        blocks = convert_song_output_to_refiner_blocks(song_output)
        restored = apply_refined_blocks_to_song_output(song_output, blocks)

        self.assertEqual(len(blocks), 1)
        self.assertEqual([item["pred_tokens"] for item in restored], [item["pred_tokens"] for item in song_output])
        self.assertEqual([item["pred_ids"] for item in restored], [item["pred_ids"] for item in song_output])

    def test_mask_note_tokens_for_refinement_masks_only_note_positions(self):
        token_to_id = {
            "PAD": 0,
            "BOS": 1,
            "EOS": 2,
            "MASK": 3,
            "DON": 4,
            "KAT": 5,
            "TS_1": 6,
        }
        masked_ids, mask_positions = mask_note_tokens_for_refinement(
            [1, 4, 6, 5, 2, 0],
            token_to_id,
            mask_ratio=1.0,
        )

        self.assertEqual(masked_ids, [1, 3, 6, 3, 2, 0])
        self.assertEqual(mask_positions, [1, 3])

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

    def test_generation_service_hybrid_model_uses_bootstrap_then_refiner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "audio.mp3"
            audio_path.write_bytes(b"fake mp3")

            bootstrap_ckpt = root / "checkpoints" / "bootstrap" / "last.ckpt"
            diffusion_ckpt = root / "checkpoints" / "diffusion" / "last.ckpt"
            bootstrap_ckpt.parent.mkdir(parents=True, exist_ok=True)
            diffusion_ckpt.parent.mkdir(parents=True, exist_ok=True)
            bootstrap_ckpt.write_bytes(b"bootstrap")
            diffusion_ckpt.write_bytes(b"diffusion")

            bootstrap_model = ModelDescriptor(
                id="bootstrap",
                label="Bootstrap",
                checkpoint_path=bootstrap_ckpt,
                architecture_name="taiko_transformer",
                default_sampling={},
                enabled=True,
            )
            hybrid_model = ModelDescriptor(
                id="diffusion_hybrid",
                label="Diffusion Hybrid",
                checkpoint_path=diffusion_ckpt,
                architecture_name="taiko_diffusion_refiner",
                default_sampling={"mask_ratio": 0.25, "temperature": 1.1},
                inference_kind="hybrid_refine",
                bootstrap_model_id="bootstrap",
                enabled=True,
            )
            service = GenerationService(
                {
                    "bootstrap": bootstrap_model,
                    "diffusion_hybrid": hybrid_model,
                }
            )

            request = GenerationRequest(
                model_id="diffusion_hybrid",
                audio_path=audio_path,
                audio_filename="audio.mp3",
                metadata=GenerationMetadataInput(title="Song", artist="Artist", version="Oni"),
                timing=GenerationTimingInput(bpm=180.0, offset_ms=0.0, meter=4),
                conditioning=GenerationConditioningInput(density_nps=6.0),
            )
            bootstrap_song_output = [
                {
                    "seq_idx": seq_idx,
                    "start_frame": seq_idx * 192,
                    "end_frame": seq_idx * 192 + 191,
                    "pred_ids": [seq_idx],
                    "pred_tokens": ["DON"] if seq_idx % 2 == 0 else ["KAT"],
                }
                for seq_idx in range(8)
            ]
            refined_song_output = [
                {
                    **item,
                    "pred_tokens": ["BIGDON"] if item["seq_idx"] % 2 == 0 else ["BIGKAT"],
                }
                for item in bootstrap_song_output
            ]

            with mock.patch.object(
                service,
                "_generate_song_output_autoregressive",
                return_value=(bootstrap_song_output, type("Spec", (), {"name": "taiko_transformer"})()),
            ) as bootstrap_mock, mock.patch.object(
                service,
                "_refine_song_output_with_diffusion_model",
                return_value=(refined_song_output, type("Spec", (), {"name": "taiko_diffusion_refiner"})()),
            ) as refine_mock:
                result = service.generate(request, output_root=root / "outputs")

            self.assertEqual(bootstrap_mock.call_count, 1)
            self.assertEqual(refine_mock.call_count, 1)
            self.assertEqual(result.model.id, "diffusion_hybrid")
            self.assertTrue(any(artifact.kind == "osz" for artifact in result.artifacts))


if __name__ == "__main__":
    unittest.main()
