import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webapp.backend.app import create_app


def _fake_executor(request, output_dir):
    chart_output_dir = output_dir / request.model_id / "Demo Chart"
    chart_output_dir.mkdir(parents=True, exist_ok=True)
    osz_path = chart_output_dir / "demo.generated.osz"
    osz_path.write_bytes(b"fake osz payload")
    return {
        "model": {
            "id": request.model_id,
            "label": request.model_id,
            "architecture_name": "fake_architecture",
            "default_sampling": {},
            "input_fields": [],
            "output_artifact_kinds": ["osz"],
            "enabled": True,
        },
        "chart_stem": "Demo Chart",
        "output_dir": str(chart_output_dir),
        "artifacts": [
            {
                "id": "generated_osz",
                "kind": "osz",
                "label": "Generated OSZ",
                "relative_path": str(osz_path.relative_to(output_dir)),
                "media_type": "application/octet-stream",
                "primary_download": True,
            }
        ],
    }


class TestWebappBackend(unittest.TestCase):
    def test_models_endpoint_uses_backend_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_path = root / "custom-checkpoints" / "my-model" / "last.ckpt"
            hybrid_path = root / "custom-checkpoints" / "my-hybrid" / "bundle.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            hybrid_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"ckpt")
            hybrid_path.write_bytes(b"bundle")
            manifest_path = root / "backend-models.json"
            manifest_path.write_text(
                """
                {
                  "models": [
                    {
                      "id": "backend_only_model",
                      "label": "Backend Only Model",
                      "checkpoint_path": "custom-checkpoints/my-model/last.ckpt",
                      "architecture_name": "taiko_transformer"
                    },
                    {
                      "id": "hybrid_model",
                      "label": "Hybrid Model",
                      "checkpoint_path": "custom-checkpoints/my-hybrid/bundle.pt",
                      "architecture_name": "taiko_diffusion_refiner",
                      "inference_kind": "hybrid_refine",
                      "bootstrap_model_id": "backend_only_model"
                    }
                  ]
                }
                """.strip(),
                encoding="utf-8",
            )

            previous_path = os.environ.get("TAIKO_MODEL_REGISTRY_PATH")
            os.environ["TAIKO_MODEL_REGISTRY_PATH"] = str(manifest_path)
            try:
                app = create_app(
                    repo_root=root,
                    runtime_root=root / "jobs",
                    executor=_fake_executor,
                )
                with TestClient(app) as client:
                    response = client.get("/api/models")
                    self.assertEqual(response.status_code, 200)
                    payload = response.json()
                    self.assertEqual(len(payload["models"]), 2)
                    by_id = {model["id"]: model for model in payload["models"]}
                    self.assertTrue(by_id["backend_only_model"]["enabled"])
                    self.assertEqual(by_id["hybrid_model"]["inference_kind"], "hybrid_refine")
                    self.assertEqual(by_id["hybrid_model"]["bootstrap_model_id"], "backend_only_model")
            finally:
                if previous_path is None:
                    os.environ.pop("TAIKO_MODEL_REGISTRY_PATH", None)
                else:
                    os.environ["TAIKO_MODEL_REGISTRY_PATH"] = previous_path

    def test_job_queue_and_download_flow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(
                repo_root=Path(__file__).resolve().parents[1],
                runtime_root=Path(tmpdir) / "jobs",
                executor=_fake_executor,
            )

            with TestClient(app) as client:
                models_response = client.get("/api/models")
                self.assertEqual(models_response.status_code, 200)
                models_payload = models_response.json()
                enabled_model = next(model for model in models_payload["models"] if model["enabled"])

                response = client.post(
                    "/api/jobs",
                    data={
                        "model_id": enabled_model["id"],
                        "metadata_json": json.dumps(
                            {
                                "title": "Song",
                                "artist": "Artist",
                                "version": "Oni",
                                "creator": "taiko-diffusion",
                            }
                        ),
                        "timing_json": json.dumps(
                            {
                                "bpm": 180.0,
                                "offset_ms": 0.0,
                                "meter": 4,
                            }
                        ),
                        "conditioning_json": json.dumps({"density_nps": 6.0}),
                        "sampling_override_json": json.dumps({}),
                    },
                    files={"audio_file": ("audio.mp3", b"fake mp3 payload", "audio/mpeg")},
                )
                self.assertEqual(response.status_code, 200)
                job_id = response.json()["job_id"]

                status_payload = None
                for _ in range(20):
                    status_response = client.get(f"/api/jobs/{job_id}")
                    self.assertEqual(status_response.status_code, 200)
                    status_payload = status_response.json()
                    if status_payload["status"] == "succeeded":
                        break
                    time.sleep(0.1)

                assert status_payload is not None
                self.assertEqual(status_payload["status"], "succeeded")
                self.assertEqual(len(status_payload["artifacts"]), 1)
                self.assertEqual(status_payload["artifacts"][0]["kind"], "osz")
                self.assertEqual(status_payload["primary_download_url"], f"/api/jobs/{job_id}/download/osz")

                download_response = client.get(f"/api/jobs/{job_id}/download/osz")
                self.assertEqual(download_response.status_code, 200)
                self.assertEqual(download_response.content, b"fake osz payload")

    def test_job_queue_accepts_temperature_sampling_override(self):
        captured = {}

        def _capturing_executor(request, output_dir):
            captured["sampling_override"] = dict(request.sampling_override or {})
            return _fake_executor(request, output_dir)

        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(
                repo_root=Path(__file__).resolve().parents[1],
                runtime_root=Path(tmpdir) / "jobs",
                executor=_capturing_executor,
            )

            with TestClient(app) as client:
                models_response = client.get("/api/models")
                self.assertEqual(models_response.status_code, 200)
                models_payload = models_response.json()
                enabled_model = next(model for model in models_payload["models"] if model["enabled"])

                response = client.post(
                    "/api/jobs",
                    data={
                        "model_id": enabled_model["id"],
                        "metadata_json": json.dumps(
                            {
                                "title": "Song",
                                "artist": "Artist",
                                "version": "Oni",
                                "creator": "taiko-diffusion",
                            }
                        ),
                        "timing_json": json.dumps(
                            {
                                "bpm": 180.0,
                                "offset_ms": 0.0,
                                "meter": 4,
                            }
                        ),
                        "conditioning_json": json.dumps({"density_nps": 6.0}),
                        "sampling_override_json": json.dumps({"temperature": 1.3}),
                    },
                    files={"audio_file": ("audio.mp3", b"fake mp3 payload", "audio/mpeg")},
                )
                self.assertEqual(response.status_code, 200)
                job_id = response.json()["job_id"]

                for _ in range(20):
                    status_response = client.get(f"/api/jobs/{job_id}")
                    self.assertEqual(status_response.status_code, 200)
                    status_payload = status_response.json()
                    if status_payload["status"] == "succeeded":
                        break
                    time.sleep(0.1)

                self.assertEqual(captured["sampling_override"], {"temperature": 1.3})

    def test_job_queue_forwards_beatmap_id_as_conditioning_only(self):
        captured = {}

        def _capturing_executor(request, output_dir):
            captured["conditioning_beatmap_id"] = request.conditioning.beatmap_id
            captured["metadata_beatmap_id"] = request.metadata.beatmap_id
            return _fake_executor(request, output_dir)

        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(
                repo_root=Path(__file__).resolve().parents[1],
                runtime_root=Path(tmpdir) / "jobs",
                executor=_capturing_executor,
            )

            with TestClient(app) as client:
                models_response = client.get("/api/models")
                self.assertEqual(models_response.status_code, 200)
                models_payload = models_response.json()
                enabled_model = next(model for model in models_payload["models"] if model["enabled"])

                response = client.post(
                    "/api/jobs",
                    data={
                        "model_id": enabled_model["id"],
                        "metadata_json": json.dumps(
                            {
                                "title": "Song",
                                "artist": "Artist",
                                "version": "Oni",
                                "creator": "taiko-diffusion",
                            }
                        ),
                        "timing_json": json.dumps(
                            {
                                "bpm": 180.0,
                                "offset_ms": 0.0,
                                "meter": 4,
                            }
                        ),
                        "conditioning_json": json.dumps({"density_nps": 6.0, "beatmap_id": 2034220}),
                        "sampling_override_json": json.dumps({}),
                    },
                    files={"audio_file": ("audio.mp3", b"fake mp3 payload", "audio/mpeg")},
                )
                self.assertEqual(response.status_code, 200)
                job_id = response.json()["job_id"]

                for _ in range(20):
                    status_response = client.get(f"/api/jobs/{job_id}")
                    self.assertEqual(status_response.status_code, 200)
                    status_payload = status_response.json()
                    if status_payload["status"] == "succeeded":
                        break
                    time.sleep(0.1)

                self.assertEqual(captured["conditioning_beatmap_id"], 2034220)
                self.assertEqual(captured["metadata_beatmap_id"], 0)


if __name__ == "__main__":
    unittest.main()
