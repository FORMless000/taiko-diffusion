from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from src.inference import (
    GenerationRequest,
    GenerationResult,
    GenerationService,
    built_in_model_registry,
    generation_request_from_payload,
    generation_result_to_payload,
    load_model_registry,
)


JobExecutor = Callable[[GenerationRequest, Path], GenerationResult | dict[str, Any]]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_upload_name(filename: str | None) -> str:
    raw = Path(str(filename or "audio.mp3")).name
    invalid = '<>:"/\\|?*'
    cleaned = "".join("_" if ch in invalid else ch for ch in raw).strip().rstrip(".")
    return cleaned or "audio.mp3"


@dataclass
class JobRecord:
    job_id: str
    model_id: str
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str = ""
    result: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        primary_artifact = next((artifact for artifact in self.artifacts if artifact.get("primary_download")), None)
        payload["primary_download_url"] = (
            f"/api/jobs/{self.job_id}/download/osz" if primary_artifact is not None and self.status == "succeeded" else None
        )
        return payload


class FileSystemJobStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def job_dir(self, job_id: str) -> Path:
        return self.root / str(job_id)

    def request_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "request.json"

    def status_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "status.json"

    def upload_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "uploads"

    def output_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "outputs"

    def create_job(self, *, job_id: str, model_id: str) -> JobRecord:
        job_dir = self.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir(job_id).mkdir(parents=True, exist_ok=True)
        self.output_dir(job_id).mkdir(parents=True, exist_ok=True)
        record = JobRecord(
            job_id=job_id,
            model_id=model_id,
            status="queued",
            created_at=_utcnow_iso(),
        )
        self.write_status(record)
        return record

    def write_status(self, record: JobRecord) -> None:
        with self._lock:
            self.status_path(record.job_id).write_text(
                json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def read_status(self, job_id: str) -> JobRecord:
        status_path = self.status_path(job_id)
        if not status_path.exists():
            raise FileNotFoundError(f"Unknown job_id '{job_id}'")
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        payload.pop("primary_download_url", None)
        return JobRecord(**payload)

    def save_request_payload(self, job_id: str, payload: dict[str, Any]) -> None:
        self.request_path(job_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_request_payload(self, job_id: str) -> dict[str, Any]:
        request_path = self.request_path(job_id)
        if not request_path.exists():
            raise FileNotFoundError(f"Missing request payload for job '{job_id}'")
        return json.loads(request_path.read_text(encoding="utf-8"))


class SingleWorkerJobRunner:
    def __init__(self, *, store: FileSystemJobStore, executor: JobExecutor):
        self.store = store
        self.executor = executor
        self._queue: Queue[str | None] = Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker_loop, name="taiko-webapp-worker", daemon=True)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        self._queue.put(None)
        self._thread.join(timeout=5)
        self._started = False

    def enqueue(self, job_id: str) -> None:
        self._queue.put(str(job_id))

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job_id = self._queue.get(timeout=0.25)
            except Empty:
                continue
            if job_id is None:
                self._queue.task_done()
                continue
            self._run_job(job_id)
            self._queue.task_done()

    def _coerce_result_payload(self, result: GenerationResult | dict[str, Any]) -> dict[str, Any]:
        if isinstance(result, GenerationResult):
            return generation_result_to_payload(result)
        return dict(result)

    def _run_job(self, job_id: str) -> None:
        record = self.store.read_status(job_id)
        record.status = "running"
        record.started_at = _utcnow_iso()
        record.error = ""
        self.store.write_status(record)

        try:
            payload = self.store.load_request_payload(job_id)
            audio_relative_path = str(payload.get("audio_relative_path", "")).strip()
            if not audio_relative_path:
                raise ValueError("Request payload did not include audio_relative_path.")
            request = generation_request_from_payload(
                payload,
                audio_path=self.store.job_dir(job_id) / audio_relative_path,
            )
            result = self.executor(request, self.store.output_dir(job_id))
            result_payload = self._coerce_result_payload(result)

            record.status = "succeeded"
            record.finished_at = _utcnow_iso()
            record.result = result_payload
            record.artifacts = list(result_payload.get("artifacts", []))
            self.store.write_status(record)
        except Exception as exc:  # pragma: no cover - exercised in integration tests with fake executor
            record.status = "failed"
            record.finished_at = _utcnow_iso()
            record.error = str(exc)
            record.result = None
            record.artifacts = []
            self.store.write_status(record)


def _parse_json_field(raw_text: str, *, field_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON for {field_name}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail=f"{field_name} must decode to a JSON object.")
    return payload


def _build_default_executor(repo_root: Path, service: GenerationService) -> JobExecutor:
    def _executor(request: GenerationRequest, output_dir: Path) -> GenerationResult:
        return service.generate(request, output_root=output_dir)

    return _executor


def _resolve_cors_origins() -> list[str]:
    raw = str(os.environ.get("TAIKO_WEBAPP_ALLOW_ORIGINS", "*")).strip()
    if not raw or raw == "*":
        return ["*"]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_model_registry_path(repo_root: Path) -> Path:
    configured = str(os.environ.get("TAIKO_MODEL_REGISTRY_PATH", "")).strip()
    if configured:
        configured_path = Path(configured)
        if not configured_path.is_absolute():
            return (repo_root / configured_path).resolve()
        return configured_path.resolve()
    return (repo_root / "webapp" / "backend" / "models.json").resolve()


def create_app(
    *,
    repo_root: str | Path | None = None,
    runtime_root: str | Path | None = None,
    executor: JobExecutor | None = None,
) -> FastAPI:
    resolved_repo_root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    resolved_runtime_root = Path(runtime_root or resolved_repo_root / "webapp" / "runtime" / "jobs").resolve()
    frontend_out_dir = resolved_repo_root / "webapp" / "frontend" / "out"

    model_registry_path = _resolve_model_registry_path(resolved_repo_root)
    try:
        model_registry = load_model_registry(model_registry_path, repo_root=resolved_repo_root)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        model_registry = built_in_model_registry(resolved_repo_root)
    generation_service = GenerationService(model_registry)
    job_store = FileSystemJobStore(resolved_runtime_root)
    runner = SingleWorkerJobRunner(
        store=job_store,
        executor=executor or _build_default_executor(resolved_repo_root, generation_service),
    )

    app = FastAPI(title="taiko-diffusion webapp", version="0.1.0")
    cors_origins = _resolve_cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.state.repo_root = resolved_repo_root
    app.state.runtime_root = resolved_runtime_root
    app.state.frontend_out_dir = frontend_out_dir
    app.state.generation_service = generation_service
    app.state.job_store = job_store
    app.state.job_runner = runner

    @app.on_event("startup")
    def _startup() -> None:
        app.state.job_runner.start()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        app.state.job_runner.stop()

    @app.get("/api/models")
    def list_models() -> dict[str, Any]:
        return {
            "models": [model.to_public_dict() for model in generation_service.list_models()],
        }

    @app.post("/api/jobs")
    async def create_job(
        model_id: str = Form(...),
        metadata_json: str = Form(...),
        timing_json: str = Form(...),
        conditioning_json: str = Form("{}"),
        sampling_override_json: str = Form("{}"),
        audio_file: UploadFile = File(...),
    ) -> dict[str, Any]:
        try:
            model = generation_service.get_model(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        metadata_payload = _parse_json_field(metadata_json, field_name="metadata_json")
        timing_payload = _parse_json_field(timing_json, field_name="timing_json")
        conditioning_payload = _parse_json_field(conditioning_json, field_name="conditioning_json")
        sampling_override_payload = _parse_json_field(sampling_override_json, field_name="sampling_override_json")

        job_id = uuid.uuid4().hex
        record = job_store.create_job(job_id=job_id, model_id=model.id)
        safe_audio_name = _safe_upload_name(audio_file.filename)
        upload_path = job_store.upload_dir(job_id) / safe_audio_name
        upload_bytes = await audio_file.read()
        upload_path.write_bytes(upload_bytes)

        request_payload = {
            "model_id": model.id,
            "audio_filename": safe_audio_name,
            "audio_relative_path": str(upload_path.relative_to(job_store.job_dir(job_id))),
            "metadata": metadata_payload,
            "timing": timing_payload,
            "conditioning": conditioning_payload,
            "sampling_override": sampling_override_payload,
        }
        job_store.save_request_payload(job_id, request_payload)
        app.state.job_runner.enqueue(job_id)
        return {"job_id": job_id, "status": record.status}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            record = job_store.read_status(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return record.to_dict()

    @app.get("/api/jobs/{job_id}/download/osz")
    def download_osz(job_id: str) -> FileResponse:
        try:
            record = job_store.read_status(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if record.status != "succeeded":
            raise HTTPException(status_code=409, detail="Job has not succeeded yet.")

        artifact = next((item for item in record.artifacts if item.get("kind") == "osz"), None)
        if artifact is None:
            raise HTTPException(status_code=404, detail="No .osz artifact is available for this job.")

        artifact_path = job_store.output_dir(job_id) / Path(str(artifact["relative_path"]))
        if not artifact_path.exists():
            raise HTTPException(status_code=404, detail=f"Artifact file is missing: {artifact_path}")
        return FileResponse(
            artifact_path,
            media_type=str(artifact.get("media_type") or "application/octet-stream"),
            filename=artifact_path.name,
        )

    @app.get("/", response_model=None)
    def root():
        index_path = frontend_out_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return JSONResponse(
            {
                "message": "Frontend build not found. Build webapp/frontend to webapp/frontend/out, then reload.",
            }
        )

    @app.get("/{full_path:path}", response_model=None)
    def serve_frontend(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")

        target_path = frontend_out_dir / full_path
        if target_path.is_file():
            return FileResponse(target_path)
        if target_path.is_dir() and (target_path / "index.html").exists():
            return FileResponse(target_path / "index.html")

        index_path = frontend_out_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return JSONResponse(
            {
                "message": "Frontend build not found. Build webapp/frontend to webapp/frontend/out, then reload.",
            },
            status_code=404,
        )

    return app
