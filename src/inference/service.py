from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from src.inference.infer_from_metadata import (
    MetadataInferenceInput,
    _build_osu_filename_from_version,
    _infer_beatmap_id_value,
    _infer_difficulty_value,
    _sanitize_filename_component,
    _write_json,
    estimate_density_nps,
    extract_primary_timing,
    load_generator_from_checkpoint,
    mark_generated_metadata,
    notes_baseline_from_reference,
    song_output_to_notes_json,
    validate_constant_bpm_timing_json,
)
from src.preprocessing.osutaiko_reconstructor import reconstruct_osu


DEFAULT_SLIDER_MULTIPLIER = 1.4
DEFAULT_SLIDER_TICK_RATE = 1.0
DEFAULT_OUTPUT_ARTIFACT_KINDS = [
    "notes_json",
    "timing_json",
    "metadata_json",
    "song_output_json",
    "osu",
    "osz",
]
DEFAULT_INPUT_FIELDS = [
    {"id": "audio_file", "label": "Audio File", "kind": "file", "required": True, "advanced": False},
    {"id": "title", "label": "Title", "kind": "text", "required": True, "advanced": False},
    {"id": "artist", "label": "Artist", "kind": "text", "required": True, "advanced": False},
    {"id": "version", "label": "Difficulty Name", "kind": "text", "required": True, "advanced": False},
    {"id": "bpm", "label": "BPM", "kind": "number", "required": True, "advanced": False},
    {"id": "offset_ms", "label": "Offset (ms)", "kind": "number", "required": True, "advanced": False},
    {"id": "creator", "label": "Creator", "kind": "text", "required": False, "advanced": True},
    {"id": "meter", "label": "Meter", "kind": "number", "required": False, "advanced": True},
    {"id": "overall_difficulty", "label": "Overall Difficulty", "kind": "number", "required": False, "advanced": True},
    {"id": "density_nps", "label": "Density NPS", "kind": "number", "required": False, "advanced": True},
    {"id": "source", "label": "Source", "kind": "text", "required": False, "advanced": True},
    {"id": "tags", "label": "Tags", "kind": "text", "required": False, "advanced": True},
]


@dataclass(frozen=True)
class ModelDescriptor:
    id: str
    label: str
    checkpoint_path: Path
    architecture_name: str
    default_sampling: dict[str, Any]
    input_fields: list[dict[str, Any]] = field(default_factory=list)
    output_artifact_kinds: list[str] = field(default_factory=list)
    enabled: bool = True

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "architecture_name": self.architecture_name,
            "default_sampling": dict(self.default_sampling),
            "input_fields": [dict(field_payload) for field_payload in self.input_fields],
            "output_artifact_kinds": list(self.output_artifact_kinds),
            "enabled": bool(self.enabled),
        }


@dataclass(frozen=True)
class GenerationMetadataInput:
    title: str
    artist: str
    version: str
    creator: str = "taiko-diffusion"
    source: str = ""
    tags: str = ""
    title_unicode: str | None = None
    artist_unicode: str | None = None
    audio_filename: str | None = None
    beatmap_id: int = 0
    beatmap_set_id: int = -1
    overall_difficulty: float | str | None = None
    slider_multiplier: float = DEFAULT_SLIDER_MULTIPLIER
    slider_tick_rate: float = DEFAULT_SLIDER_TICK_RATE


@dataclass(frozen=True)
class GenerationTimingInput:
    bpm: float
    offset_ms: float
    meter: int = 4


@dataclass(frozen=True)
class GenerationConditioningInput:
    difficulty_value: float | None = None
    density_nps: float | None = 6.0
    beatmap_id: int | None = None


@dataclass(frozen=True)
class GenerationRequest:
    model_id: str
    audio_path: Path
    audio_filename: str
    metadata: GenerationMetadataInput
    timing: GenerationTimingInput
    conditioning: GenerationConditioningInput = field(default_factory=GenerationConditioningInput)
    sampling_override: dict[str, Any] | None = None
    chart_stem: str | None = None


@dataclass(frozen=True)
class ArtifactDescriptor:
    id: str
    kind: str
    label: str
    relative_path: str
    media_type: str
    primary_download: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationResult:
    model: ModelDescriptor
    chart_stem: str
    output_dir: Path
    artifacts: list[ArtifactDescriptor]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_sampling_payload() -> dict[str, Any]:
    return {
        "max_decode_len": 64,
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": 8,
        "ts_top_k": 16,
        "min_event_candidates": 2,
        "repetition_penalty": 1.0,
        "audio_cache_size": 8,
        "device": None,
    }


def _shared_input_fields() -> list[dict[str, Any]]:
    return [dict(field_payload) for field_payload in DEFAULT_INPUT_FIELDS]


def _shared_output_artifact_kinds() -> list[str]:
    return list(DEFAULT_OUTPUT_ARTIFACT_KINDS)


def _coerce_model_descriptor(
    payload: dict[str, Any],
    *,
    repo_root: Path,
    default_enabled: bool | None = None,
) -> ModelDescriptor:
    model_id = str(payload["id"]).strip()
    label = str(payload.get("label") or model_id).strip()
    architecture_name = str(payload["architecture_name"]).strip()
    checkpoint_raw = str(payload["checkpoint_path"]).strip()
    if not model_id:
        raise ValueError("Model id cannot be empty.")
    if not architecture_name:
        raise ValueError(f"Model '{model_id}' is missing architecture_name.")
    if not checkpoint_raw:
        raise ValueError(f"Model '{model_id}' is missing checkpoint_path.")

    checkpoint_path = Path(checkpoint_raw)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (repo_root / checkpoint_path).resolve()
    else:
        checkpoint_path = checkpoint_path.resolve()

    default_sampling = dict(_default_sampling_payload())
    default_sampling.update(dict(payload.get("default_sampling", {}) or {}))
    input_fields = payload.get("input_fields")
    output_artifact_kinds = payload.get("output_artifact_kinds")
    enabled = checkpoint_path.exists() if default_enabled is None else bool(default_enabled and checkpoint_path.exists())

    return ModelDescriptor(
        id=model_id,
        label=label,
        checkpoint_path=checkpoint_path,
        architecture_name=architecture_name,
        default_sampling=default_sampling,
        input_fields=[dict(field_payload) for field_payload in (input_fields or _shared_input_fields())],
        output_artifact_kinds=list(output_artifact_kinds or _shared_output_artifact_kinds()),
        enabled=enabled,
    )


def built_in_model_registry(repo_root: str | Path | None = None) -> dict[str, ModelDescriptor]:
    root = Path(repo_root or _repo_root()).resolve()
    models = [
        {
            "id": "sample_large_context",
            "label": "Sample Large Context",
            "checkpoint_path": "checkpoints/sample_large_context/last.ckpt",
            "architecture_name": "taiko_context_transformer",
        },
        {
            "id": "sample_large_baseline",
            "label": "Sample Large Baseline",
            "checkpoint_path": "checkpoints/sample_large_baseline/last.ckpt",
            "architecture_name": "taiko_transformer",
        },
        {
            "id": "sample_large_baseline_maxopt",
            "label": "Sample Large Baseline Maxopt",
            "checkpoint_path": "checkpoints/sample_large_baseline_maxopt/last.ckpt",
            "architecture_name": "taiko_transformer",
        },
    ]
    return {payload["id"]: _coerce_model_descriptor(payload, repo_root=root) for payload in models}


def load_model_registry(path: str | Path, repo_root: str | Path | None = None) -> dict[str, ModelDescriptor]:
    root = Path(repo_root or _repo_root()).resolve()
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = (root / manifest_path).resolve()
    else:
        manifest_path = manifest_path.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Model registry manifest not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        models_payload = payload.get("models")
        if models_payload is None:
            raise ValueError(f"Model registry manifest '{manifest_path}' must contain a 'models' list.")
    elif isinstance(payload, list):
        models_payload = payload
    else:
        raise ValueError(f"Model registry manifest '{manifest_path}' must decode to a JSON object or list.")

    if not isinstance(models_payload, list):
        raise ValueError(f"Model registry manifest '{manifest_path}' must contain a list of models.")

    registry: dict[str, ModelDescriptor] = {}
    for item in models_payload:
        if not isinstance(item, dict):
            raise ValueError(f"Model registry manifest '{manifest_path}' contains a non-object model entry.")
        model = _coerce_model_descriptor(item, repo_root=root)
        if model.id in registry:
            raise ValueError(f"Duplicate model id '{model.id}' in manifest '{manifest_path}'.")
        registry[model.id] = model
    return registry


def default_model_registry(repo_root: str | Path | None = None) -> dict[str, ModelDescriptor]:
    return built_in_model_registry(repo_root=repo_root)


def build_chart_stem(metadata: GenerationMetadataInput) -> str:
    artist = _sanitize_filename_component(metadata.artist or "Unknown Artist") or "Unknown Artist"
    title = _sanitize_filename_component(metadata.title or "Untitled") or "Untitled"
    creator = _sanitize_filename_component(metadata.creator or "taiko-diffusion") or "taiko-diffusion"
    version = _sanitize_filename_component(metadata.version or "Generated") or "Generated"
    return f"{artist} - {title} ({creator}) [{version}]"


def build_metadata_json(metadata: GenerationMetadataInput, *, audio_filename: str, chart_stem: str) -> dict[str, Any]:
    title_unicode = metadata.title_unicode or metadata.title
    artist_unicode = metadata.artist_unicode or metadata.artist
    overall_difficulty = metadata.overall_difficulty
    if overall_difficulty is None or str(overall_difficulty).strip() == "":
        overall_difficulty = _infer_difficulty_value(metadata.version)

    source_osu = f"{chart_stem}.osu"
    return {
        "format": 2,
        "source_osu": source_osu,
        "general": {
            "AudioFilename": audio_filename,
            "AudioLeadIn": "0",
            "PreviewTime": "-1",
            "Countdown": "0",
            "SampleSet": "Normal",
            "StackLeniency": "0.7",
            "Mode": "1",
            "LetterboxInBreaks": "0",
            "WidescreenStoryboard": "0",
        },
        "metadata": {
            "Title": metadata.title,
            "TitleUnicode": title_unicode,
            "Artist": metadata.artist,
            "ArtistUnicode": artist_unicode,
            "Creator": metadata.creator,
            "Version": metadata.version,
            "Source": metadata.source,
            "Tags": metadata.tags,
            "BeatmapID": str(int(metadata.beatmap_id)),
            "BeatmapSetID": str(int(metadata.beatmap_set_id)),
        },
        "difficulty": {
            "HPDrainRate": "5",
            "CircleSize": "5",
            "OverallDifficulty": str(overall_difficulty),
            "ApproachRate": "5",
            "SliderMultiplier": str(float(metadata.slider_multiplier)),
            "SliderTickRate": str(float(metadata.slider_tick_rate)),
        },
    }


def build_timing_json(
    timing: GenerationTimingInput,
    *,
    chart_stem: str,
    slider_multiplier: float = DEFAULT_SLIDER_MULTIPLIER,
    slider_tick_rate: float = DEFAULT_SLIDER_TICK_RATE,
) -> dict[str, Any]:
    bpm = float(timing.bpm)
    if bpm <= 0:
        raise ValueError("BPM must be positive.")
    meter = max(1, int(timing.meter))
    offset_ms = float(timing.offset_ms)
    ms_per_beat = 60000.0 / bpm
    return {
        "format": 2,
        "source_osu": f"{chart_stem}.osu",
        "slider_multiplier": float(slider_multiplier),
        "slider_tick_rate": float(slider_tick_rate),
        "timing_points": [
            {
                "offset": offset_ms,
                "raw_offset": int(round(offset_ms)),
                "ms_per_beat": ms_per_beat,
                "meter": meter,
                "sample_set": 1,
                "sample_index": 0,
                "volume": 100,
                "uninherited": 1,
                "effects": 0,
            }
        ],
    }


def package_osz(audio_file: str | Path, osu_file: str | Path, out_path: str | Path) -> Path:
    audio_path = Path(audio_file).resolve()
    osu_path = Path(osu_file).resolve()
    out_path = Path(out_path).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not osu_path.exists():
        raise FileNotFoundError(f"OSU file not found: {osu_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(out_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(audio_path, arcname=audio_path.name)
        archive.write(osu_path, arcname=osu_path.name)
    return out_path


def _build_model_output_slug(model: ModelDescriptor) -> str:
    parts = [
        model.id,
        model.checkpoint_path.parent.name,
        model.checkpoint_path.stem,
        model.architecture_name,
    ]
    return _sanitize_filename_component("_".join(str(part) for part in parts if str(part).strip()))


def _build_sampling_config(model: ModelDescriptor, overrides: dict[str, Any] | None):
    from src.model.generation import SamplingConfig

    payload = dict(model.default_sampling)
    payload.update(dict(overrides or {}))
    return SamplingConfig(
        temperature=float(payload.get("temperature", 1.0)),
        top_p=float(payload.get("top_p", 0.9)),
        top_k=max(1, int(payload.get("top_k", 8))),
        ts_top_k=max(0, int(payload.get("ts_top_k", 16))),
        min_event_candidates=max(1, int(payload.get("min_event_candidates", 2))),
        repetition_penalty=float(payload.get("repetition_penalty", 1.0)),
    )


class GenerationService:
    def __init__(self, model_registry: dict[str, ModelDescriptor] | None = None):
        self.model_registry = dict(model_registry or default_model_registry())

    def list_models(self) -> list[ModelDescriptor]:
        return [self.model_registry[key] for key in sorted(self.model_registry)]

    def get_model(self, model_id: str) -> ModelDescriptor:
        model = self.model_registry.get(str(model_id).strip())
        if model is None:
            available = ", ".join(sorted(self.model_registry)) or "<none>"
            raise KeyError(f"Unknown model_id '{model_id}'. Available: {available}")
        if not model.enabled:
            raise FileNotFoundError(f"Checkpoint is not available for model '{model_id}': {model.checkpoint_path}")
        return model

    def generate(self, request: GenerationRequest, *, output_root: str | Path) -> GenerationResult:
        model = self.get_model(request.model_id)
        output_root = Path(output_root).resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        chart_stem = request.chart_stem or build_chart_stem(request.metadata)
        metadata_json = build_metadata_json(
            request.metadata,
            audio_filename=request.audio_filename,
            chart_stem=chart_stem,
        )
        timing_json = build_timing_json(
            request.timing,
            chart_stem=chart_stem,
            slider_multiplier=request.metadata.slider_multiplier,
            slider_tick_rate=request.metadata.slider_tick_rate,
        )
        validate_constant_bpm_timing_json(timing_json)

        offset_ms, bpm, meter = extract_primary_timing(timing_json)
        difficulty_value = (
            float(request.conditioning.difficulty_value)
            if request.conditioning.difficulty_value is not None
            else float(
                _infer_difficulty_value(
                    metadata_json.get("difficulty", {}).get("OverallDifficulty", metadata_json.get("metadata", {}).get("Version", ""))
                )
            )
        )
        beatmap_id = (
            int(request.conditioning.beatmap_id)
            if request.conditioning.beatmap_id is not None
            else int(
                _infer_beatmap_id_value(
                    chart_stem,
                    metadata_json.get("metadata", {}).get("BeatmapID", ""),
                )
            )
        )
        density_nps = (
            float(request.conditioning.density_nps)
            if request.conditioning.density_nps is not None
            else float(estimate_density_nps({"notes": []}, offset_ms=offset_ms))
        )

        chart_input = MetadataInferenceInput(
            chart_stem=chart_stem,
            audio_path=Path(request.audio_path).resolve(),
            metadata_json=metadata_json,
            timing_json=timing_json,
            offset_ms=offset_ms,
            bpm=bpm,
            meter=meter,
            difficulty_value=difficulty_value,
            beatmap_id=beatmap_id,
            density_nps=density_nps,
            reference_notes_json=None,
        )

        sampling_payload = dict(model.default_sampling)
        sampling_payload.update(dict(request.sampling_override or {}))
        max_decode_len = max(1, int(sampling_payload.get("max_decode_len", model.default_sampling.get("max_decode_len", 64))))
        audio_cache_size = max(1, int(sampling_payload.get("audio_cache_size", model.default_sampling.get("audio_cache_size", 8))))
        device = sampling_payload.get("device", model.default_sampling.get("device"))
        sampling_config = _build_sampling_config(model, request.sampling_override)
        output_slug = _build_model_output_slug(model)
        chart_output_dir = output_root / model.id / chart_stem
        chart_output_dir.mkdir(parents=True, exist_ok=True)

        generator, architecture_spec = load_generator_from_checkpoint(
            model.checkpoint_path,
            device=device,
            max_decode_len=max_decode_len,
            audio_cache_size=audio_cache_size,
        )
        song_output = generator.generate_song_structure(
            audio_path=chart_input.audio_path,
            offset_ms=chart_input.offset_ms,
            bpm=chart_input.bpm,
            meter=chart_input.meter,
            difficulty=chart_input.difficulty_value,
            density_nps=chart_input.density_nps,
            beatmap_id=chart_input.beatmap_id,
            sampling_config=sampling_config,
        )

        sv_default, vol_default = notes_baseline_from_reference(chart_input.reference_notes_json)
        generated_notes = song_output_to_notes_json(
            song_output,
            source_osu=str(chart_input.metadata_json.get("source_osu", f"{chart_input.chart_stem}.osu")),
            offset_ms=chart_input.offset_ms,
            bpm=chart_input.bpm,
            meter=chart_input.meter,
            sv_default=sv_default,
            volume_default=vol_default,
        )
        generated_metadata = mark_generated_metadata(
            chart_input.metadata_json,
            architecture_name=str(architecture_spec.name),
            checkpoint_name=model.id,
        )
        generated_timing = copy.deepcopy(chart_input.timing_json)

        notes_path = chart_output_dir / f"{chart_input.chart_stem}.{output_slug}.generated.notes.json"
        timing_path = chart_output_dir / f"{chart_input.chart_stem}.{output_slug}.generated.timing.json"
        metadata_path = chart_output_dir / f"{chart_input.chart_stem}.{output_slug}.generated.metadata.json"
        song_output_path = chart_output_dir / f"{chart_input.chart_stem}.{output_slug}.song_output.json"
        osu_path = chart_output_dir / f"{_build_osu_filename_from_version(chart_input.chart_stem, str(generated_metadata.get('metadata', {}).get('Version', 'Generated')))[:-4]}.{output_slug}.osu"
        osz_path = chart_output_dir / f"{chart_input.chart_stem}.{output_slug}.generated.osz"

        _write_json(notes_path, generated_notes)
        _write_json(timing_path, generated_timing)
        _write_json(metadata_path, generated_metadata)
        _write_json(song_output_path, {"song_output": song_output})
        reconstruct_osu(
            notes_path=notes_path,
            out_path=osu_path,
            timing_path=timing_path,
            metadata_path=metadata_path,
        )
        package_osz(chart_input.audio_path, osu_path, osz_path)

        artifacts = [
            ArtifactDescriptor(
                id="notes_json",
                kind="notes_json",
                label="Generated Notes JSON",
                relative_path=str(notes_path.relative_to(output_root)),
                media_type="application/json",
            ),
            ArtifactDescriptor(
                id="timing_json",
                kind="timing_json",
                label="Generated Timing JSON",
                relative_path=str(timing_path.relative_to(output_root)),
                media_type="application/json",
            ),
            ArtifactDescriptor(
                id="metadata_json",
                kind="metadata_json",
                label="Generated Metadata JSON",
                relative_path=str(metadata_path.relative_to(output_root)),
                media_type="application/json",
            ),
            ArtifactDescriptor(
                id="song_output_json",
                kind="song_output_json",
                label="Raw Song Output JSON",
                relative_path=str(song_output_path.relative_to(output_root)),
                media_type="application/json",
            ),
            ArtifactDescriptor(
                id="generated_osu",
                kind="osu",
                label="Generated OSU",
                relative_path=str(osu_path.relative_to(output_root)),
                media_type="text/plain",
            ),
            ArtifactDescriptor(
                id="generated_osz",
                kind="osz",
                label="Generated OSZ",
                relative_path=str(osz_path.relative_to(output_root)),
                media_type="application/octet-stream",
                primary_download=True,
            ),
        ]
        return GenerationResult(
            model=model,
            chart_stem=chart_stem,
            output_dir=chart_output_dir,
            artifacts=artifacts,
        )


def generation_request_from_payload(payload: dict[str, Any], *, audio_path: str | Path) -> GenerationRequest:
    metadata_payload = dict(payload.get("metadata", {}))
    timing_payload = dict(payload.get("timing", {}))
    conditioning_payload = dict(payload.get("conditioning", {}))
    return GenerationRequest(
        model_id=str(payload["model_id"]),
        audio_path=Path(audio_path).resolve(),
        audio_filename=str(payload.get("audio_filename") or Path(audio_path).name),
        metadata=GenerationMetadataInput(**metadata_payload),
        timing=GenerationTimingInput(**timing_payload),
        conditioning=GenerationConditioningInput(**conditioning_payload),
        sampling_override=dict(payload.get("sampling_override", {}) or {}),
        chart_stem=payload.get("chart_stem"),
    )


def generation_request_to_payload(request: GenerationRequest) -> dict[str, Any]:
    return {
        "model_id": request.model_id,
        "audio_filename": request.audio_filename,
        "chart_stem": request.chart_stem,
        "metadata": asdict(request.metadata),
        "timing": asdict(request.timing),
        "conditioning": asdict(request.conditioning),
        "sampling_override": dict(request.sampling_override or {}),
    }


def generation_result_to_payload(result: GenerationResult) -> dict[str, Any]:
    return {
        "model": result.model.to_public_dict(),
        "chart_stem": result.chart_stem,
        "output_dir": str(result.output_dir),
        "artifacts": [artifact.to_dict() for artifact in result.artifacts],
    }
