from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from src.inference.infer_from_metadata import (
    MetadataInferenceInput,
    _default_device,
    _build_osu_filename_from_version,
    _infer_beatmap_id_value,
    _infer_difficulty_value,
    _sanitize_filename_component,
    _write_json,
    estimate_density_nps,
    extract_primary_timing,
    load_generator_from_checkpoint,
    load_inference_model_from_checkpoint,
    mark_generated_metadata,
    notes_baseline_from_reference,
    song_output_to_notes_json,
    validate_constant_bpm_timing_json,
)
from src.preprocessing.osutaiko_reconstructor import reconstruct_osu


DEFAULT_SLIDER_MULTIPLIER = 1.4
DEFAULT_SLIDER_TICK_RATE = 1.0
FRAMES_PER_BEAT = 48
BASELINE_WINDOW_BEATS = 4
REFINER_BLOCK_BEATS = 32
FRAMES_PER_WINDOW = FRAMES_PER_BEAT * BASELINE_WINDOW_BEATS
FRAMES_PER_REFINER_BLOCK = FRAMES_PER_BEAT * REFINER_BLOCK_BEATS
WINDOWS_PER_REFINER_BLOCK = REFINER_BLOCK_BEATS // BASELINE_WINDOW_BEATS
MODEL_INFERENCE_KINDS = {"autoregressive", "hybrid_refine"}
SPECIAL_REFINER_TOKENS = {"PAD", "BOS", "EOS", "MASK"}
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
    inference_kind: str = "autoregressive"
    bootstrap_model_id: str | None = None
    input_fields: list[dict[str, Any]] = field(default_factory=list)
    output_artifact_kinds: list[str] = field(default_factory=list)
    enabled: bool = True

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "architecture_name": self.architecture_name,
            "default_sampling": dict(self.default_sampling),
            "inference_kind": self.inference_kind,
            "bootstrap_model_id": self.bootstrap_model_id,
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
        "mask_ratio": 0.25,
        "audio_cache_size": 8,
        "device": None,
    }


def _shared_input_fields() -> list[dict[str, Any]]:
    return [dict(field_payload) for field_payload in DEFAULT_INPUT_FIELDS]


def _shared_output_artifact_kinds() -> list[str]:
    return list(DEFAULT_OUTPUT_ARTIFACT_KINDS)


def _finalize_model_registry(registry: dict[str, ModelDescriptor]) -> dict[str, ModelDescriptor]:
    finalized: dict[str, ModelDescriptor] = {}
    for model_id, model in registry.items():
        enabled = bool(model.checkpoint_path.exists())
        if model.inference_kind == "hybrid_refine":
            bootstrap_model_id = str(model.bootstrap_model_id or "").strip()
            if not bootstrap_model_id:
                raise ValueError(f"Hybrid model '{model_id}' is missing bootstrap_model_id.")
            if bootstrap_model_id == model_id:
                raise ValueError(f"Hybrid model '{model_id}' cannot use itself as bootstrap_model_id.")
            bootstrap_model = registry.get(bootstrap_model_id)
            if bootstrap_model is None:
                raise ValueError(
                    f"Hybrid model '{model_id}' references unknown bootstrap_model_id '{bootstrap_model_id}'."
                )
            if bootstrap_model.inference_kind != "autoregressive":
                raise ValueError(
                    f"Hybrid model '{model_id}' must reference an autoregressive bootstrap model, "
                    f"got '{bootstrap_model.inference_kind}' for '{bootstrap_model_id}'."
                )
            enabled = enabled and bool(bootstrap_model.checkpoint_path.exists())
        finalized[model_id] = replace(model, enabled=enabled)
    return finalized


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
    inference_kind = str(payload.get("inference_kind") or "autoregressive").strip().lower()
    if inference_kind not in MODEL_INFERENCE_KINDS:
        raise ValueError(
            f"Model '{model_id}' has unsupported inference_kind '{inference_kind}'. "
            f"Allowed: {sorted(MODEL_INFERENCE_KINDS)}"
        )
    bootstrap_model_id = payload.get("bootstrap_model_id")
    if inference_kind == "hybrid_refine" and not str(bootstrap_model_id or "").strip():
        raise ValueError(f"Hybrid model '{model_id}' must define bootstrap_model_id.")

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
        inference_kind=inference_kind,
        bootstrap_model_id=None if bootstrap_model_id is None else str(bootstrap_model_id).strip(),
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
    registry = {payload["id"]: _coerce_model_descriptor(payload, repo_root=root) for payload in models}
    return _finalize_model_registry(registry)


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
    return _finalize_model_registry(registry)


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


def _resolve_inference_device(device: Any):
    if device is None or str(device).strip() == "":
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised through runtime checks
            raise RuntimeError("PyTorch is required for checkpoint-based generation.") from exc
        return torch.device(_default_device())

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised through runtime checks
        raise RuntimeError("PyTorch is required for checkpoint-based generation.") from exc
    return torch.device(device)


def _build_refiner_sampling_payload(model: ModelDescriptor, overrides: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(model.default_sampling)
    payload.update(dict(overrides or {}))
    payload["mask_ratio"] = float(max(0.0, min(1.0, payload.get("mask_ratio", 0.25))))
    payload["temperature"] = float(payload.get("temperature", 1.1))
    return payload


def _song_events_from_tokens(tokens: list[str], *, start_frame: int) -> list[tuple[int, str]]:
    cursor_frame = 0
    events: list[tuple[int, str]] = []
    for token in list(tokens):
        token_text = str(token).strip().upper()
        if not token_text:
            continue
        if token_text.startswith("TS_"):
            try:
                cursor_frame = max(0, cursor_frame + int(token_text[3:]))
            except ValueError:
                continue
            continue
        events.append((int(start_frame) + cursor_frame, token_text))
    return events


def _tokens_from_song_events(events: list[tuple[int, str]], *, start_frame: int) -> list[str]:
    if not events:
        return []

    ordered = sorted((int(frame), str(token).strip().upper()) for frame, token in events)
    tokens: list[str] = []
    prev_local_frame = 0
    for frame, token in ordered:
        local_frame = max(0, int(frame) - int(start_frame))
        delta = local_frame - prev_local_frame
        if delta > 0:
            tokens.append(f"TS_{delta}")
        tokens.append(token)
        prev_local_frame = local_frame
    return tokens


def convert_song_output_to_refiner_blocks(
    song_output: list[dict[str, Any]],
    *,
    windows_per_block: int = WINDOWS_PER_REFINER_BLOCK,
) -> list[dict[str, Any]]:
    ordered = sorted(song_output, key=lambda item: int(item.get("seq_idx", 0)))
    if windows_per_block <= 0:
        raise ValueError("windows_per_block must be positive.")

    blocks: list[dict[str, Any]] = []
    full_block_count = len(ordered) // windows_per_block
    for block_idx in range(full_block_count):
        chunk = ordered[block_idx * windows_per_block:(block_idx + 1) * windows_per_block]
        start_frame = int(chunk[0]["start_frame"])
        end_frame = start_frame + windows_per_block * FRAMES_PER_WINDOW - 1
        events: list[tuple[int, str]] = []
        for item in chunk:
            events.extend(_song_events_from_tokens(list(item.get("pred_tokens", [])), start_frame=int(item["start_frame"])))
        blocks.append(
            {
                "block_idx": block_idx,
                "seq_start_idx": int(chunk[0]["seq_idx"]),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "pred_tokens": _tokens_from_song_events(events, start_frame=start_frame),
            }
        )
    return blocks


def apply_refined_blocks_to_song_output(
    song_output: list[dict[str, Any]],
    refined_blocks: list[dict[str, Any]],
    *,
    windows_per_block: int = WINDOWS_PER_REFINER_BLOCK,
) -> list[dict[str, Any]]:
    updated_items = [dict(item) for item in sorted(song_output, key=lambda item: int(item.get("seq_idx", 0)))]
    by_seq_idx = {int(item["seq_idx"]): item for item in updated_items}

    for block in refined_blocks:
        block_events = _song_events_from_tokens(list(block.get("pred_tokens", [])), start_frame=int(block["start_frame"]))
        seq_start_idx = int(block.get("seq_start_idx", int(block["start_frame"]) // FRAMES_PER_WINDOW))
        for offset in range(windows_per_block):
            seq_idx = seq_start_idx + offset
            target = by_seq_idx.get(seq_idx)
            if target is None:
                break
            seq_start_frame = int(target["start_frame"])
            seq_end_frame = int(target["end_frame"])
            seq_events = [
                (frame, token)
                for frame, token in block_events
                if seq_start_frame <= int(frame) <= seq_end_frame
            ]
            target["pred_tokens"] = _tokens_from_song_events(seq_events, start_frame=seq_start_frame)
    return [by_seq_idx[int(item["seq_idx"])] for item in sorted(updated_items, key=lambda item: int(item["seq_idx"]))]


def mask_note_tokens_for_refinement(
    input_ids: list[int],
    token_to_id: dict[str, int],
    *,
    mask_ratio: float,
    generator=None,
) -> tuple[list[int], list[int]]:
    import random

    if "MASK" not in token_to_id:
        raise ValueError("Refiner vocabulary must include MASK.")

    mask_id = int(token_to_id["MASK"])
    special_ids = {
        int(token_to_id[token])
        for token in SPECIAL_REFINER_TOKENS
        if token in token_to_id
    }
    ts_ids = {
        int(token_id)
        for token, token_id in token_to_id.items()
        if str(token).startswith("TS_")
    }

    note_positions = [
        idx
        for idx, token_id in enumerate(list(input_ids))
        if int(token_id) not in special_ids and int(token_id) not in ts_ids
    ]
    if not note_positions or mask_ratio <= 0.0:
        return list(input_ids), []

    if mask_ratio >= 1.0:
        chosen_positions = list(note_positions)
    else:
        num_to_mask = max(1, int(round(len(note_positions) * float(mask_ratio))))
        if generator is not None:
            try:
                import torch
            except ImportError:
                generator = None
            else:
                perm = torch.randperm(len(note_positions), generator=generator)
                chosen_positions = [note_positions[int(idx)] for idx in perm[:num_to_mask].tolist()]
        if generator is None:
            rng = random.Random(42)
            chosen_positions = rng.sample(note_positions, k=min(num_to_mask, len(note_positions)))

    masked_ids = list(int(token_id) for token_id in input_ids)
    for pos in chosen_positions:
        masked_ids[int(pos)] = mask_id
    return masked_ids, sorted(int(pos) for pos in chosen_positions)


def _sample_token_from_subset(logits_1d, candidate_ids, *, temperature: float) -> int:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised through runtime checks
        raise RuntimeError("PyTorch is required for checkpoint-based generation.") from exc

    if not candidate_ids:
        raise ValueError("candidate_ids must not be empty.")

    candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=logits_1d.device)
    candidate_logits = logits_1d.index_select(0, candidate_tensor)
    if temperature <= 0.0:
        return int(candidate_tensor[int(torch.argmax(candidate_logits).item())].item())

    scaled_logits = candidate_logits / max(1e-6, float(temperature))
    probs = torch.softmax(scaled_logits, dim=0)
    sampled_idx = torch.multinomial(probs, num_samples=1)
    return int(candidate_tensor[int(sampled_idx.item())].item())


def _build_aligned_audio_blocks(
    audio_path: str | Path,
    *,
    offset_ms: float,
    bpm: float,
    meter: int,
    frames_per_block: int = FRAMES_PER_REFINER_BLOCK,
) -> list[Any]:
    from src.preprocessing.beat_aligned_dataset import (
        build_beat_aligned_frame_timeline,
        build_raw_mel_spectrogram,
        compute_beat_grid_info,
        get_audio_info,
        interpolate_raw_mel_to_beat_aligned_timeline,
    )

    beat_duration_ms = 60000.0 / max(float(bpm), 1e-6)
    audio_info = get_audio_info(Path(audio_path))
    beat_grid_info, _ = compute_beat_grid_info(
        offset_ms=float(offset_ms),
        beat_duration_ms=float(beat_duration_ms),
        audio_duration_ms=float(audio_info["audio_duration_ms"]),
    )
    total_blocks = int(beat_grid_info.total_frames // frames_per_block)
    if total_blocks <= 0:
        return []

    frame_times_ms = build_beat_aligned_frame_timeline(
        offset_ms=float(offset_ms),
        beat_duration_ms=float(beat_duration_ms),
        total_frames=beat_grid_info.total_frames,
    )
    mel_spec_db, orig_frame_times_ms = build_raw_mel_spectrogram(
        waveform=audio_info["waveform"],
        sample_rate=audio_info["sample_rate"],
    )
    aligned_mel_db = interpolate_raw_mel_to_beat_aligned_timeline(
        mel_spec_db=mel_spec_db,
        orig_frame_times_ms=orig_frame_times_ms,
        beat_aligned_frame_times_ms=frame_times_ms,
    )

    return [
        aligned_mel_db[block_idx * frames_per_block:(block_idx + 1) * frames_per_block]
        for block_idx in range(total_blocks)
    ]


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

    def _build_chart_input(self, request: GenerationRequest) -> MetadataInferenceInput:
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

        return MetadataInferenceInput(
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

    def _generate_song_output_autoregressive(
        self,
        model: ModelDescriptor,
        *,
        chart_input: MetadataInferenceInput,
        sampling_override: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], Any]:
        sampling_payload = dict(model.default_sampling)
        sampling_payload.update(dict(sampling_override or {}))
        max_decode_len = max(1, int(sampling_payload.get("max_decode_len", model.default_sampling.get("max_decode_len", 64))))
        audio_cache_size = max(1, int(sampling_payload.get("audio_cache_size", model.default_sampling.get("audio_cache_size", 8))))
        device = sampling_payload.get("device", model.default_sampling.get("device"))
        sampling_config = _build_sampling_config(model, sampling_override)

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
        return song_output, architecture_spec

    def _refine_song_output_with_diffusion_model(
        self,
        model: ModelDescriptor,
        *,
        chart_input: MetadataInferenceInput,
        song_output: list[dict[str, Any]],
        sampling_override: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], Any]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised through runtime checks
            raise RuntimeError("PyTorch is required for checkpoint-based generation.") from exc

        device_payload = _build_refiner_sampling_payload(model, sampling_override)
        resolved_device = _resolve_inference_device(device_payload.get("device", model.default_sampling.get("device")))
        model_instance, architecture_spec, token_to_id, id_to_token = load_inference_model_from_checkpoint(
            model.checkpoint_path,
            device=resolved_device,
        )
        if str(architecture_spec.name).strip() != "taiko_diffusion_refiner":
            raise ValueError(
                f"Hybrid refiner model '{model.id}' expected architecture 'taiko_diffusion_refiner', "
                f"got '{architecture_spec.name}'."
            )
        if "MASK" not in token_to_id:
            raise ValueError(f"Hybrid refiner model '{model.id}' vocabulary is missing MASK.")
        if "BOS" not in token_to_id:
            raise ValueError(f"Hybrid refiner model '{model.id}' vocabulary is missing BOS.")

        note_token_ids = [
            int(token_id)
            for token, token_id in token_to_id.items()
            if str(token) not in SPECIAL_REFINER_TOKENS and not str(token).startswith("TS_")
        ]
        if not note_token_ids:
            raise ValueError(f"Hybrid refiner model '{model.id}' vocabulary does not contain any note tokens.")

        audio_blocks = _build_aligned_audio_blocks(
            chart_input.audio_path,
            offset_ms=chart_input.offset_ms,
            bpm=chart_input.bpm,
            meter=chart_input.meter,
        )
        refiner_blocks = convert_song_output_to_refiner_blocks(song_output)
        if not audio_blocks or not refiner_blocks:
            return song_output, architecture_spec

        generator = torch.Generator()
        generator.manual_seed(42)
        refined_blocks: list[dict[str, Any]] = []
        full_block_count = min(len(audio_blocks), len(refiner_blocks))

        with torch.no_grad():
            for block_idx in range(full_block_count):
                block = dict(refiner_blocks[block_idx])
                raw_tokens = list(block.get("pred_tokens", []))
                input_ids = [int(token_to_id["BOS"])] + [int(token_to_id[token]) for token in raw_tokens]
                masked_ids, mask_positions = mask_note_tokens_for_refinement(
                    input_ids,
                    token_to_id,
                    mask_ratio=float(device_payload["mask_ratio"]),
                    generator=generator,
                )
                if not mask_positions:
                    refined_blocks.append(block)
                    continue

                audio_tensor = torch.tensor(audio_blocks[block_idx], dtype=torch.float32, device=resolved_device).unsqueeze(0)
                input_tensor = torch.tensor(masked_ids, dtype=torch.long, device=resolved_device).unsqueeze(0)
                attention_mask = torch.ones_like(input_tensor, device=resolved_device)
                logits = model_instance(
                    audio_tensor,
                    input_tensor,
                    decoder_attention_mask=attention_mask,
                )

                refined_input_ids = list(masked_ids)
                for pos in mask_positions:
                    refined_input_ids[pos] = _sample_token_from_subset(
                        logits[0, pos, :],
                        note_token_ids,
                        temperature=float(device_payload["temperature"]),
                    )

                block["pred_tokens"] = [id_to_token[int(token_id)] for token_id in refined_input_ids[1:]]
                refined_blocks.append(block)

        refined_song_output = apply_refined_blocks_to_song_output(song_output, refined_blocks)
        return refined_song_output, architecture_spec

    def _generate_song_output_for_model(
        self,
        model: ModelDescriptor,
        *,
        chart_input: MetadataInferenceInput,
        sampling_override: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], Any]:
        if model.inference_kind == "autoregressive":
            return self._generate_song_output_autoregressive(
                model,
                chart_input=chart_input,
                sampling_override=sampling_override,
            )

        if model.inference_kind == "hybrid_refine":
            bootstrap_model = self.get_model(str(model.bootstrap_model_id))
            bootstrap_song_output, _ = self._generate_song_output_autoregressive(
                bootstrap_model,
                chart_input=chart_input,
                sampling_override=sampling_override,
            )
            return self._refine_song_output_with_diffusion_model(
                model,
                chart_input=chart_input,
                song_output=bootstrap_song_output,
                sampling_override=sampling_override,
            )

        raise ValueError(f"Unsupported inference_kind '{model.inference_kind}' for model '{model.id}'.")

    def generate(self, request: GenerationRequest, *, output_root: str | Path) -> GenerationResult:
        model = self.get_model(request.model_id)
        output_root = Path(output_root).resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        chart_input = self._build_chart_input(request)
        output_slug = _build_model_output_slug(model)
        chart_output_dir = output_root / model.id / chart_input.chart_stem
        chart_output_dir.mkdir(parents=True, exist_ok=True)

        song_output, architecture_spec = self._generate_song_output_for_model(
            model,
            chart_input=chart_input,
            sampling_override=request.sampling_override,
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
            chart_stem=chart_input.chart_stem,
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
