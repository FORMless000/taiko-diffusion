from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - handled by runtime check
    torch = None

from src.preprocessing.osutaiko_reconstructor import reconstruct_osu


@dataclass
class MetadataInferenceInput:
    chart_stem: str
    audio_path: Path
    metadata_json: dict[str, Any]
    timing_json: dict[str, Any]
    offset_ms: float
    bpm: float
    meter: int
    difficulty_value: float
    beatmap_id: int
    density_nps: float
    reference_notes_json: dict[str, Any] | None = None


_PLAYABLE_TOKEN_TO_TYPE = {
    "DON": "don",
    "KAT": "kat",
    "BIGDON": "bigdon",
    "BIGKAT": "bigkat",
    "DRUMROLL": "drumroll",
    "SLIDERSTART": "sliderstart",
    "SLIDEREND": "sliderend",
}


def _safe_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
        if parsed != parsed:
            return default
        return parsed
    except Exception:
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _default_device() -> str:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _infer_difficulty_value(raw_difficulty: Any) -> float:
    if raw_difficulty is None:
        return 5.0
    text = str(raw_difficulty).strip()
    if not text:
        return 5.0
    numeric = _safe_float(text, float("nan"))
    if numeric == numeric:
        return float(max(0.0, min(10.0, numeric)))

    lowered = text.lower()
    mapping = {
        "kantan": 2.0,
        "futsuu": 4.0,
        "muzukashii": 6.5,
        "oni": 8.5,
        "inner": 9.2,
        "ura": 9.5,
    }
    for key, value in mapping.items():
        if key in lowered:
            return value
    return 5.0


def _infer_beatmap_id_value(chart_id: str, explicit_beatmap_id: Any = None) -> int:
    explicit = _safe_int(explicit_beatmap_id, -1)
    if explicit > 0:
        return explicit
    try:
        return int(str(chart_id).split("_", 1)[0])
    except Exception:
        return 1_000_000


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_primary_timing(timing_json: dict[str, Any]) -> tuple[float, float, int]:
    timing_points = list(timing_json.get("timing_points", []))
    bpm_points = [
        tp for tp in timing_points
        if int(tp.get("uninherited", 0)) == 1 and _safe_float(tp.get("ms_per_beat", 0.0), 0.0) > 0.0
    ]
    if not bpm_points:
        raise ValueError("No uninherited BPM timing points found.")
    first = bpm_points[0]
    offset_ms = float(_safe_float(first.get("offset", 0.0), 0.0))
    ms_per_beat = float(_safe_float(first.get("ms_per_beat", 500.0), 500.0))
    bpm = 60000.0 / max(ms_per_beat, 1e-6)
    meter = max(1, int(_safe_int(first.get("meter", 4), 4)))
    return offset_ms, bpm, meter


def validate_constant_bpm_timing_json(timing_json: dict[str, Any]) -> None:
    timing_points = list(timing_json.get("timing_points", []))
    bpm_points = [
        tp for tp in timing_points
        if int(tp.get("uninherited", 0)) == 1 and _safe_float(tp.get("ms_per_beat", 0.0), 0.0) > 0.0
    ]
    if not bpm_points:
        raise ValueError("No uninherited BPM timing points found.")

    unique_mpb = {round(_safe_float(tp.get("ms_per_beat", 0.0), 0.0), 10) for tp in bpm_points}
    if len(unique_mpb) != 1:
        raise ValueError("Only constant-BPM timing payloads are supported for inference.")


def estimate_density_nps(notes_json: dict[str, Any], offset_ms: float) -> float:
    notes = [n for n in list(notes_json.get("notes", [])) if str(n.get("type", "")).lower() != "bpmchange"]
    if not notes:
        return 0.0
    end_ms = max(_safe_float(n.get("time", 0.0), 0.0) for n in notes)
    duration_sec = max((end_ms - offset_ms) / 1000.0, 1e-6)
    return float(len(notes) / duration_sec)


def notes_baseline_from_reference(notes_json: dict[str, Any] | None) -> tuple[float, int]:
    if notes_json is None:
        return 1.0, 100
    notes = list(notes_json.get("notes", []))
    for note in notes:
        if str(note.get("type", "")).lower() == "bpmchange":
            continue
        return (
            float(_safe_float(note.get("sv", 1.0), 1.0)),
            int(_safe_int(note.get("volume", 100), 100)),
        )
    return 1.0, 100


def song_output_to_notes_json(
    song_output: list[dict[str, Any]],
    *,
    source_osu: str,
    offset_ms: float,
    bpm: float,
    meter: int,
    sv_default: float,
    volume_default: int,
) -> dict[str, Any]:
    beat_duration_ms = 60000.0 / max(float(bpm), 1e-6)
    tick_ms = beat_duration_ms / 48.0
    seq_duration_ms = tick_ms * 192.0

    notes: list[dict[str, Any]] = [
        {
            "type": "bpmchange",
            "time": float(offset_ms),
            "raw_time": float(offset_ms),
            "sv": float(sv_default),
            "volume": int(volume_default),
            "bpm": float(bpm),
            "meter": int(meter),
        }
    ]

    for seq in song_output:
        seq_idx = int(_safe_int(seq.get("seq_idx", 0), 0))
        seq_start_ms = float(offset_ms) + seq_idx * seq_duration_ms
        cursor_tick = 0
        for token in list(seq.get("pred_tokens", [])):
            token_text = str(token).strip().upper()
            if token_text.startswith("TS_"):
                shift = _safe_int(token_text[3:], 0)
                cursor_tick = max(0, cursor_tick + shift)
                continue
            note_type = _PLAYABLE_TOKEN_TO_TYPE.get(token_text)
            if note_type is None:
                continue
            note_time_ms = seq_start_ms + cursor_tick * tick_ms
            notes.append(
                {
                    "type": note_type,
                    "time": float(note_time_ms),
                    "raw_time": float(note_time_ms),
                    "sv": float(sv_default),
                    "volume": int(volume_default),
                    "bpm": None,
                    "meter": None,
                }
            )

    type_priority = {
        "bpmchange": 0,
        "don": 1,
        "kat": 2,
        "bigdon": 3,
        "bigkat": 4,
        "sliderstart": 5,
        "drumroll": 6,
        "sliderend": 7,
    }
    notes.sort(key=lambda n: (float(_safe_float(n.get("time", 0.0), 0.0)), type_priority.get(str(n.get("type", "")).lower(), 999)))
    return {
        "format": 2,
        "mode": 1,
        "source_osu": source_osu,
        "notes": notes,
    }


def mark_generated_metadata(metadata_json: dict[str, Any], *, architecture_name: str, checkpoint_name: str) -> dict[str, Any]:
    out = copy.deepcopy(metadata_json)
    md = out.setdefault("metadata", {})
    original_version = str(md.get("Version", "Generated")).strip() or "Generated"
    original_creator = str(md.get("Creator", "Unknown")).strip() or "Unknown"
    tags = str(md.get("Tags", "")).strip()

    md["Version"] = f"{original_version} [AI {architecture_name}]"
    md["Creator"] = f"{original_creator} + taiko-diffusion"
    if "ai-generated" not in tags.lower():
        md["Tags"] = (tags + " ai-generated").strip()
    md["BeatmapID"] = "0"

    out.setdefault("inference", {})
    out["inference"]["checkpoint"] = checkpoint_name
    out["inference"]["architecture"] = architecture_name
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sanitize_filename_component(text: str) -> str:
    invalid = '<>:"/\\|?*'
    out = "".join("_" if ch in invalid else ch for ch in str(text))
    return out.strip().rstrip(".")


def _build_osu_filename_from_version(chart_stem: str, version_name: str) -> str:
    safe_version = _sanitize_filename_component(version_name) or "Generated"
    stem_text = str(chart_stem).strip()
    left = stem_text.rfind("[")
    right = stem_text.rfind("]")
    if left != -1 and right != -1 and right > left:
        rebuilt = f"{stem_text[:left].rstrip()} [{safe_version}]"
    else:
        rebuilt = f"{stem_text} [{safe_version}]"
    return f"{_sanitize_filename_component(rebuilt)}.osu"


def _checkpoint_output_slug(checkpoint_path: Path, architecture_name: str) -> str:
    label = f"{checkpoint_path.parent.name}_{checkpoint_path.stem}_{architecture_name}"
    return _sanitize_filename_component(label)


def load_generator_from_checkpoint(
    checkpoint_path: Path,
    *,
    device,
    max_decode_len: int,
    audio_cache_size: int,
) -> tuple[Any, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for checkpoint-based generation. Install torch to run inference.")
    from src.model.checkpoints import load_inference_artifacts
    from src.model.factory import build_model
    from src.model.generation import TaikoBeatmapGenerator
    from src.model.specs import ArchitectureSpec

    payload = load_inference_artifacts(checkpoint_path, map_location="cpu")
    architecture_spec = ArchitectureSpec.from_dict(payload["architecture_spec"])
    vocab_payload = payload["vocab"]
    token_to_id_raw = vocab_payload.get("token_to_id", {})
    token_to_id = {str(token): int(idx) for token, idx in token_to_id_raw.items()}
    id_to_token = {int(idx): token for token, idx in token_to_id.items()}

    model = build_model(architecture_spec, vocab_size=len(token_to_id))
    model.load_state_dict(payload["model_state_dict"])
    model = model.to(device)
    model.eval()

    generator = TaikoBeatmapGenerator(
        model=model,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        device=device,
        max_len=max_decode_len,
        audio_cache_size=audio_cache_size,
    )
    return generator, architecture_spec


def infer_for_checkpoints(
    checkpoints: list[str | Path],
    chart_input: MetadataInferenceInput,
    *,
    output_root: str | Path,
    sampling_config: Any,
    device: str | None = None,
    max_decode_len: int = 64,
    audio_cache_size: int = 8,
    source_group_name: str = "metadata_input",
) -> list[Path]:
    if torch is None:
        raise RuntimeError("PyTorch is required for inference. Install torch in this environment.")

    output_root = Path(output_root).resolve()
    resolved_device = torch.device(device or _default_device())
    output_paths: list[Path] = []

    for checkpoint_raw in checkpoints:
        checkpoint_path = Path(checkpoint_raw).resolve()
        generator, architecture_spec = load_generator_from_checkpoint(
            checkpoint_path,
            device=resolved_device,
            max_decode_len=max(1, int(max_decode_len)),
            audio_cache_size=max(1, int(audio_cache_size)),
        )
        checkpoint_name = checkpoint_path.stem
        output_slug = _checkpoint_output_slug(checkpoint_path, str(architecture_spec.name))
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
            checkpoint_name=checkpoint_name,
        )
        generated_timing = copy.deepcopy(chart_input.timing_json)
        version_name = str(generated_metadata.get("metadata", {}).get("Version", f"AI {architecture_spec.name}"))

        # Keep all model outputs for the same chart in one shared folder.
        chart_output_dir = output_root / source_group_name / chart_input.chart_stem
        notes_path = chart_output_dir / f"{chart_input.chart_stem}.{output_slug}.generated.notes.json"
        timing_path = chart_output_dir / f"{chart_input.chart_stem}.{output_slug}.generated.timing.json"
        metadata_path = chart_output_dir / f"{chart_input.chart_stem}.{output_slug}.generated.metadata.json"
        song_output_path = chart_output_dir / f"{chart_input.chart_stem}.{output_slug}.song_output.json"
        out_osu_filename = _build_osu_filename_from_version(
            chart_stem=chart_input.chart_stem,
            version_name=version_name,
        )
        out_osu_path = chart_output_dir / f"{Path(out_osu_filename).stem}.{output_slug}.osu"

        _write_json(notes_path, generated_notes)
        _write_json(timing_path, generated_timing)
        _write_json(metadata_path, generated_metadata)
        _write_json(song_output_path, {"song_output": song_output})

        reconstruct_osu(
            notes_path=notes_path,
            out_path=out_osu_path,
            timing_path=timing_path,
            metadata_path=metadata_path,
        )
        output_paths.append(out_osu_path)
    return output_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer beatmaps from metadata/timing and audio (without requiring .osz selection stage)."
    )
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths to run.")
    parser.add_argument("--audio-path", required=True, help="Audio path used for inference.")
    parser.add_argument("--timing-json", required=True, help="Timing JSON path for conditioning/reconstruction.")
    parser.add_argument("--metadata-json", required=True, help="Metadata JSON path for conditioning/reconstruction.")
    parser.add_argument("--notes-json", default=None, help="Optional reference notes JSON (for sv/volume baseline).")
    parser.add_argument("--chart-stem", default=None, help="Optional chart stem label for outputs.")
    parser.add_argument("--difficulty-value", type=float, default=None, help="Override difficulty value used for conditioning.")
    parser.add_argument("--beatmap-id", type=int, default=None, help="Override beatmap id used for conditioning.")
    parser.add_argument("--density-nps", type=float, default=None, help="Override density used for conditioning.")
    parser.add_argument("--output-root", default="sample_data/inference_outputs", help="Output root.")
    parser.add_argument("--source-group-name", default="metadata_input", help="Group folder under checkpoint output.")
    parser.add_argument("--device", default=None, help="Torch device. Defaults to cuda/mps/cpu auto.")
    parser.add_argument("--max-decode-len", type=int, default=64, help="Max generated tokens per 4-beat window.")
    parser.add_argument("--audio-cache-size", type=int, default=8, help="Audio cache size.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.82, help="Nucleus sampling threshold.")
    parser.add_argument("--top-k", type=int, default=4, help="Top-k candidate count.")
    parser.add_argument("--ts-top-k", type=int, default=2, help="Top-k count for TS_* tokens.")
    parser.add_argument("--min-event-candidates", type=int, default=2, help="Minimum event token candidates.")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty factor.")
    return parser


def main(argv: list[str] | None = None) -> int:
    if torch is None:
        raise RuntimeError("PyTorch is required for inference. Install torch in this environment.")
    from src.model.generation import SamplingConfig

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    timing_json = load_json(Path(args.timing_json))
    metadata_json = load_json(Path(args.metadata_json))
    notes_json = load_json(Path(args.notes_json)) if args.notes_json else None
    validate_constant_bpm_timing_json(timing_json)

    offset_ms, bpm, meter = extract_primary_timing(timing_json)
    chart_stem = args.chart_stem or Path(args.metadata_json).name.replace(".metadata.json", "")
    difficulty_value = (
        float(args.difficulty_value)
        if args.difficulty_value is not None
        else _infer_difficulty_value(metadata_json.get("difficulty", {}).get("OverallDifficulty", metadata_json.get("metadata", {}).get("Version", "")))
    )
    beatmap_id = (
        int(args.beatmap_id)
        if args.beatmap_id is not None
        else _infer_beatmap_id_value(chart_stem, metadata_json.get("metadata", {}).get("BeatmapID", ""))
    )
    density_nps = (
        float(args.density_nps)
        if args.density_nps is not None
        else estimate_density_nps(notes_json or {"notes": []}, offset_ms=offset_ms)
    )

    chart_input = MetadataInferenceInput(
        chart_stem=chart_stem,
        audio_path=Path(args.audio_path).resolve(),
        metadata_json=metadata_json,
        timing_json=timing_json,
        offset_ms=offset_ms,
        bpm=bpm,
        meter=meter,
        difficulty_value=difficulty_value,
        beatmap_id=beatmap_id,
        density_nps=density_nps,
        reference_notes_json=notes_json,
    )
    sampling_config = SamplingConfig(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=max(1, int(args.top_k)),
        ts_top_k=max(0, int(args.ts_top_k)),
        min_event_candidates=max(1, int(args.min_event_candidates)),
        repetition_penalty=float(args.repetition_penalty),
    )
    outputs = infer_for_checkpoints(
        checkpoints=list(args.checkpoints),
        chart_input=chart_input,
        output_root=args.output_root,
        sampling_config=sampling_config,
        device=args.device,
        max_decode_len=args.max_decode_len,
        audio_cache_size=args.audio_cache_size,
        source_group_name=str(args.source_group_name),
    )
    for out in outputs:
        print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
