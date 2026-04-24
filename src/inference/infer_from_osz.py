from __future__ import annotations

import argparse
from glob import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - handled by runtime check
    torch = None

from src.inference.infer_from_metadata import (
    MetadataInferenceInput,
    estimate_density_nps,
    extract_primary_timing,
    infer_for_checkpoints,
    load_json,
    song_output_to_notes_json,  # re-exported for compatibility/tests
)
from src.preprocessing.osutaiko_parser import parse_osu_file_to_jsons
from src.preprocessing.unpack_osz import unpack_osz_files


@dataclass
class ChartCandidate:
    stem: str
    unpacked_dir: Path
    metadata_path: Path
    timing_path: Path
    notes_path: Path
    audio_path: Path
    metadata_json: dict[str, Any]
    timing_json: dict[str, Any]
    notes_json: dict[str, Any]
    offset_ms: float
    bpm: float
    meter: int
    difficulty_value: float
    beatmap_id: int
    density_nps: float
    playable_notes: int
    overall_difficulty: float


def _default_device() -> str:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def _resolve_osz_input_paths(osz_inputs: list[str]) -> list[str]:
    resolved: list[str] = []
    for raw_input in osz_inputs:
        text = str(raw_input)
        path = Path(text)
        if path.is_dir():
            resolved.extend(sorted(str(p) for p in path.glob("*.osz")))
            continue
        matches = sorted(glob(text))
        if matches:
            resolved.extend(matches)
            continue
        if path.exists() and path.suffix.lower() == ".osz":
            resolved.append(str(path))

    deduped = sorted({str(Path(p).resolve()) for p in resolved})
    if not deduped:
        raise FileNotFoundError("No .osz files matched the provided --osz-inputs.")
    return deduped


def _is_constant_bpm_chart(timing_json: dict[str, Any]) -> bool:
    timing_points = list(timing_json.get("timing_points", []))
    bpm_points = [
        tp for tp in timing_points
        if int(tp.get("uninherited", 0)) == 1 and _safe_float(tp.get("ms_per_beat", 0.0), 0.0) > 0.0
    ]
    unique_mpb = {round(_safe_float(tp.get("ms_per_beat", 0.0), 0.0), 10) for tp in bpm_points}
    return len(unique_mpb) == 1


def _count_playable_notes(notes_json: dict[str, Any]) -> int:
    notes = list(notes_json.get("notes", []))
    return sum(1 for note in notes if str(note.get("type", "")).lower() != "bpmchange")


def _resolve_audio_path(unpacked_dir: Path, metadata_json: dict[str, Any]) -> Path:
    audio_filename = str(metadata_json.get("general", {}).get("AudioFilename", "")).strip()
    if audio_filename:
        candidate = unpacked_dir / audio_filename
        if candidate.exists():
            return candidate
    for suffix in (".mp3", ".ogg", ".wav", ".flac", ".m4a", ".aac", ".opus"):
        matches = list(unpacked_dir.glob(f"*{suffix}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No audio file found in unpacked directory: {unpacked_dir}")


def _build_candidate(unpacked_dir: Path, stem: str, parsed_dir: Path) -> ChartCandidate:
    metadata_path = parsed_dir / f"{stem}.metadata.json"
    timing_path = parsed_dir / f"{stem}.timing.json"
    notes_path = parsed_dir / f"{stem}.notes.json"
    metadata_json = load_json(metadata_path)
    timing_json = load_json(timing_path)
    notes_json = load_json(notes_path)
    offset_ms, bpm, meter = extract_primary_timing(timing_json)
    playable_notes = _count_playable_notes(notes_json)
    density_nps = estimate_density_nps(notes_json, offset_ms=offset_ms)
    version_name = str(metadata_json.get("metadata", {}).get("Version", ""))
    od_value = _safe_float(
        metadata_json.get("difficulty", {}).get("OverallDifficulty", ""),
        float("nan"),
    )
    if od_value != od_value:
        od_value = float(_infer_difficulty_value(version_name))

    beatmap_id_raw = metadata_json.get("metadata", {}).get("BeatmapID", "")
    beatmap_id = _safe_int(
        beatmap_id_raw,
        int(_infer_beatmap_id_value(chart_id=stem, explicit_beatmap_id=beatmap_id_raw)),
    )
    difficulty_value = float(od_value)
    audio_path = _resolve_audio_path(unpacked_dir, metadata_json)

    return ChartCandidate(
        stem=stem,
        unpacked_dir=unpacked_dir,
        metadata_path=metadata_path,
        timing_path=timing_path,
        notes_path=notes_path,
        audio_path=audio_path,
        metadata_json=metadata_json,
        timing_json=timing_json,
        notes_json=notes_json,
        offset_ms=offset_ms,
        bpm=bpm,
        meter=meter,
        difficulty_value=difficulty_value,
        beatmap_id=beatmap_id,
        density_nps=density_nps,
        playable_notes=playable_notes,
        overall_difficulty=float(od_value),
    )


def select_top_difficulty_chart(unpacked_dir: Path) -> ChartCandidate:
    parsed_dir = unpacked_dir / "parsed"
    if not parsed_dir.exists():
        raise FileNotFoundError(f"Parsed directory does not exist: {parsed_dir}")

    candidates: list[ChartCandidate] = []
    for metadata_path in sorted(parsed_dir.glob("*.metadata.json")):
        stem = metadata_path.name[:-len(".metadata.json")]
        timing_path = parsed_dir / f"{stem}.timing.json"
        notes_path = parsed_dir / f"{stem}.notes.json"
        if not timing_path.exists() or not notes_path.exists():
            continue
        timing_json = load_json(timing_path)
        if not _is_constant_bpm_chart(timing_json):
            continue
        try:
            candidate = _build_candidate(unpacked_dir, stem, parsed_dir)
        except Exception:
            continue
        candidates.append(candidate)

    if not candidates:
        raise RuntimeError(f"No eligible constant-BPM taiko chart candidate found in: {parsed_dir}")

    candidates.sort(
        key=lambda c: (
            float(c.overall_difficulty),
            int(c.playable_notes),
            str(c.stem),
        ),
        reverse=True,
    )
    return candidates[0]


def _ensure_parsed_jsons(unpacked_dir: Path, overwrite_parsed: bool = False) -> None:
    parsed_dir = unpacked_dir / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    for osu_path in sorted(unpacked_dir.glob("*.osu")):
        stem = osu_path.stem
        expected = [
            parsed_dir / f"{stem}.metadata.json",
            parsed_dir / f"{stem}.timing.json",
            parsed_dir / f"{stem}.notes.json",
        ]
        if not overwrite_parsed and all(path.exists() for path in expected):
            continue
        try:
            parse_osu_file_to_jsons(
                osu_path=osu_path,
                out_dir=parsed_dir,
                include_bpm_events=True,
            )
        except Exception:
            continue


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run model inference from .osz inputs and reconstruct generated .osu charts.")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Checkpoint paths to run (baseline and/or context).",
    )
    parser.add_argument(
        "--osz-inputs",
        nargs="+",
        default=["sample_data/raw/*2034220*.osz"],
        help="One or more .osz files, directories, or glob patterns. Defaults to beachballs sample glob.",
    )
    parser.add_argument(
        "--work-root",
        default="sample_data/inference_work",
        help="Working directory for unpacked + parsed inference assets.",
    )
    parser.add_argument(
        "--output-root",
        default="sample_data/inference_outputs",
        help="Output root for generated notes/metadata/timing/.osu files.",
    )
    parser.add_argument("--overwrite-unpack", action="store_true", help="Re-unpack .osz archives even if extracted folders exist.")
    parser.add_argument("--overwrite-parsed", action="store_true", help="Re-parse .osu files even if parsed JSONs already exist.")
    parser.add_argument("--device", default=None, help="Torch device for inference. Defaults to cuda/mps/cpu auto.")
    parser.add_argument("--max-decode-len", type=int, default=64, help="Maximum generated tokens per 4-beat window.")
    parser.add_argument("--audio-cache-size", type=int, default=8, help="Audio preprocessing cache size in generator.")
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

    resolved_osz = _resolve_osz_input_paths(args.osz_inputs)
    work_root = Path(args.work_root).resolve()
    output_root = Path(args.output_root).resolve()
    unpacked_root = work_root / "unpacked"
    unpacked_dirs = unpack_osz_files(
        source_paths=resolved_osz,
        destination_root=unpacked_root,
        overwrite=bool(args.overwrite_unpack),
        keep_only_chart_and_audio=True,
        progress_desc="Unpacking inference .osz files",
    )

    selected_charts: list[ChartCandidate] = []
    for unpacked_dir in unpacked_dirs:
        unpacked_dir = Path(unpacked_dir).resolve()
        _ensure_parsed_jsons(unpacked_dir, overwrite_parsed=bool(args.overwrite_parsed))
        top_chart = select_top_difficulty_chart(unpacked_dir)
        selected_charts.append(top_chart)
        print(
            f"[select] {unpacked_dir.name} -> {top_chart.stem} | "
            f"OD={top_chart.overall_difficulty:.2f} notes={top_chart.playable_notes} "
            f"bpm={top_chart.bpm:.3f} offset_ms={top_chart.offset_ms:.1f}"
        )

    if not selected_charts:
        raise RuntimeError("No eligible charts were selected for inference.")

    sampling_config = SamplingConfig(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=max(1, int(args.top_k)),
        ts_top_k=max(0, int(args.ts_top_k)),
        min_event_candidates=max(1, int(args.min_event_candidates)),
        repetition_penalty=float(args.repetition_penalty),
    )
    device_text = args.device or _default_device()

    for chart in selected_charts:
        chart_input = MetadataInferenceInput(
            chart_stem=chart.stem,
            audio_path=chart.audio_path,
            metadata_json=chart.metadata_json,
            timing_json=chart.timing_json,
            offset_ms=chart.offset_ms,
            bpm=chart.bpm,
            meter=chart.meter,
            difficulty_value=chart.difficulty_value,
            beatmap_id=chart.beatmap_id,
            density_nps=chart.density_nps,
            reference_notes_json=chart.notes_json,
        )
        outputs = infer_for_checkpoints(
            checkpoints=list(args.checkpoints),
            chart_input=chart_input,
            output_root=output_root,
            sampling_config=sampling_config,
            device=device_text,
            max_decode_len=max(1, int(args.max_decode_len)),
            audio_cache_size=max(1, int(args.audio_cache_size)),
            source_group_name=chart.unpacked_dir.name,
        )
        for out in outputs:
            print(f"[done] chart={chart.stem} output={out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

