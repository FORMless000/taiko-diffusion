from pathlib import Path
import json
import math
import torch

from src.preprocessing.beat_aligned_dataset import (
    get_audio_info,
    compute_beat_grid_info,
    build_beat_aligned_frame_timeline,
    build_raw_mel_spectrogram,
    interpolate_raw_mel_to_beat_aligned_timeline,
    segment_aligned_mel_into_4beat_sequences,
)


class TaikoBeatmapGenerator:
    def __init__(self, model, token_to_id, id_to_token, device, max_len=64):
        self.model = model
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.device = device
        self.max_len = max_len

    def build_timing_info(self, audio_path, offset_ms, bpm, meter=4):
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if bpm <= 0:
            raise ValueError("bpm must be positive.")
        if meter <= 0:
            raise ValueError("meter must be positive.")

        beat_duration_ms = 60000.0 / bpm
        return {
            "audio_path": str(audio_path),
            "offset_ms": float(offset_ms),
            "bpm": float(bpm),
            "meter": int(meter),
            "beat_duration_ms": float(beat_duration_ms),
        }

    @torch.no_grad()
    def greedy_decode(self, audio):
        self.model.eval()

        bos_id = self.token_to_id["BOS"]
        eos_id = self.token_to_id["EOS"]

        if audio.dim() == 2:
            audio = audio.unsqueeze(0)

        audio = audio.to(self.device)
        generated = torch.tensor([[bos_id]], dtype=torch.long, device=self.device)

        for _ in range(self.max_len):
            decoder_attention_mask = torch.ones_like(generated, device=self.device)
            logits = self.model(
                audio=audio,
                input_ids=generated,
                decoder_attention_mask=decoder_attention_mask,
            )

            next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token_id], dim=1)

            if next_token_id.item() == eos_id:
                break

        generated_ids = generated[0].tolist()

        if generated_ids and generated_ids[0] == bos_id:
            generated_ids = generated_ids[1:]
        if generated_ids and generated_ids[-1] == eos_id:
            generated_ids = generated_ids[:-1]

        generated_tokens = [self.id_to_token[i] for i in generated_ids]
        return generated_ids, generated_tokens

    def preprocess_audio(self, audio_path, offset_ms, bpm, meter=4):
        timing_info = self.build_timing_info(
            audio_path=audio_path,
            offset_ms=offset_ms,
            bpm=bpm,
            meter=meter,
        )

        audio_info = get_audio_info(Path(timing_info["audio_path"]))
        waveform = audio_info["waveform"]
        sample_rate = audio_info["sample_rate"]
        audio_duration_ms = audio_info["audio_duration_ms"]

        beat_grid_info, _ = compute_beat_grid_info(
            offset_ms=timing_info["offset_ms"],
            beat_duration_ms=timing_info["beat_duration_ms"],
            audio_duration_ms=audio_duration_ms,
        )

        beat_aligned_frame_times_ms = build_beat_aligned_frame_timeline(
            offset_ms=timing_info["offset_ms"],
            beat_duration_ms=timing_info["beat_duration_ms"],
            total_frames=beat_grid_info.total_frames,
        )

        mel_spec_db, orig_frame_times_ms = build_raw_mel_spectrogram(
            waveform=waveform,
            sample_rate=sample_rate,
        )

        aligned_mel_db = interpolate_raw_mel_to_beat_aligned_timeline(
            mel_spec_db=mel_spec_db,
            orig_frame_times_ms=orig_frame_times_ms,
            beat_aligned_frame_times_ms=beat_aligned_frame_times_ms,
        )

        audio_sequences = segment_aligned_mel_into_4beat_sequences(
            aligned_mel_db=aligned_mel_db,
            total_sequences=beat_grid_info.total_sequences,
        )

        return audio_sequences

    def generate_tokens(self, audio_path, offset_ms, bpm, meter=4):
        audio_sequences = self.preprocess_audio(
            audio_path=audio_path,
            offset_ms=offset_ms,
            bpm=bpm,
            meter=meter,
        )

        all_pred_tokens = []
        all_pred_ids = []

        for seq_idx in range(len(audio_sequences)):
            audio_seq = torch.tensor(audio_sequences[seq_idx], dtype=torch.float32)
            pred_ids, pred_tokens = self.greedy_decode(audio_seq)
            all_pred_ids.append(pred_ids)
            all_pred_tokens.append(pred_tokens)

        return all_pred_tokens

    def generate_tokens_with_ids(self, audio_path, offset_ms, bpm, meter=4):
        audio_sequences = self.preprocess_audio(
            audio_path=audio_path,
            offset_ms=offset_ms,
            bpm=bpm,
            meter=meter,
        )

        all_pred_tokens = []
        all_pred_ids = []

        for seq_idx in range(len(audio_sequences)):
            audio_seq = torch.tensor(audio_sequences[seq_idx], dtype=torch.float32)
            pred_ids, pred_tokens = self.greedy_decode(audio_seq)
            all_pred_ids.append(pred_ids)
            all_pred_tokens.append(pred_tokens)

        return all_pred_ids, all_pred_tokens

    def generate_song_structure(self, audio_path, offset_ms, bpm, meter=4):
        audio_sequences = self.preprocess_audio(
            audio_path=audio_path,
            offset_ms=offset_ms,
            bpm=bpm,
            meter=meter,
        )

        song_output = []

        for seq_idx in range(len(audio_sequences)):
            audio_seq = torch.tensor(audio_sequences[seq_idx], dtype=torch.float32)
            pred_ids, pred_tokens = self.greedy_decode(audio_seq)

            start_frame = seq_idx * 192
            end_frame = start_frame + 191

            song_output.append(
                {
                    "seq_idx": seq_idx,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "pred_ids": pred_ids,
                    "pred_tokens": pred_tokens,
                }
            )

        return song_output


def compare_song_output_with_notes_json(song_output, gt_json_path, max_sequences=10):
    gt_json_path = Path(gt_json_path)
    if not gt_json_path.exists():
        raise FileNotFoundError(f"File not found: {gt_json_path}")

    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    if "notes" not in gt_data:
        raise ValueError("This file is not a raw notes.json with top-level key 'notes'.")

    raw_notes = gt_data["notes"]

    bpm_events = [x for x in raw_notes if x.get("type") == "bpmchange"]
    if len(bpm_events) == 0:
        raise ValueError("No bpmchange found in notes.json, cannot infer timing.")

    first_bpm = bpm_events[0]
    offset_ms = float(first_bpm["time"])
    bpm = float(first_bpm["bpm"])
    meter = int(first_bpm["meter"]) if first_bpm["meter"] is not None else 4

    if bpm <= 0:
        raise ValueError("Invalid bpm in bpmchange.")

    beat_duration_ms = 60000.0 / bpm
    tick_ms = beat_duration_ms / 48.0
    seq_ticks = 192
    seq_duration_ms = tick_ms * seq_ticks

    type_to_token = {
        "don": "DON",
        "kat": "KAT",
        "bigdon": "BIGDON",
        "bigkat": "BIGKAT",
        "drumroll": "DRUMROLL",
        "sliderstart": "SLIDERSTART",
        "sliderend": "SLIDEREND",
    }

    event_notes = []
    for note in raw_notes:
        note_type = note.get("type")
        if note_type in type_to_token:
            event_notes.append(
                {
                    "time": float(note["time"]),
                    "token": type_to_token[note_type],
                }
            )

    seq_to_events = {}

    for ev in event_notes:
        rel_ms = ev["time"] - offset_ms
        seq_idx = int(math.floor((rel_ms + 1e-6) / seq_duration_ms))
        seq_idx = max(seq_idx, 0)

        seq_start_ms = offset_ms + seq_idx * seq_duration_ms
        rel_in_seq_ms = ev["time"] - seq_start_ms
        pos_tick = int(round(rel_in_seq_ms / tick_ms))
        pos_tick = max(0, min(seq_ticks - 1, pos_tick))

        seq_to_events.setdefault(seq_idx, []).append((pos_tick, ev["token"]))

    gt_song_output = []
    max_seq_idx_from_gt = max(seq_to_events.keys()) if seq_to_events else -1
    total_gt_sequences = max_seq_idx_from_gt + 1

    for seq_idx in range(total_gt_sequences):
        events = seq_to_events.get(seq_idx, [])
        events = sorted(events, key=lambda x: (x[0], x[1]))

        tokens = []
        cursor = 0

        for pos_tick, token in events:
            gap = pos_tick - cursor
            if gap > 0:
                tokens.append(f"TS_{gap}")
            tokens.append(token)
            cursor = pos_tick

        gt_song_output.append(
            {
                "seq_idx": seq_idx,
                "start_frame": seq_idx * 192,
                "end_frame": seq_idx * 192 + 191,
                "tokens": tokens,
            }
        )

    n = min(len(song_output), len(gt_song_output), max_sequences)

    lines = []
    lines.append(f"offset_ms={offset_ms}, bpm={bpm}, meter={meter}")
    lines.append("=" * 80)

    for i in range(n):
        gt_tokens = gt_song_output[i]["tokens"]
        pred_tokens = song_output[i]["pred_tokens"]

        gt_str = " ".join(gt_tokens)
        pred_str = " ".join(pred_tokens)

        lines.append(f"Sequence {i}")
        lines.append(f"GT  : {gt_str}")
        lines.append(f"PRED: {pred_str}")
        lines.append("-" * 80)

    return "\n".join(lines)
