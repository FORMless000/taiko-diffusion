from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import json
import math
import torch
import torch.nn.functional as F

from src.model.data import (
    preprocess_beatmap_id,
    preprocess_density_nps,
    preprocess_difficulty_value,
)
from src.preprocessing.beat_aligned_dataset import (
    get_audio_info,
    compute_beat_grid_info,
    build_beat_aligned_frame_timeline,
    build_raw_mel_spectrogram,
    interpolate_raw_mel_to_beat_aligned_timeline,
    segment_aligned_mel_into_4beat_sequences,
)


@dataclass
class SamplingConfig:
    temperature: float = 0.9
    top_p: float = 0.82
    top_k: int = 4
    ts_top_k: int = 2
    min_event_candidates: int = 2
    repetition_penalty: float = 1.0


class TaikoBeatmapGenerator:
    def __init__(self, model, token_to_id, id_to_token, device, max_len=64, audio_cache_size=8):
        self.model = model
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.device = device
        self.max_len = max_len
        self.audio_cache_size = max(1, int(audio_cache_size))
        self._audio_cache = OrderedDict()

        self.pad_id = token_to_id.get("PAD")
        self.bos_id = token_to_id["BOS"]
        self.eos_id = token_to_id["EOS"]
        self.context_enabled = bool(getattr(model, "supports_long_context", False))
        self.history_max_tokens = max(1, int(getattr(model, "history_max_tokens", 1024)))
        self.retrieval_top_k = max(0, int(getattr(model, "retrieval_top_k", 0)))
        self.retrieval_max_tokens_per_window = max(1, int(getattr(model, "retrieval_max_tokens_per_window", 64)))
        self.retrieval_exclude_last_n_windows = max(0, int(getattr(model, "retrieval_exclude_last_n_windows", 0)))
        self.use_motif_retrieval = bool(getattr(model, "use_motif_retrieval", False))

        ts_ids = []
        event_ids = []
        for tid, tok in id_to_token.items():
            if tok in {"PAD", "BOS"}:
                continue
            if tok.startswith("TS_"):
                ts_ids.append(tid)
            else:
                event_ids.append(tid)

        self.ts_token_ids = sorted(ts_ids)
        self.event_token_ids = sorted(event_ids)
        self.ts_token_ids_tensor = torch.tensor(self.ts_token_ids, dtype=torch.long, device=self.device)
        self.event_token_ids_tensor = torch.tensor(self.event_token_ids, dtype=torch.long, device=self.device)

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

    def _cache_get_or_compute(self, cache_key, compute_fn):
        if cache_key in self._audio_cache:
            value = self._audio_cache.pop(cache_key)
            self._audio_cache[cache_key] = value
            return value

        value = compute_fn()
        self._audio_cache[cache_key] = value
        if len(self._audio_cache) > self.audio_cache_size:
            self._audio_cache.popitem(last=False)
        return value

    def _resolve_condition_values(self, difficulty=5.0, density_nps=6.0, beatmap_id=1_000_000):
        difficulty_norm = preprocess_difficulty_value(difficulty)
        density_norm = preprocess_density_nps(density_nps)
        beatmap_id_norm = preprocess_beatmap_id(beatmap_id)

        difficulty_values = torch.tensor([difficulty_norm], dtype=torch.float32, device=self.device)
        density_values = torch.tensor([density_norm], dtype=torch.float32, device=self.device)
        beatmap_id_values = torch.tensor([beatmap_id_norm], dtype=torch.float32, device=self.device)
        return difficulty_values, density_values, beatmap_id_values

    def _pool_audio_key_from_memory(self, memory):
        if hasattr(self.model, "_pool_audio_memory"):
            return self.model._pool_audio_memory(memory)[0]

        pooled = memory.mean(dim=1)
        return F.normalize(pooled, dim=-1)[0]

    def _build_recent_history_ids(self, prior_window_token_ids):
        flattened = []
        for token_ids in prior_window_token_ids:
            flattened.extend(token_ids)

        if len(flattened) > self.history_max_tokens:
            flattened = flattened[-self.history_max_tokens :]
        return flattened

    def _truncate_retrieved_window_ids(self, token_ids):
        limit = self.retrieval_max_tokens_per_window
        payload = list(token_ids[: max(0, limit - 1)])
        payload.append(self.eos_id)
        return payload[:limit]

    def _build_retrieved_ids(self, current_audio_key, prior_audio_keys, prior_window_token_ids):
        if not self.context_enabled or not self.use_motif_retrieval or self.retrieval_top_k <= 0:
            return []

        eligible_count = len(prior_window_token_ids) - self.retrieval_exclude_last_n_windows
        if eligible_count <= 0:
            return []

        candidate_keys = torch.stack(prior_audio_keys[:eligible_count], dim=0)
        similarity = torch.matmul(candidate_keys, current_audio_key.unsqueeze(-1)).squeeze(-1)
        top_k = min(self.retrieval_top_k, int(similarity.numel()))
        if top_k <= 0:
            return []

        _, top_indices = torch.topk(similarity, k=top_k)
        retrieved_ids = []
        for idx in top_indices.tolist():
            retrieved_ids.extend(self._truncate_retrieved_window_ids(prior_window_token_ids[idx]))
        return retrieved_ids

    def _build_context_tensors(self, generated, recent_history_ids=None, retrieved_token_ids=None):
        if not self.context_enabled:
            return generated, None

        history_ids = list(recent_history_ids or [])
        retrieved_ids = list(retrieved_token_ids or [])
        current_ids = generated[0].tolist()

        full_input_ids = history_ids + retrieved_ids + current_ids
        segment_ids = (
            [0] * len(history_ids)
            + [1] * len(retrieved_ids)
            + [2] * len(current_ids)
        )

        input_ids = torch.tensor([full_input_ids], dtype=torch.long, device=self.device)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=self.device)
        return input_ids, segment_ids

    def _apply_repetition_penalty(self, logits, generated_ids, repetition_penalty):
        if repetition_penalty is None or repetition_penalty <= 1.0 or generated_ids.numel() == 0:
            return logits

        adjusted = logits.clone()
        unique_ids = torch.unique(generated_ids)
        for token_id in unique_ids:
            idx = int(token_id.item())
            if adjusted[idx] < 0:
                adjusted[idx] *= repetition_penalty
            else:
                adjusted[idx] /= repetition_penalty
        return adjusted

    def _class_aware_candidate_ids(self, logits_1d, top_k, ts_top_k, min_event_candidates):
        if self.event_token_ids_tensor.numel() == 0:
            return torch.arange(logits_1d.size(0), device=logits_1d.device)

        event_k = max(int(min_event_candidates), min(int(top_k), int(self.event_token_ids_tensor.numel())))
        if event_k <= 0:
            event_k = min_event_candidates

        event_scores = logits_1d.index_select(0, self.event_token_ids_tensor)
        _, event_local_idx = torch.topk(event_scores, k=event_k)
        event_ids = self.event_token_ids_tensor.index_select(0, event_local_idx)

        if self.ts_token_ids_tensor.numel() > 0 and ts_top_k > 0:
            ts_k = min(int(ts_top_k), int(self.ts_token_ids_tensor.numel()))
            ts_scores = logits_1d.index_select(0, self.ts_token_ids_tensor)
            _, ts_local_idx = torch.topk(ts_scores, k=ts_k)
            ts_ids = self.ts_token_ids_tensor.index_select(0, ts_local_idx)
            candidates = torch.cat([event_ids, ts_ids], dim=0)
        else:
            candidates = event_ids

        if self.eos_id not in candidates.tolist():
            candidates = torch.cat(
                [candidates, torch.tensor([self.eos_id], dtype=torch.long, device=logits_1d.device)],
                dim=0,
            )

        return torch.unique(candidates)

    def _apply_top_p(self, probs_1d, token_ids, top_p):
        top_p = float(max(0.05, min(1.0, top_p)))

        sorted_probs, sorted_idx = torch.sort(probs_1d, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=0)
        keep_sorted = cumulative <= top_p
        keep_sorted[0] = True

        kept_idx = sorted_idx[keep_sorted]
        kept_ids = token_ids.index_select(0, kept_idx)
        kept_probs = probs_1d.index_select(0, kept_idx)
        kept_probs = kept_probs / kept_probs.sum()
        return kept_ids, kept_probs

    def _sample_next_token(self, logits_last, generated_ids, sampling_config):
        temperature = float(sampling_config.temperature)
        top_p = float(sampling_config.top_p)
        top_k = int(max(1, sampling_config.top_k))
        ts_top_k = int(max(0, sampling_config.ts_top_k))
        min_event_candidates = int(max(1, sampling_config.min_event_candidates))

        logits_1d = logits_last[0]
        logits_1d = self._apply_repetition_penalty(
            logits_1d,
            generated_ids,
            repetition_penalty=float(sampling_config.repetition_penalty),
        )

        if temperature <= 0.0:
            next_id = torch.argmax(logits_1d).view(1, 1)
            return next_id

        scaled_logits = logits_1d / max(1e-6, temperature)
        candidate_ids = self._class_aware_candidate_ids(
            scaled_logits,
            top_k=top_k,
            ts_top_k=ts_top_k,
            min_event_candidates=min_event_candidates,
        )

        candidate_logits = scaled_logits.index_select(0, candidate_ids)
        candidate_probs = torch.softmax(candidate_logits, dim=0)

        filtered_ids, filtered_probs = self._apply_top_p(candidate_probs, candidate_ids, top_p)

        event_mask = torch.isin(filtered_ids, self.event_token_ids_tensor)
        if event_mask.sum().item() < min_event_candidates:
            event_candidates = self.event_token_ids_tensor
            event_scores = scaled_logits.index_select(0, event_candidates)
            _, extra_idx = torch.topk(
                event_scores,
                k=min(min_event_candidates, int(event_candidates.numel())),
            )
            extra_ids = event_candidates.index_select(0, extra_idx)

            merged_ids = torch.unique(torch.cat([filtered_ids, extra_ids], dim=0))
            merged_logits = scaled_logits.index_select(0, merged_ids)
            merged_probs = torch.softmax(merged_logits, dim=0)
            filtered_ids, filtered_probs = self._apply_top_p(merged_probs, merged_ids, top_p=1.0)

        sampled_local = torch.multinomial(filtered_probs, num_samples=1)
        next_id = filtered_ids.index_select(0, sampled_local).view(1, 1)
        return next_id

    @torch.no_grad()
    def decode_sequence(
        self,
        audio,
        difficulty=5.0,
        density_nps=6.0,
        beatmap_id=1_000_000,
        sampling_config=None,
        memory=None,
        recent_history_ids=None,
        retrieved_token_ids=None,
    ):
        self.model.eval()
        sampling_config = sampling_config or SamplingConfig()

        if memory is None:
            if audio is None:
                raise ValueError("audio or memory must be provided for decoding.")
            if audio.dim() == 2:
                audio = audio.unsqueeze(0)
            audio = audio.to(self.device)
            memory = self.model.encode_audio(audio)
        else:
            memory = memory.to(self.device)

        generated = torch.tensor([[self.bos_id]], dtype=torch.long, device=self.device)
        difficulty_values, density_values, beatmap_id_values = self._resolve_condition_values(
            difficulty=difficulty,
            density_nps=density_nps,
            beatmap_id=beatmap_id,
        )

        for _ in range(self.max_len):
            decode_input_ids, segment_ids = self._build_context_tensors(
                generated,
                recent_history_ids=recent_history_ids,
                retrieved_token_ids=retrieved_token_ids,
            )
            decoder_attention_mask = torch.ones_like(decode_input_ids, device=self.device)

            decode_kwargs = {
                "memory": memory,
                "input_ids": decode_input_ids,
                "decoder_attention_mask": decoder_attention_mask,
                "difficulty_values": difficulty_values,
                "density_values": density_values,
                "beatmap_id_values": beatmap_id_values,
            }
            if segment_ids is not None:
                decode_kwargs["segment_ids"] = segment_ids

            logits = self.model.decode_with_memory(**decode_kwargs)

            next_token_id = self._sample_next_token(
                logits_last=logits[:, -1, :],
                generated_ids=generated[0],
                sampling_config=sampling_config,
            )
            generated = torch.cat([generated, next_token_id], dim=1)

            if int(next_token_id.item()) == self.eos_id:
                break

        generated_ids = generated[0].tolist()

        if generated_ids and generated_ids[0] == self.bos_id:
            generated_ids = generated_ids[1:]
        if generated_ids and generated_ids[-1] == self.eos_id:
            generated_ids = generated_ids[:-1]

        generated_tokens = [self.id_to_token[i] for i in generated_ids]
        return generated_ids, generated_tokens

    @torch.no_grad()
    def greedy_decode(self, audio, difficulty=5.0, density_nps=6.0, beatmap_id=1_000_000):
        greedy_cfg = SamplingConfig(temperature=0.0, top_p=1.0, top_k=1, ts_top_k=0)
        return self.decode_sequence(
            audio=audio,
            difficulty=difficulty,
            density_nps=density_nps,
            beatmap_id=beatmap_id,
            sampling_config=greedy_cfg,
        )

    def preprocess_audio(self, audio_path, offset_ms, bpm, meter=4):
        timing_info = self.build_timing_info(
            audio_path=audio_path,
            offset_ms=offset_ms,
            bpm=bpm,
            meter=meter,
        )

        audio_key = (
            str(Path(timing_info["audio_path"]).resolve()),
            round(timing_info["offset_ms"], 3),
            round(timing_info["bpm"], 6),
            int(timing_info["meter"]),
        )

        def _compute_audio_sequences():
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

            return segment_aligned_mel_into_4beat_sequences(
                aligned_mel_db=aligned_mel_db,
                total_sequences=beat_grid_info.total_sequences,
            )

        return self._cache_get_or_compute(audio_key, _compute_audio_sequences)

    def generate_tokens(
        self,
        audio_path,
        offset_ms,
        bpm,
        meter=4,
        difficulty=5.0,
        density_nps=6.0,
        beatmap_id=1_000_000,
        sampling_config=None,
    ):
        return [
            item["pred_tokens"]
            for item in self.generate_song_structure(
                audio_path=audio_path,
                offset_ms=offset_ms,
                bpm=bpm,
                meter=meter,
                difficulty=difficulty,
                density_nps=density_nps,
                beatmap_id=beatmap_id,
                sampling_config=sampling_config,
            )
        ]

    def generate_tokens_with_ids(
        self,
        audio_path,
        offset_ms,
        bpm,
        meter=4,
        difficulty=5.0,
        density_nps=6.0,
        beatmap_id=1_000_000,
        sampling_config=None,
    ):
        song_output = self.generate_song_structure(
            audio_path=audio_path,
            offset_ms=offset_ms,
            bpm=bpm,
            meter=meter,
            difficulty=difficulty,
            density_nps=density_nps,
            beatmap_id=beatmap_id,
            sampling_config=sampling_config,
        )
        all_pred_ids = [item["pred_ids"] for item in song_output]
        all_pred_tokens = [item["pred_tokens"] for item in song_output]
        return all_pred_ids, all_pred_tokens

    def generate_song_structure(
        self,
        audio_path,
        offset_ms,
        bpm,
        meter=4,
        difficulty=5.0,
        density_nps=6.0,
        beatmap_id=1_000_000,
        sampling_config=None,
    ):
        audio_sequences = self.preprocess_audio(
            audio_path=audio_path,
            offset_ms=offset_ms,
            bpm=bpm,
            meter=meter,
        )

        song_output = []
        prior_window_token_ids = []
        prior_audio_keys = []

        for seq_idx in range(len(audio_sequences)):
            audio_seq = torch.tensor(audio_sequences[seq_idx], dtype=torch.float32)
            decode_kwargs = {
                "audio": audio_seq,
                "difficulty": difficulty,
                "density_nps": density_nps,
                "beatmap_id": beatmap_id,
                "sampling_config": sampling_config,
            }

            if self.context_enabled:
                audio_batch = audio_seq.unsqueeze(0).to(self.device)
                memory = self.model.encode_audio(audio_batch)
                current_audio_key = self._pool_audio_key_from_memory(memory)
                recent_history_ids = self._build_recent_history_ids(prior_window_token_ids)
                retrieved_ids = self._build_retrieved_ids(
                    current_audio_key,
                    prior_audio_keys,
                    prior_window_token_ids,
                )
                decode_kwargs["memory"] = memory
                decode_kwargs["recent_history_ids"] = recent_history_ids
                decode_kwargs["retrieved_token_ids"] = retrieved_ids

            pred_ids, pred_tokens = self.decode_sequence(
                **decode_kwargs,
            )

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

            if self.context_enabled:
                prior_window_token_ids.append(pred_ids + [self.eos_id])
                prior_audio_keys.append(current_audio_key.detach().clone())

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
