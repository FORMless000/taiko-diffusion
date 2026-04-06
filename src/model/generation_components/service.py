from __future__ import annotations

import torch

from src.model.data import (
    preprocess_beatmap_id,
    preprocess_density_nps,
    preprocess_difficulty_value,
)

from .audio import CachedAudioPreprocessor
from .sampling import SamplingConfig, class_aware_candidate_ids, sample_next_token


class TaikoBeatmapGenerator:
    def __init__(self, model, token_to_id, id_to_token, device, max_len=64, audio_cache_size=8):
        self.model = model
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.device = device
        self.max_len = max_len
        self.audio_preprocessor = CachedAudioPreprocessor(audio_cache_size=audio_cache_size)

        self.pad_id = token_to_id.get("PAD")
        self.bos_id = token_to_id["BOS"]
        self.eos_id = token_to_id["EOS"]

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
        return self.audio_preprocessor.build_timing_info(audio_path, offset_ms, bpm, meter)

    def _cache_get_or_compute(self, cache_key, compute_fn):
        return self.audio_preprocessor.cache_get_or_compute(cache_key, compute_fn)

    def _resolve_condition_values(self, difficulty=5.0, density_nps=6.0, beatmap_id=1_000_000):
        difficulty_norm = preprocess_difficulty_value(difficulty)
        density_norm = preprocess_density_nps(density_nps)
        beatmap_id_norm = preprocess_beatmap_id(beatmap_id)

        difficulty_values = torch.tensor([difficulty_norm], dtype=torch.float32, device=self.device)
        density_values = torch.tensor([density_norm], dtype=torch.float32, device=self.device)
        beatmap_id_values = torch.tensor([beatmap_id_norm], dtype=torch.float32, device=self.device)
        return difficulty_values, density_values, beatmap_id_values

    def _class_aware_candidate_ids(self, logits_1d, top_k, ts_top_k, min_event_candidates):
        return class_aware_candidate_ids(
            logits_1d,
            event_token_ids_tensor=self.event_token_ids_tensor,
            ts_token_ids_tensor=self.ts_token_ids_tensor,
            eos_id=self.eos_id,
            top_k=top_k,
            ts_top_k=ts_top_k,
            min_event_candidates=min_event_candidates,
        )

    def _sample_next_token(self, logits_last, generated_ids, sampling_config):
        return sample_next_token(
            logits_last,
            generated_ids,
            sampling_config,
            event_token_ids_tensor=self.event_token_ids_tensor,
            ts_token_ids_tensor=self.ts_token_ids_tensor,
            eos_id=self.eos_id,
        )

    @torch.no_grad()
    def decode_sequence(
        self,
        audio,
        difficulty=5.0,
        density_nps=6.0,
        beatmap_id=1_000_000,
        sampling_config=None,
    ):
        self.model.eval()
        sampling_config = sampling_config or SamplingConfig()

        if audio.dim() == 2:
            audio = audio.unsqueeze(0)

        audio = audio.to(self.device)
        memory = self.model.encode_audio(audio)

        generated = torch.tensor([[self.bos_id]], dtype=torch.long, device=self.device)
        difficulty_values, density_values, beatmap_id_values = self._resolve_condition_values(
            difficulty=difficulty,
            density_nps=density_nps,
            beatmap_id=beatmap_id,
        )

        for _ in range(self.max_len):
            decoder_attention_mask = torch.ones_like(generated, device=self.device)
            logits = self.model.decode_with_memory(
                memory=memory,
                input_ids=generated,
                decoder_attention_mask=decoder_attention_mask,
                difficulty_values=difficulty_values,
                density_values=density_values,
                beatmap_id_values=beatmap_id_values,
            )

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
        return self.audio_preprocessor.preprocess_audio(audio_path, offset_ms, bpm, meter)

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
        audio_sequences = self.preprocess_audio(
            audio_path=audio_path,
            offset_ms=offset_ms,
            bpm=bpm,
            meter=meter,
        )

        all_pred_tokens = []

        for seq_idx in range(len(audio_sequences)):
            audio_seq = torch.tensor(audio_sequences[seq_idx], dtype=torch.float32)
            _, pred_tokens = self.decode_sequence(
                audio=audio_seq,
                difficulty=difficulty,
                density_nps=density_nps,
                beatmap_id=beatmap_id,
                sampling_config=sampling_config,
            )
            all_pred_tokens.append(pred_tokens)

        return all_pred_tokens

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
            pred_ids, pred_tokens = self.decode_sequence(
                audio=audio_seq,
                difficulty=difficulty,
                density_nps=density_nps,
                beatmap_id=beatmap_id,
                sampling_config=sampling_config,
            )
            all_pred_ids.append(pred_ids)
            all_pred_tokens.append(pred_tokens)

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

        for seq_idx in range(len(audio_sequences)):
            audio_seq = torch.tensor(audio_sequences[seq_idx], dtype=torch.float32)
            pred_ids, pred_tokens = self.decode_sequence(
                audio=audio_seq,
                difficulty=difficulty,
                density_nps=density_nps,
                beatmap_id=beatmap_id,
                sampling_config=sampling_config,
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

        return song_output
