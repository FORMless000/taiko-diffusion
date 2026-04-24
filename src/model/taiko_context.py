import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import TaikoTransformer, generate_causal_mask


class TaikoContextTransformer(TaikoTransformer):
    def __init__(
        self,
        vocab_size,
        input_dim=128,
        d_model=256,
        nhead=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_len=1536,
        history_max_tokens=256,
        retrieval_top_k=1,
        retrieval_max_tokens_per_window=24,
        retrieval_exclude_last_n_windows=2,
        use_motif_retrieval=True,
    ):
        super().__init__(
            vocab_size=vocab_size,
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )

        self.segment_embed = nn.Embedding(3, d_model)
        self.history_max_tokens = max(1, int(history_max_tokens))
        self.retrieval_top_k = max(0, int(retrieval_top_k))
        self.retrieval_max_tokens_per_window = max(1, int(retrieval_max_tokens_per_window))
        self.retrieval_exclude_last_n_windows = max(0, int(retrieval_exclude_last_n_windows))
        self.use_motif_retrieval = bool(use_motif_retrieval)
        self.supports_long_context = True

    def _pool_audio_memory(self, memory):
        pooled = memory.mean(dim=1)
        return F.normalize(pooled, dim=-1)

    def decode_with_memory(
        self,
        memory,
        input_ids,
        decoder_attention_mask=None,
        difficulty_values=None,
        density_values=None,
        beatmap_id_values=None,
        segment_ids=None,
    ):
        batch_size = input_ids.size(0)
        device = input_ids.device

        tok_x = self.token_embed(input_ids)
        tok_x = self.token_pos_enc(tok_x)

        if segment_ids is None:
            segment_ids = torch.full_like(input_ids, 2, device=device)
        else:
            segment_ids = segment_ids.to(device=device, dtype=torch.long)
        tok_x = tok_x + self.segment_embed(segment_ids)

        resolved_difficulty = self._resolve_condition_values(difficulty_values, batch_size, device)
        resolved_density = self._resolve_condition_values(density_values, batch_size, device)
        resolved_beatmap = self._resolve_condition_values(beatmap_id_values, batch_size, device)

        condition_input = torch.stack(
            [resolved_difficulty, resolved_density, resolved_beatmap],
            dim=-1,
        )
        condition_vec = self.condition_proj(condition_input)
        tok_x = tok_x + self.condition_gate * condition_vec.unsqueeze(1)

        length = input_ids.size(1)
        tgt_mask = generate_causal_mask(length, device=device)

        tgt_key_padding_mask = None
        if decoder_attention_mask is not None:
            tgt_key_padding_mask = decoder_attention_mask == 0

        dec_out = self.chart_decoder(
            tgt=tok_x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.output_head(dec_out)

    def forward(
        self,
        audio,
        input_ids,
        decoder_attention_mask=None,
        difficulty_values=None,
        density_values=None,
        beatmap_id_values=None,
        segment_ids=None,
    ):
        memory = self.encode_audio(audio)
        return self.decode_with_memory(
            memory=memory,
            input_ids=input_ids,
            decoder_attention_mask=decoder_attention_mask,
            difficulty_values=difficulty_values,
            density_values=density_values,
            beatmap_id_values=beatmap_id_values,
            segment_ids=segment_ids,
        )
