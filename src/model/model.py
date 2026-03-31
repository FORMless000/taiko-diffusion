import math
import torch
import torch.nn as nn


class AudioEmbedding(nn.Module):
    def __init__(self, input_dim=128, d_model=256):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)

    def forward(self, x):
        return self.proj(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        t = x.size(1)
        return x + self.pe[:, :t, :]


class AudioEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, input_ids):
        return self.embed(input_ids)


class ChartDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        return self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )


def generate_causal_mask(seq_len, device=None):
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


class OutputHead(nn.Module):
    def __init__(self, d_model=256, vocab_size=83):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.proj(x)


class TaikoTransformer(nn.Module):
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
        max_len=512,
    ):
        super().__init__()

        self.audio_embed = AudioEmbedding(input_dim=input_dim, d_model=d_model)
        self.audio_pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.audio_encoder = AudioEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.token_embed = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.token_pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len)

        self.condition_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.condition_gate = nn.Parameter(torch.tensor(1.0))

        self.chart_decoder = ChartDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.output_head = OutputHead(d_model=d_model, vocab_size=vocab_size)

    def _resolve_condition_values(self, values, batch_size, device):
        if values is None:
            return torch.full((batch_size,), 0.5, dtype=torch.float32, device=device)
        if values.dim() == 0:
            return values.expand(batch_size).to(device=device, dtype=torch.float32)
        if values.dim() == 2 and values.size(1) == 1:
            return values[:, 0].to(device=device, dtype=torch.float32)
        return values.to(device=device, dtype=torch.float32)

    def encode_audio(self, audio):
        x = self.audio_embed(audio)
        x = self.audio_pos_enc(x)
        return self.audio_encoder(x)

    def decode_with_memory(
        self,
        memory,
        input_ids,
        decoder_attention_mask=None,
        difficulty_values=None,
        density_values=None,
        beatmap_id_values=None,
    ):
        batch_size = input_ids.size(0)
        device = input_ids.device

        tok_x = self.token_embed(input_ids)
        tok_x = self.token_pos_enc(tok_x)

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
        tgt_mask = generate_causal_mask(length, device=input_ids.device)

        tgt_key_padding_mask = None
        if decoder_attention_mask is not None:
            tgt_key_padding_mask = decoder_attention_mask == 0

        dec_out = self.chart_decoder(
            tgt=tok_x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        logits = self.output_head(dec_out)
        return logits

    def forward(
        self,
        audio,
        input_ids,
        decoder_attention_mask=None,
        difficulty_values=None,
        density_values=None,
        beatmap_id_values=None,
    ):
        memory = self.encode_audio(audio)
        return self.decode_with_memory(
            memory=memory,
            input_ids=input_ids,
            decoder_attention_mask=decoder_attention_mask,
            difficulty_values=difficulty_values,
            density_values=density_values,
            beatmap_id_values=beatmap_id_values,
        )
