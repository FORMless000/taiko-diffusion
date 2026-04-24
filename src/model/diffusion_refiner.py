import torch
import torch.nn as nn


class TaikoDiffusionRefiner(nn.Module):
    """Checkpoint-compatible masked-token refiner from the diffusion notebook."""

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
        max_len=2048,
    ):
        super().__init__()

        # Keep module names aligned with the notebook so exported checkpoints load cleanly.
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_decoder = nn.Parameter(torch.randn(1, max_len, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        self.output_layer = nn.Linear(d_model, vocab_size)

    def encode_audio(self, audio):
        encoded = self.input_proj(audio)
        encoded = encoded + self.pos_encoder[:, : encoded.size(1), :]
        return self.encoder(encoded)

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
        del difficulty_values, density_values, beatmap_id_values, segment_ids

        tok_x = self.token_embed(input_ids)
        tok_x = tok_x + self.pos_decoder[:, : tok_x.size(1), :]

        tgt_key_padding_mask = None
        if decoder_attention_mask is not None:
            tgt_key_padding_mask = decoder_attention_mask == 0

        decoded = self.decoder(
            tgt=tok_x,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.output_layer(decoded)

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
