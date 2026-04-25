# Model Brief For Presentation

## Main Idea

The core model is an audio-conditioned autoregressive transformer for osu!taiko chart generation.

- Input: beat-aligned mel-spectrogram window `x in R^(192 x 128)`
- Output: token sequence `y = (y_1, ..., y_T)` containing timing tokens `TS_*` and event tokens such as `DON`, `KAT`, `BIGDON`, `BIGKAT`
- Goal: model the conditional distribution

`p(y | x) = product over t of p(y_t | y_<t, x)`

So the model listens to a short audio window, then writes the chart token by token.

## AR Transformer

The baseline is an encoder-decoder transformer.

- Audio encoder:
  - project mel features into model space
  - add positional encoding
  - encode with transformer self-attention
- Token decoder:
  - embed previous tokens
  - add positional encoding
  - apply causal self-attention
  - cross-attend to encoded audio
  - predict the next token

Compact math:

`H = Encoder(PosEnc(W_a x))`

`z_t = TokenEmbed(y_<t) + PosEnc + Conditioning`

`logits_t = Decoder(z_t, H)`

`p(y_t | y_<t, x) = softmax(logits_t)`

Training objective:

`L_AR = - sum over t of log p(y_t* | y_<t, x)`

This is standard teacher-forced cross-entropy over the target token sequence.

## Metadata Conditioning

The decoder also receives three global conditioning signals:

- difficulty
- density
- beatmap ID as a style / identity proxy

These are normalized, stacked, projected, and added to decoder token states:

`c = [difficulty, density, beatmap_id]`

`g = MLP(c)`

`z_t <- z_t + alpha g`

where `alpha` is a learned gate.

Interpretation:

- difficulty biases chart hardness
- density biases note rate
- beatmap ID acts as a weak style prior for mapper/chart tendencies

## Context Transformer Variation

The context model keeps the same AR backbone, but changes the decoder input context.

Instead of decoding from only the current window, it prepends:

- recent history tokens
- retrieved motif tokens from earlier similar windows
- current-window tokens

Segment embeddings tell the decoder which tokens come from which source.

This is meant to improve:

- cross-window continuity
- motif recurrence
- phrase-level consistency

## Retrieval Math

For each prior window, the model computes a pooled audio key from the encoded memory:

`k_i = normalize(mean(H_i))`

For the current window:

`q = normalize(mean(H_current))`

Similarity is cosine similarity:

`s_i = q^T k_i`

Then retrieve the top `K` prior windows with highest `s_i`, excluding the most recent few windows to avoid trivial copying.

The decoder input becomes:

`[history tokens ; retrieved motif tokens ; current tokens]`

So the context model is effectively using retrieval-augmented sequence modeling over earlier musical structure.

## Key Optimizations

### Data / Training Setup

- Beat-aligned representation at `48` frames per beat
- 4-beat training windows for stable fixed-size supervision
- constant-BPM taiko-only filtering
- optional `keep_only_max_notes_per_song` dataset curation
- snapshot datasets for faster experiments

### Runtime / Engineering

- cached manifests, splits, and sequence indexes
- mixed precision (`bf16` / `fp16` when available)
- tuned dataloader settings: `pin_memory`, workers, prefetch, persistent workers
- exported inference snapshots for easy deployment/testing during training

## One-Line Takeaway

The baseline model is an audio-to-token autoregressive transformer, and the main extension is a context-and-retrieval mechanism that helps the model reuse earlier musical/chart patterns while keeping training and deployment practical through strong data and runtime optimizations.
