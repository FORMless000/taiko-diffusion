# Webapp-Wired AR Models Methodology Summary

This document summarizes the non-diffusion model methodologies currently wired into the web app:

- `sample_large_baseline`
- `sample_large_baseline_maxopt`
- `baseline_maxopt_step_055000`
- `baseline_snapshot_maxopt`
- `sample_large_context`

The diffusion refiner and hybrid diffusion wiring are intentionally out of scope for this summary.

## Project Framing

This project targets automatic osu!taiko beatmap generation from music. In the current non-diffusion system, the core idea is to convert both audio and beatmaps into a shared beat-aligned representation, then train an autoregressive transformer to generate chart tokens from audio features.

At a high level, the end-to-end pipeline is:

1. Parse `.osu` / `.osz` beatmaps into note, timing, and metadata JSON.
2. Convert audio into mel-spectrogram features.
3. Interpolate those features onto a beat-aligned grid at `48` frames per beat.
4. Split the aligned signal into fixed 4-beat windows with shape `(192, 128)`.
5. Serialize note timing and note types into tokens such as `TS_*`, `DON`, `KAT`, `BIGDON`, `BIGKAT`, `DRUMROLL`, `SLIDERSTART`, and `SLIDEREND`.
6. Train a transformer that predicts those token sequences from the corresponding audio windows.
7. During inference, decode chart tokens window by window and reconstruct them back into `.osu` / `.osz` artifacts.

The web app exposes several non-diffusion models because the repo explores multiple variants of this same basic idea: a shared transformer baseline, metadata-conditioned generation, context-aware generation, and training/data optimizations.

## Shared Autoregressive Transformer Baseline

The baseline architecture is implemented in [src/model/model.py](C:/Users/28548/PythonNotebooks/taiko-diffusion/src/model/model.py). It is a transformer encoder-decoder model:

- The encoder consumes beat-aligned audio for one 4-beat window.
- The decoder consumes previously generated chart tokens under a causal mask.
- The output is the next token in the chart sequence.

This matters because the system is not just predicting note labels per frame. It is generating a structured sequence that jointly represents:

- note type
- note ordering
- note timing deltas through `TS_*` tokens

So a good way to describe the baseline is: the model listens to a short chunk of music and then writes the chart one token at a time.

### Architecture Shape

The baseline model contains:

- an audio embedding layer that projects the 128-bin mel features into model space
- sinusoidal positional encoding for audio
- a transformer encoder over the audio sequence
- a token embedding layer for chart tokens
- sinusoidal positional encoding for tokens
- a transformer decoder that attends autoregressively to prior tokens and cross-attends to encoded audio
- a linear output head over the vocabulary

The standard training windows are 4 beats long, which corresponds to:

- `48` frames per beat
- `192` frames per sequence
- `128` mel bins

This gives a clean fixed-size training representation while still preserving beat-relative timing structure.

### Token Representation

Timing is represented explicitly through `TS_*` tokens instead of forcing the model to infer absolute note times directly. That gives the decoder a vocabulary over relative rhythmic structure. Event tokens then represent playable taiko objects such as `DON`, `KAT`, large hits, and slider/drumroll boundaries.

This tokenization makes the problem more like language modeling over rhythmic events rather than direct continuous-time regression.

### Song-Level Inference Loop

At inference time, generation proceeds window by window. The generator in [src/model/generation.py](C:/Users/28548/PythonNotebooks/taiko-diffusion/src/model/generation.py) decodes one token at a time for each 4-beat chunk, using the encoded audio memory for that chunk. The generated tokens are then stitched together into a full song output and reconstructed into a beatmap.

The inference path also supports sampling controls such as:

- temperature
- top-k
- top-p
- repetition penalty

and special handling to keep both note-event tokens and `TS_*` timing tokens active during decoding.

## Additions To The Transformer AR Model

One of the biggest methodological additions over a minimal audio-to-token transformer is explicit chart conditioning. The model is not only conditioned on audio; it also receives chart-level scalar signals intended to control the style and density of the generated output.

These are implemented in [src/model/model.py](C:/Users/28548/PythonNotebooks/taiko-diffusion/src/model/model.py) and normalized in [src/model/data.py](C:/Users/28548/PythonNotebooks/taiko-diffusion/src/model/data.py).

### Difficulty Encoding

The repo supports a `difficulty_value` conditioning channel. During preprocessing and dataset loading, a difficulty-like scalar is inferred or read from metadata, normalized, and supplied to the model.

Inside the transformer, difficulty is combined with the other conditioning channels into a 3-value vector that is projected through `condition_proj` and then added to the decoder token representations through a learned `condition_gate`.

Conceptually, this gives the model a global hint for how hard the chart should feel.

### Density Encoding

The repo also supports `density_nps`, a notes-per-second style density feature. This is a more direct proxy for how busy the chart is. It is normalized and injected into the decoder the same way as difficulty.

This channel is useful because density is closely related to mapping feel even when traditional difficulty labels are noisy or inconsistent.

### Style Encoding / Embedding

The current code does not implement a clean categorical “style embedding” for mapper identity in the usual sense. The closest thing the repo has to a style signal is `beatmap_id`, which is normalized with a log-like transform and fed into the same conditioning projection as difficulty and density.

So the most honest presentation description is:

- the architecture includes a beatmap-ID-based identity/style proxy
- it is used as a lightweight conditioning signal for chart tendencies
- it is not a separate human-interpretable mapper-style label system

This still matters methodologically, because it gives the decoder an extra bias toward chart-specific patterns beyond what the audio alone explains.

### How These Controls Are Injected

The conditioning flow is simple but important:

1. Normalize `difficulty_value`, `density_nps`, and `beatmap_id`.
2. Stack them into a 3D conditioning vector.
3. Project them through `condition_proj`.
4. Add the resulting vector to every decoder token embedding using a learned gate.

This means the conditioning acts as a global decoder-side context signal for the entire sequence.

### Important Limitation

There is an important caveat in the current preprocessing implementation. In [src/preprocessing/beat_aligned_dataset.py](C:/Users/28548/PythonNotebooks/taiko-diffusion/src/preprocessing/beat_aligned_dataset.py), the beat-aligned dataset builder currently sets:

`difficulty_value = density_nps`

So while the architecture is designed to support separate difficulty and density channels, the current generated training metadata does not fully disentangle them. That is a valuable point to mention in a presentation:

- the model design supports richer conditioning
- the current dataset wiring only partially realizes that potential

## The Context Model

The context model is implemented in [src/model/taiko_context.py](C:/Users/28548/PythonNotebooks/taiko-diffusion/src/model/taiko_context.py). It extends the same audio-conditioned autoregressive transformer, but changes what the decoder sees during generation.

The baseline model mostly treats each 4-beat window as a local problem. That is efficient and practical, but it also creates a weakness: music has recurring phrases, motifs, and longer-range structure that often span multiple windows. A purely local decoder can generate believable short segments while still missing larger-scale consistency.

The context model is designed to address that.

### What It Adds

The context model introduces three decoder token regions:

- recent history tokens
- retrieved motif tokens
- current window tokens

It uses `segment_embed` to mark which region each token belongs to. This gives the model structured long-context information instead of one flat token stream.

### Recent History

Recent history gives the model direct access to tokens from earlier windows in the same chart. This helps maintain continuity across local boundaries and makes it easier to keep patterns coherent across adjacent windows.

### Retrieved Motifs

The more interesting addition is motif retrieval. The model pools encoded audio windows into compact keys, compares the current window to earlier windows, and retrieves the most similar prior windows as extra context.

This gives the decoder access to earlier patterns that may match the current musical material. In effect, it is a way of saying:

- “this part of the song sounds like that earlier part”
- “the charting pattern used there may be relevant again here”

That is a strong inductive bias for rhythm game charting, where repeated musical phrases often deserve repeated or related note patterns.

### Excluding Recent Windows

The retrieval system excludes the most recent few windows before matching. This is important because otherwise the model could simply retrieve the immediate local past and copy it trivially. By excluding recent windows, the retrieval mechanism is pushed toward actual motif recurrence rather than short-range leakage.

### What Problems The Context Model Tries To Solve

The context model is best understood as an attempt to improve:

- phrase continuity across windows
- motif recurrence and reuse
- local pattern consistency
- musical structure awareness without requiring full-song decoding

This is a more targeted and efficient solution than simply increasing decoder length everywhere.

### Key Difference From The Baseline

The core architectural difference is not the audio encoder or the basic autoregressive setup. It is that the context model changes the decoder input from:

- “just current-window generated tokens”

to:

- “recent history + retrieved motif hints + current window tokens”

with segment embeddings to distinguish them.

That makes `sample_large_context` the webapp’s main experiment in long-range musical coherence.

## Training And Data Optimizations For The Transformer AR Family

Another major part of the methodology is not in the network alone, but in how the training data and runtime are engineered.

### Constant-BPM Taiko-Only Filtering

The preprocessing pipeline in [src/preprocessing/prepare_training_data.py](C:/Users/28548/PythonNotebooks/taiko-diffusion/src/preprocessing/prepare_training_data.py) filters charts aggressively:

- only osu!taiko charts are kept
- charts with BPM changes are rejected
- malformed or unsupported inputs are filtered out

This narrows the problem to a more stable setting. It is a limitation, but also a deliberate simplification that makes beat alignment and tokenization much cleaner.

### Large-Sample Vs Maxopt Data Curation

The “maxopt” variants use `keep_only_max_notes_per_song`, which is wired through preprocessing and training-data construction. This keeps only the chart with the highest model-note count per song.

The likely motivation is:

- reduce multiple competing chart labels for the same song
- keep the densest supervision target
- simplify the mapping distribution during training

This does not change the model architecture, but it does change what the model learns from.

### Snapshot Dataset Construction

The snapshot pipeline in [src/preprocessing/build_snapshot_dataset.py](C:/Users/28548/PythonNotebooks/taiko-diffusion/src/preprocessing/build_snapshot_dataset.py) creates a smaller curated dataset from a larger cache. This is the methodology behind `baseline_snapshot_maxopt`.

This is useful because it enables:

- faster experiments
- more reproducible subsets
- lower-cost training iterations

The tradeoff is that a smaller snapshot may capture less total variation than the full larger dataset.

### Cached Manifests, Splits, And Indexes

Training setup in [src/model/train_api.py](C:/Users/28548/PythonNotebooks/taiko-diffusion/src/model/train_api.py) caches chart manifests, split assignments, and sequence indexes. That saves repeated preprocessing work and makes reruns much cheaper and more reproducible.

This is an engineering optimization rather than a modeling change, but it is important for a practical training workflow.

### Precision And Runtime Optimizations

The runtime helpers in [src/model/runtime.py](C:/Users/28548/PythonNotebooks/taiko-diffusion/src/model/runtime.py) and the training path in [src/model/train_api.py](C:/Users/28548/PythonNotebooks/taiko-diffusion/src/model/train_api.py) support:

- `precision="auto"` with automatic selection of `bf16` when available on CUDA
- `fp16` autocast where appropriate
- gradient scaling for `fp16`
- `pin_memory`
- `persistent_workers`
- `prefetch_factor`

These are standard but meaningful training optimizations. They make the training runs more efficient without changing the learned architecture.

### Exported Inference Snapshots

Training can periodically export inference-ready checkpoints during optimization. That is how `baseline_maxopt_step_055000` can appear in the web app as a deployable model even though it is not simply a final `last.ckpt`. It is an exported intermediate inference snapshot from the same AR family.

This is a useful deployment idea:

- you do not have to wait for the very end of training to test a model in the app
- you can compare intermediate checkpoints directly in inference conditions

### Throughput And Adherence Metrics

The training loop in [src/model/trainer.py](C:/Users/28548/PythonNotebooks/taiko-diffusion/src/model/trainer.py) logs not only loss, but also runtime throughput and proxy adherence metrics such as density error and difficulty drift.

These are monitoring metrics rather than direct extra loss terms, but they help assess whether the model is behaving consistently with the intended conditioning scheme.

## Model-By-Model Comparison

### `sample_large_baseline`

This is the clean reference baseline:

- architecture: `taiko_transformer`
- training style: large-sample baseline run
- role: simplest audio-conditioned AR benchmark in the app

Its main strength is conceptual clarity. Its main weakness is that it mostly treats each 4-beat window as a local problem.

### `sample_large_baseline_maxopt`

This uses the same baseline AR architecture, but changes the training data curation with max-note-per-song filtering.

Its main intended benefit is cleaner supervision per song and reduced redundancy across multiple chart variants. The tradeoff is that it may bias training toward denser interpretations.

### `baseline_maxopt_step_055000`

This is not a fundamentally different model family. It is an exported inference snapshot of the same AR baseline family.

Its importance is methodological and operational rather than architectural:

- it represents checkpoint selection during training
- it lets the team evaluate an intermediate model state in the real app pipeline

### `baseline_snapshot_maxopt`

This keeps the same AR baseline model but trains it on a curated snapshot dataset with max-note-per-song filtering.

Its purpose is to support faster, more controlled experimentation while keeping the rest of the train/infer system consistent.

### `sample_large_context`

This is the dedicated context model:

- architecture: `taiko_context_transformer`
- key additions: recent history, motif retrieval, segment embeddings
- goal: improve cross-window coherence and phrase-level structure

This is the repo’s main architectural attempt to move beyond short-window local generation.

## Practical Takeaway

The non-diffusion models in the web app are best understood as a structured progression:

1. Start from a workable audio-to-token autoregressive transformer baseline.
2. Add chart-level conditioning to influence difficulty, density, and chart-identity/style tendencies.
3. Improve training efficiency and data quality through curation, snapshotting, caching, and precision/runtime tuning.
4. Extend the decoder with explicit history and motif retrieval so generation can better respect long-range musical structure.

That is the core methodological story of the current AR side of the project.

## TL;DR

The webapp’s non-diffusion models are all variants of an audio-conditioned autoregressive transformer that generates taiko chart tokens from beat-aligned 4-beat audio windows. The baseline model uses an encoder-decoder transformer with `TS_*` timing tokens and note-event tokens, while the main architectural additions are decoder-side conditioning for difficulty, density, and a beatmap-ID-based style proxy. The context model extends this by feeding the decoder recent history and retrieved similar past motifs, which is meant to improve phrase continuity and recurring pattern consistency across windows. The training-side optimizations include constant-BPM taiko-only filtering, max-note-per-song data curation, snapshot dataset construction, cached manifests and splits, mixed-precision/runtime tuning, and periodic export of inference-ready checkpoints. The reason there are multiple AR models in the app is that the repo is comparing different tradeoffs in data curation, efficiency, checkpoint choice, and long-context handling rather than only swapping out one completely different model for another.
