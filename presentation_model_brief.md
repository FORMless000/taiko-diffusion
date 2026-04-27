# Model Brief For Presentation

## Main Idea

The core model is an audio-conditioned autoregressive transformer for osu!taiko chart generation.

- Input: beat-aligned mel-spectrogram window $X \in \mathbb{R}^{192 \times 128}$
- Output: token sequence $\mathbf{y} = (y_1, \dots, y_T)$ containing timing tokens `TS_*` and event tokens such as `DON`, `KAT`, `BIGDON`, `BIGKAT`
- Goal: model the conditional distribution

$$
p(\mathbf{y} \mid X) = \prod_{t=1}^{T} p(y_t \mid y_{<t}, X)
$$

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

$$
H = \mathrm{Enc}(\mathrm{PE}(X W_a))
$$

$$
E_{<t} = \mathrm{PE}(\mathrm{TokEmb}(y_{<t}))
$$

$$
h_t = \mathrm{Dec}(E_{<t}, H)
$$

$$
p(y_t \mid y_{<t}, X) = \mathrm{softmax}(W_o h_t + b_o)
$$

Training objective:

$$
\mathcal{L}_{\mathrm{AR}} = - \sum_{t=1}^{T} \log p(y_t^{\ast} \mid y_{<t}^{\ast}, X)
$$

This is standard teacher-forced cross-entropy over the target token sequence.

## Metadata Conditioning

The decoder also receives three global conditioning signals:

- difficulty
- density
- beatmap ID as a style / identity proxy

Let:

- $d_{\mathrm{diff}}$ be the difficulty number after scaling it into a range the model can use
- $d_{\mathrm{dens}}$ be the density number after scaling it into a range the model can use
- $d_{\mathrm{id}}$ be the beatmap-ID number after scaling it, used as a rough style hint
- $\mathbf{c}$ be the 3-number column made by putting those values together
- $\phi(\cdot)$ be the small neural network that turns those 3 numbers into a richer feature
- $\mathbf{g}$ be the output of that small neural network, meaning the final metadata feature the model uses
- $E_{<t}$ be the decoder’s token representation before metadata is added
- $\tilde{E}_{<t}$ be the decoder’s token representation after metadata is added
- $\alpha$ be a learned number that controls how strongly the metadata affects the decoder

These are normalized, stacked, projected, and added to decoder token states:

$$
\mathbf{c} = [d_{\mathrm{diff}}, d_{\mathrm{dens}}, d_{\mathrm{id}}]^{\top}
$$

$$
\mathbf{g} = \phi(\mathbf{c})
$$

$$
\tilde{E}_{<t} = E_{<t} + \alpha \mathbf{g}
$$

Interpretation of the symbols:

- $\mathbf{c} \in \mathbb{R}^{3}$ just means “a list of 3 real numbers”
- $\mathbf{g}$ is the model’s learned summary of the metadata
- the same $\mathbf{g}$ is copied to every decoder step, because the whole chart window shares the same metadata
- $\alpha$ lets the model decide whether metadata should matter a little or a lot

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

Define the symbols locally:

- $H_i$ is the encoder output for an earlier audio window $i$
- $H_{\mathrm{cur}}$ is the encoder output for the current audio window the model is working on now
- $\mathrm{mean}(H_i)$ means “average all the time-step vectors in window $i$ into one summary vector”
- $\mathbf{k}_i$ is the stored lookup vector for earlier window $i$
- $\mathbf{q}$ is the lookup vector for the current window
- $s_i$ is the similarity score between the current window and earlier window $i$
- $K$ is how many earlier windows we keep after retrieval
- $\mathbf{z}_{\mathrm{hist}}$ is the block of token representations from recent chart history
- $\mathbf{z}_{\mathrm{retr}}$ is the block of token representations copied from retrieved earlier motifs
- $\mathbf{z}_{\mathrm{cur}}$ is the block of token representations for the current part being decoded
- $\mathbf{z}_{\mathrm{ctx}}$ is the full decoder input after joining history, retrieval, and current tokens together

For each prior window, the model computes a pooled audio key from the encoded memory:

$$
\mathbf{k}_i = \frac{\mathrm{mean}(H_i)}{\lVert \mathrm{mean}(H_i) \rVert_2}
$$

For the current window:

$$
\mathbf{q} = \frac{\mathrm{mean}(H_{\mathrm{cur}})}{\lVert \mathrm{mean}(H_{\mathrm{cur}}) \rVert_2}
$$

Similarity is cosine similarity:

$$
s_i = \mathbf{q}^{\top} \mathbf{k}_i
$$

Interpretation of the symbols:

- $\mathbf{k}_i$ is the “saved fingerprint” for earlier window $i$
- $\mathbf{q}$ is the “current fingerprint” used to search for similar earlier windows
- $s_i$ gets larger when the current window and earlier window look more alike
- this is cosine similarity because both vectors are scaled to length 1 before comparison

Then retrieve the top $K$ prior windows with highest $s_i$, excluding the most recent few windows to avoid trivial copying.

The decoder input becomes:

$$
\mathbf{z}_{\mathrm{ctx}} = [\mathbf{z}_{\mathrm{hist}} ; \mathbf{z}_{\mathrm{retr}} ; \mathbf{z}_{\mathrm{cur}}]
$$

Here $[\cdot ; \cdot ; \cdot]$ denotes concatenation along the token dimension.

In simpler words, this means the decoder reads:

- some recent chart history
- some retrieved old patterns that match the current music
- the current generation region

all as one long input sequence.

So the context model is effectively using retrieval-augmented sequence modeling over earlier musical structure.

## Key Optimizations

### Data / Training Setup

- Beat-aligned representation at `48` frames per beat
  - This turns raw audio into a rhythm-aware grid, so the model sees music in units that line up with beats instead of arbitrary waveform chunks.
  - That makes the token timing task much easier, because note placement is learned relative to beat structure.
  - In practice, this is a problem simplification step: instead of learning timing directly from raw continuous audio, the model learns timing on a normalized beat grid.
- 4-beat training windows
  - The repo uses short fixed windows with shape `(192, 128)`.
  - This keeps the problem computationally manageable and gives stable fixed-size supervision for the AR model.
  - Short windows also reduce sequence length inside the transformer, which lowers memory cost and speeds up both training and inference.
- constant-BPM taiko-only filtering
  - Charts with BPM changes are removed from the main training/inference path.
  - This simplifies beat alignment, timing-token generation, and reconstruction, reducing noise in the training data.
  - The preprocessing code uses a fast `.osu` screen before full parsing, so obviously ineligible charts can be skipped early instead of wasting time on expensive processing.
- optional `keep_only_max_notes_per_song` curation
  - When multiple charts exist for the same song, this option keeps only the densest chart.
  - The idea is to reduce conflicting supervision from many alternate difficulties and focus training on one strong target per song.
  - This is especially useful when many charts of one song are highly similar but differ in density. Keeping all of them can blur the target behavior the model should learn.
- snapshot datasets for faster iteration
  - Instead of always training on the full eligible pool, the repo can build a curated snapshot subset.
  - This makes experiments cheaper and faster while still using the same preprocessing and model code.
  - The snapshot builder is not just random copying. It first screens set folders for eligibility, including:
    - exactly one audio file
    - bounded audio size
    - taiko mode only
    - constant BPM
    - complete parsed chart triples
  - After that, it samples a reproducible subset using a fixed seed and writes a selection manifest, which makes experiments much easier to repeat later.
- off-grid note rejection during dataset build
  - The preprocessing path can reject notes that do not land close enough to the beat-aligned grid, using a configurable tolerance in milliseconds.
  - This is a data-cleaning optimization: it prevents the model from learning from timing examples that do not fit the grid representation well.

### Runtime / Engineering

- cached manifests, splits, and sequence indexes
  - The repo caches chart manifests and train/validation/test sequence indexes.
  - This avoids repeating expensive bookkeeping every time a new run starts, which speeds up experimentation and improves reproducibility.
  - The cache is keyed by a dataset signature that includes file fingerprints, split seed, split ratios, and architecture mode. In other words, the code only reuses cached indexes when it is safe to do so.
  - This is a practical optimization for notebook-driven experimentation, where startup time can otherwise become a real bottleneck.
- mixed precision (`bf16` / `fp16` when available)
  - Training can automatically use lower-precision math on supported hardware.
  - This reduces memory use and often increases throughput, allowing larger batches or faster epochs without changing the model design.
  - The runtime resolves precision automatically: on CUDA it prefers `bf16` when available, otherwise falls back to `fp16`, and on non-CUDA devices it falls back to `fp32`.
  - `fp16` training uses gradient scaling, while `bf16` can usually avoid that extra stabilization step.
- tuned dataloader settings
  - Settings such as `pin_memory`, worker count, prefetching, and persistent workers are exposed and used in optimized runs.
  - These are not model changes, but they matter because slow data loading can bottleneck GPU training.
  - The runtime also chooses sensible defaults based on device type. For example, `pin_memory` is automatically useful on CUDA but not on CPU-only runs.
  - This is a good example of engineering optimization: even a good model can train slowly if the GPU is waiting for data.
- context-budget controls for the context transformer
  - The context model exposes extra runtime controls such as:
    - `history_max_tokens`
    - `retrieval_top_k`
    - `retrieval_max_tokens_per_window`
    - `retrieval_exclude_last_n_windows`
    - `max_cached_charts`
  - These are important because long-context decoding can become expensive quickly. The code keeps the retrieval/context budget explicit so memory and speed stay manageable.
- inference-ready bundle export during training
  - The training path can save standard checkpoints such as `last.ckpt` and `best.ckpt`, but it can also export inference bundles at intermediate steps.
  - Those inference snapshots contain the model weights plus the architecture spec and vocab needed for direct loading in the actual generation path.
  - This is operationally valuable because the team can test partially trained models in the same inference stack the web app uses, instead of needing a separate conversion step every time.
- exported inference snapshots during training
  - The training loop can periodically save inference-ready checkpoints instead of only saving the final model.
  - This lets the team test intermediate models directly in the real inference path and compare quality before the end of training.
- throughput and adherence monitoring
  - The repo tracks runtime measures such as tokens-per-second as well as proxy adherence metrics such as density error and difficulty drift.
  - These diagnostics help evaluate not just whether loss is decreasing, but whether the conditioning behavior is staying aligned with the intended chart properties.
  - This matters for a conditioned generation model because a low loss alone does not guarantee that the difficulty and density controls are actually being respected.
  - The training loop therefore logs both optimization metrics and behavior metrics:
    - optimization metrics: loss, batch time, samples per second, tokens per second
    - behavior metrics: density proxy absolute error and difficulty proxy drift
- resumable, reproducible training state
  - Checkpoints store more than just model weights. They also store optimizer state, RNG state, training specs, architecture specs, split IDs, vocab, and artifact paths.
  - This is an engineering optimization for research workflow: it makes it much easier to resume experiments faithfully and avoid "it worked before but I cannot reproduce it" problems.

### Why These Optimizations Matter

These optimizations matter because the project is not only about designing a model, but also about making the full training-and-inference workflow practical.

- Data filtering reduces label noise and keeps the task well-defined.
- Windowing and beat alignment make the learning problem simpler.
- Snapshot datasets and maxopt curation make experiments faster and more controlled.
- Mixed precision and dataloader tuning improve hardware efficiency.
- Cached indexes and inference snapshots reduce iteration cost.
- Richer checkpoint contents and runtime metrics make experiments easier to resume, compare, and audit.

So the optimization story is not just "train faster." It is:

- make the task simpler through representation choices
- make the supervision cleaner through filtering and curation
- make experiments cheaper through caching and snapshots
- make hardware usage better through precision and dataloader tuning
- make results easier to trust through reproducible checkpoints and behavior monitoring

## One-Line Takeaway

The baseline model is an audio-to-token autoregressive transformer, and the main extension is a context-and-retrieval mechanism that helps the model reuse earlier musical/chart patterns while keeping training and deployment practical through strong data and runtime optimizations.
