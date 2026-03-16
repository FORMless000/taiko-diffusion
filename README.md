# taiko-diffusion

Automatic osu!taiko beatmap generation from music.

## Summary

This project is building toward a system that can generate osu!taiko beatmaps directly from audio using a transformer + diffusion pipeline.

Right now, the repository contains:

- a preprocessing pipeline that parses `.osu` / `.osz` beatmaps into training-friendly JSON
- a beat-aligned dataset builder that converts audio + notes into fixed 4-beat training sequences
- an initial transformer baseline for token generation conditioned on beat-aligned audio features

Diffusion is still the long-term target rather than the current implemented model.

## Goals

The project goal is to generate osu!taiko beatmaps from music using transformer + diffusion models.

In practice, the current roadmap looks like this:

1. Parse ranked osu!taiko beatmaps into reusable training artifacts.
2. Build a beat-aligned paired dataset of audio features and tokenized note events.
3. Train a transformer baseline that predicts note-token sequences from audio.
4. Use that baseline and dataset design to support a stronger diffusion-based generator later.

## Data

Dataset:

- We used all 10,048 ranked beatmap sets for osu!taiko, downloaded using `batch-beatmap-downloader`: https://github.com/nzbasic/batch-beatmap-downloader
- Downloaded dataset is available at https://www.dropbox.com/scl/fo/ipbgqy80vnithcbhjp33f/ALzhBrzJm013t9gtPpoa9VU?rlkey=kqo1ia6iq6zo1ep4j9usybg1u&st=unov6n5l&dl=0
- Manual download of each beatmap set is avaiilable at https://osu.ppy.sh/beatmapsets?m=1&s=ranked 
- Only ranked beatmap sets were chosen because they are quality-assured by the community.

Current repository data flow:

1. Raw `.osz` archives are unpacked.
2. Each taiko chart is parsed into intermediate files:
   - `*.notes.json`
   - `*.timing.json`
   - `*.metadata.json`
3. Audio is converted into mel-spectrogram features.
4. Features are interpolated onto a beat-aligned grid with `48` frames per beat.
5. The aligned audio is split into 4-beat windows with shape `(192, 128)`.
6. Notes inside each 4-beat window are serialized into token sequences such as `TS_24`, `DON`, `KAT`, `BIGDON`, `BIGKAT`, `DRUMROLL`, `SLIDERSTART`, and `SLIDEREND`.

The beat-aligned dataset builder writes:

- `beat_aligned_dataset/audio_npz/*.npz`
- `beat_aligned_dataset/token_json/*.json`
- `beat_aligned_dataset/sequence_metadata.csv`
- chart-level summaries under `chart_index/`

## Current Progress

The project has moved beyond a minimal parser scaffold.

- `data_preprocessing.ipynb` demonstrates parsing and reconstruction of osu!taiko charts.
- [`beat_aligned_dataset.ipynb`](/c:/Users/28548/PythonNotebooks/taiko-diffusion/beat_aligned_dataset.ipynb) implements the full beat-aligned dataset pipeline.
- [`otsu_Transformer.ipynb`](/c:/Users/28548/PythonNotebooks/taiko-diffusion/otsu_Transformer.ipynb) implements an encoder-decoder transformer baseline for sequence prediction and sample generation.
- The shared logic has also been moved into reusable Python modules under [`src/preprocessing`](/c:/Users/28548/PythonNotebooks/taiko-diffusion/src/preprocessing) and [`src/model`](/c:/Users/28548/PythonNotebooks/taiko-diffusion/src/model).

From the notebook runs currently saved in the repo:

- the beat-aligned pipeline was run on a subset of `102` folders / `290` chart triples
- that run produced `31,119` training sequences
- the derived vocabulary size was `82`
- the transformer notebook shows early baseline training and qualitative token-generation examples

## Pipeline

### 1. Parse and reconstruct charts

The parser in [`src/preprocessing/osutaiko_parser.py`](/c:/Users/28548/PythonNotebooks/taiko-diffusion/src/preprocessing/osutaiko_parser.py) converts an osu!taiko `.osu` file into:

- note events for training
- timing-point reference data
- metadata needed for reconstruction

The reconstructor in [`src/preprocessing/osutaiko_reconstructor.py`](/c:/Users/28548/PythonNotebooks/taiko-diffusion/src/preprocessing/osutaiko_reconstructor.py) can rebuild a `.osu` file from those exported artifacts.

### 2. Build a beat-aligned dataset

The beat-aligned dataset builder in [`src/preprocessing/beat_aligned_dataset.py`](/c:/Users/28548/PythonNotebooks/taiko-diffusion/src/preprocessing/beat_aligned_dataset.py) does the following for each chart:

1. Match one audio file with one parsed chart triple.
2. Read timing and construct a beat grid.
3. Map note events onto beat-relative frame indices.
4. Compute mel spectrograms from audio.
5. Interpolate the spectrogram onto the beat-aligned timeline.
6. Split the aligned audio into 4-beat sequences.
7. Convert note events into per-sequence token lists.

Current dataset settings:

- `48` frames per beat
- `4` beats per sequence
- `192` frames per sequence
- `128` mel bins

### 3. Train a transformer baseline

The baseline model in [`src/model/model.py`](/c:/Users/28548/PythonNotebooks/taiko-diffusion/src/model/model.py) is a transformer encoder-decoder:

- encoder input: beat-aligned audio sequence `(192, 128)`
- decoder input: autoregressive note tokens
- output: the next token in the chart sequence

Supporting code lives in:

- [`src/model/data.py`](/c:/Users/28548/PythonNotebooks/taiko-diffusion/src/model/data.py) for manifests, splits, vocabulary, dataset, and collation
- [`src/model/trainer.py`](/c:/Users/28548/PythonNotebooks/taiko-diffusion/src/model/trainer.py) for training and validation loops
- [`src/model/generation.py`](/c:/Users/28548/PythonNotebooks/taiko-diffusion/src/model/generation.py) for greedy decoding and song-level generation

## Quick Start

1. Install the package:

```bash
pip install -e .
```

2. Unpack `.osz` files:

```bash
python src/preprocessing/unpack_osz.py
```

3. Parse and reconstruct sample beatmaps:

- open [`data_preprocessing.ipynb`](/c:/Users/28548/PythonNotebooks/taiko-diffusion/data_preprocessing.ipynb) and run the notebook

4. Build the beat-aligned dataset:

- open [`beat_aligned_dataset.ipynb`](/c:/Users/28548/PythonNotebooks/taiko-diffusion/beat_aligned_dataset.ipynb) and run the pipeline cells

5. Train or inspect the transformer baseline:

- open [`otsu_Transformer.ipynb`](/c:/Users/28548/PythonNotebooks/taiko-diffusion/otsu_Transformer.ipynb)

## Repository Layout

- [`src/preprocessing`](/c:/Users/28548/PythonNotebooks/taiko-diffusion/src/preprocessing): unpacking, parsing, reconstruction, and beat-aligned dataset building
- [`src/model`](/c:/Users/28548/PythonNotebooks/taiko-diffusion/src/model): transformer baseline, data utilities, training loop, and generation helpers
- [`sample_data`](/c:/Users/28548/PythonNotebooks/taiko-diffusion/sample_data): small local examples for parser/reconstruction work
- notebooks: exploratory and milestone notebooks for preprocessing, dataset creation, analysis, and modeling

## Current Limitations

- The current beat-aligned dataset builder only supports constant-BPM charts; charts with BPM changes are rejected during dataset creation.
- Several notebooks and scripts still use machine-specific absolute paths and are not yet packaged as a clean end-to-end CLI workflow.
- The implemented model is currently a transformer baseline; diffusion training/inference is still future work.
