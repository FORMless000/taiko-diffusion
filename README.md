# taiko-diffusion

Automatic osu!taiko beatmap generation from music.

Quick backend start from the repo root:

- PowerShell: `.\main.ps1`
- Bash: `./main.sh`

Both scripts boot the FastAPI backend on `0.0.0.0:12205` by default.

## Project Description

`taiko-diffusion` is a research and engineering project for generating osu!taiko beatmaps from audio. The repository includes the full pipeline from beatmap parsing and beat-aligned dataset construction to model training, checkpoint-backed inference, and a deployable web app.

The current implemented focus is the transformer side of the project: autoregressive transformer baselines and a context transformer variant are trained and wired into the web app. Diffusion is part of the broader project direction and is present in the repo, but it is not the main documented or operational path in this README.

For the short project vision, see [PROJECT_INTENT.md](C:/Users/28548/PythonNotebooks/taiko-diffusion/PROJECT_INTENT.md).

## Data Source

The dataset source is ranked osu!taiko beatmap sets.

- We used all `10,048` ranked beatmap sets for osu!taiko.
- Download automation used `batch-beatmap-downloader`: <https://github.com/nzbasic/batch-beatmap-downloader>
- The shared dataset link recorded in this repo is:
  - <https://www.dropbox.com/scl/fo/ipbgqy80vnithcbhjp33f/ALzhBrzJm013t9gtPpoa9VU?rlkey=kqo1ia6iq6zo1ep4j9usybg1u&st=unov6n5l&dl=0>
- Manual beatmap-set download is also available through:
  - <https://osu.ppy.sh/beatmapsets?m=1&s=ranked>

Ranked taiko charts were chosen because they are community-reviewed and give higher-quality supervision for beatmap generation.

### Repository Data Pipeline

The preprocessing pipeline converts raw beatmaps and audio into training-ready paired data:

1. Unpack raw `.osz` archives.
2. Parse each taiko chart into:
   - `*.notes.json`
   - `*.timing.json`
   - `*.metadata.json`
3. Convert audio into mel-spectrogram features.
4. Interpolate audio onto a beat-aligned grid with `48` frames per beat.
5. Split aligned audio into 4-beat windows with shape `(192, 128)`.
6. Serialize note timing and note events into token sequences using `TS_*` and event tokens such as `DON`, `KAT`, `BIGDON`, `BIGKAT`, `DRUMROLL`, `SLIDERSTART`, and `SLIDEREND`.

The beat-aligned dataset builder writes artifacts such as:

- `beat_aligned_dataset/audio_npz/*.npz`
- `beat_aligned_dataset/token_json/*.json`
- `beat_aligned_dataset/sequence_metadata.csv`
- chart summaries under `chart_index/`

## Required Packages

### Install Commands

Core Python package:

```bash
pip install -e .
```

Web app backend extras:

```bash
pip install -e .[webapp]
```

Optional W&B logging extras:

```bash
pip install -e .[wandb]
```

Frontend dependencies:

```bash
cd webapp/frontend
npm install
```

### Key Packages Used

Python core:

- `torch`
- `numpy`
- `pandas`
- `librosa`
- `scikit-learn`
- `matplotlib`
- `tqdm`

Web backend:

- `fastapi`
- `uvicorn`
- `python-multipart`

Frontend:

- `next`
- `react`
- `react-dom`
- `typescript`

The canonical dependency declarations live in:

- [pyproject.toml](C:/Users/28548/PythonNotebooks/taiko-diffusion/pyproject.toml)
- [webapp/frontend/package.json](C:/Users/28548/PythonNotebooks/taiko-diffusion/webapp/frontend/package.json)

## How To Run The Code

### 1. Preprocessing And Dataset Construction

Unpack beatmap archives:

```bash
python src/preprocessing/unpack_osz.py
```

Then use the notebooks and shared modules to inspect parsing and build beat-aligned data:

- [data_preprocessing.ipynb](C:/Users/28548/PythonNotebooks/taiko-diffusion/data_preprocessing.ipynb)
- [beat_aligned_dataset.ipynb](C:/Users/28548/PythonNotebooks/taiko-diffusion/beat_aligned_dataset.ipynb)

### 2. Training / Model Inspection

The model code lives under:

- [src/model](C:/Users/28548/PythonNotebooks/taiko-diffusion/src/model)

The repo includes training notebooks for the baseline and context models, including large-sample, maxopt, and snapshot variants. The original exploratory transformer notebook is:

- [otsu_Transformer.ipynb](C:/Users/28548/PythonNotebooks/taiko-diffusion/otsu_Transformer.ipynb)

### 3. Local Web App Inference

Build the frontend:

```bash
cd webapp/frontend
npm run build
```

Run the backend:

```bash
python -m webapp.backend.main
```

This starts the FastAPI service on `http://127.0.0.1:8000`.

Useful endpoints:

- `GET /api/models`
- `POST /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/download/osz`

The backend serves its model list from [webapp/backend/models.json](C:/Users/28548/PythonNotebooks/taiko-diffusion/webapp/backend/models.json). That manifest currently includes multiple AR/context checkpoints plus diffusion-hybrid model options layered on top of them.

### 4. Web App Inputs

The frontend accepts:

- required fields:
  - audio file
  - title
  - artist
  - difficulty name
  - BPM
  - offset
- advanced fields:
  - overall difficulty
  - density NPS
  - beatmap ID
  - temperature
  - creator
  - source
  - tags

Important notes:

- `beatmap_id` is used as a model conditioning / auditing input, not normal beatmap metadata for the user.
- `temperature` is passed through sampling overrides at inference time.
- current inference assumes constant-BPM timing.

## Repository Layout

- [src/preprocessing](C:/Users/28548/PythonNotebooks/taiko-diffusion/src/preprocessing): unpacking, parsing, reconstruction, and beat-aligned dataset building
- [src/model](C:/Users/28548/PythonNotebooks/taiko-diffusion/src/model): transformer models, datasets, training logic, and generation helpers
- [src/inference](C:/Users/28548/PythonNotebooks/taiko-diffusion/src/inference): checkpoint-backed inference and generation service logic
- [webapp](C:/Users/28548/PythonNotebooks/taiko-diffusion/webapp): FastAPI backend, Next.js frontend, job runtime workspace, and deployment helpers
- [sample_data](C:/Users/28548/PythonNotebooks/taiko-diffusion/sample_data): local parser/reconstruction examples

For deployment and operational web app details, see [webapp/README.md](C:/Users/28548/PythonNotebooks/taiko-diffusion/webapp/README.md).

## Current Limitations

- The beat-aligned dataset builder currently supports only constant-BPM charts.
- The web app and metadata-driven inference path also currently assume constant-BPM timing.
- Several notebooks still use machine-specific paths and remain part of the research workflow rather than a polished end-to-end CLI.
- The repo direction includes diffusion, but the main mature path today is the transformer AR/context pipeline.
