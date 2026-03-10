# taiko-diffusion

Minimal project scaffold for generating osu!taiko charts from music.

## Current status

- Project intent is documented in `PROJECT_INTENT.md`.
- `src/preprocessing/unpack_osz.py` unpacks `.osz` beatmap archives.
- `notebooks/unpack_osz.ipynb` runs the unpacking flow interactively.

## Quick start

1. Install dependencies:
   - `pip install -e .`
2. Run from notebook:
   - Open `notebooks/unpack_osz.ipynb` and run all cells.
3. Or run as a script:
   - `python src/preprocessing/unpack_osz.py`

