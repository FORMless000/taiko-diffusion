# taiko-diffusion

Minimal project scaffold for preprocessing and reconstructing osu!taiko beatmaps.

## Current status

- Project intent is documented in `PROJECT_INTENT.md`.
- `.osz` archives can be unpacked with `src/preprocessing/unpack_osz.py`.
- End-to-end preprocessing + reconstruction is demonstrated in `data_preprocessing.ipynb`.

## Preprocessing pipeline

The current pipeline is:

1. Unpack `.osz` files from `sample_data/raw/*.osz` into `sample_data/unpacked/<map_id>/`.
2. Parse a selected `.osu` into three intermediate JSON files:
   - `*.notes.json` (training-oriented sequence)
   - `*.timing.json` (timing reference)
   - `*.metadata.json` (General/Metadata/Difficulty reference)
3. Optionally include explicit `bpmchange` events in `*.notes.json`.
4. Reconstruct `.osu` from parsed JSON files for round-trip validation.

## Quick start

1. Install dependencies:
   - `pip install -e .`
2. Unpack archives (script):
   - `python src/preprocessing/unpack_osz.py`
3. Run end-to-end preprocessing and reconstruction (notebook):
   - Open `data_preprocessing.ipynb` and run all cells.

## Notebook flow (`data_preprocessing.ipynb`)

The notebook follows this pattern:

```python
from pathlib import Path
from src.preprocessing.osutaiko_parser import parse_osu_file_to_jsons

map_name = "OU - Mountain (FORMless000) [dang dang di dang]"
map_id = "2516534"

osu_path = Path(f"sample_data/unpacked/{map_id}/{map_name}.osu")
out_dir = Path(f"sample_data/unpacked/{map_id}/parsed")

parse_osu_file_to_jsons(
    osu_path=osu_path,
    out_dir=out_dir,
    include_bpm_events=True,
)
```

Then load:

- `out_dir / f"{map_name}.notes.json"`
- `out_dir / f"{map_name}.timing.json"`
- `out_dir / f"{map_name}.metadata.json"`

And reconstruct:

```python
from src.preprocessing.osutaiko_reconstructor import reconstruct_osu

reconstruct_osu(
    notes_path=out_dir / f"{map_name}.notes.json",
    timing_path=out_dir / f"{map_name}.timing.json",
    metadata_path=out_dir / f"{map_name}.metadata.json",
    out_path=out_dir / f"{map_name}.reconstructed.osu",
)
```

You can also reconstruct from notes only (without timing/metadata) by passing `None` for `timing_path` and `metadata_path`.

## Notes on timing behavior

- Parser snapping keeps raw float snapped timestamps in parsed outputs.
- Reconstructed `.osu` timing and hitobject offsets are rounded to integer milliseconds during export (osu! format compatible).
