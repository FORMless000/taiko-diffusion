"""Compatibility wrapper for the beat-aligned dataset pipeline."""

from .beat_aligned import *  # noqa: F401,F403


if __name__ == "__main__":
    setup_logging()
    run_pipeline(
        unpacked_root=DEFAULT_UNPACKED_ROOT,
        index_dir=DEFAULT_INDEX_DIR,
        dataset_dir=DEFAULT_DATASET_DIR,
    )
