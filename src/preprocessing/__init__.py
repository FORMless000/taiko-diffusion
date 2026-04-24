"""Preprocessing utilities for osu!taiko data."""

from .unpack_osz import unpack_osz_files

__all__ = ["unpack_osz_files"]

try:
    from .build_snapshot_dataset import build_snapshot_dataset
except ImportError:
    build_snapshot_dataset = None
else:
    __all__.append("build_snapshot_dataset")

try:
    from .prepare_training_data import (
        TrainingDataArtifacts,
        prepare_training_data,
        resolve_osz_input_paths,
    )
except ImportError:
    TrainingDataArtifacts = None
    prepare_training_data = None
    resolve_osz_input_paths = None
else:
    __all__.extend(
        [
            "TrainingDataArtifacts",
            "prepare_training_data",
            "resolve_osz_input_paths",
        ]
    )
