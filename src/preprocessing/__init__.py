"""Preprocessing utilities for osu!taiko data.

Modules that require optional audio dependencies are imported lazily so parser
and reconstructor helpers still work in minimal environments.
"""

from .osutaiko_parser import *  # noqa: F401,F403
from .osutaiko_reconstructor import *  # noqa: F401,F403
from .unpack_osz import unpack_osz_files, unpack_osz_paths

__all__ = [
    "unpack_osz_files",
    "unpack_osz_paths",
]

try:
    from .beat_aligned_dataset import *  # noqa: F401,F403
except ModuleNotFoundError:
    pass
