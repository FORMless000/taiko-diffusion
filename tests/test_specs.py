import importlib.util
import sys
import unittest
from pathlib import Path

_SPEC_PATH = Path(__file__).resolve().parents[1] / "src" / "model" / "specs.py"
_MODULE_SPEC = importlib.util.spec_from_file_location("taiko_model_specs_for_test", _SPEC_PATH)
_MODULE = importlib.util.module_from_spec(_MODULE_SPEC)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
sys.modules[_MODULE_SPEC.name] = _MODULE
_MODULE_SPEC.loader.exec_module(_MODULE)
ArchitectureSpec = _MODULE.ArchitectureSpec


class TestArchitectureSpec(unittest.TestCase):
    def test_context_kwargs_excludes_dataset_only_fields(self):
        spec = ArchitectureSpec(
            name="taiko_context_transformer",
            history_max_tokens=128,
            retrieval_top_k=3,
            retrieval_max_tokens_per_window=32,
            retrieval_exclude_last_n_windows=1,
            use_motif_retrieval=True,
            max_cached_charts=7,
        )
        self.assertNotIn("max_cached_charts", spec.context_kwargs())
        self.assertEqual(spec.dataset_context_kwargs()["max_cached_charts"], 7)

    def test_roundtrip_preserves_max_cached_charts(self):
        original = ArchitectureSpec(name="taiko_context_transformer", max_cached_charts=11)
        restored = ArchitectureSpec.from_dict(original.to_dict())
        self.assertEqual(restored.max_cached_charts, 11)


if __name__ == "__main__":
    unittest.main()
