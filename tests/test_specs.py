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
TrainingSpec = _MODULE.TrainingSpec


class TestArchitectureSpec(unittest.TestCase):
    def test_context_defaults_are_fast_profile(self):
        spec = ArchitectureSpec(name="taiko_context_transformer")
        self.assertEqual(spec.history_max_tokens, 256)
        self.assertEqual(spec.retrieval_top_k, 1)
        self.assertEqual(spec.retrieval_max_tokens_per_window, 24)

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


class TestTrainingSpec(unittest.TestCase):
    def test_training_defaults_include_auto_precision_and_loader_autos(self):
        spec = TrainingSpec()
        self.assertEqual(spec.precision, "auto")
        self.assertIsNone(spec.pin_memory)
        self.assertIsNone(spec.persistent_workers)
        self.assertIsNone(spec.prefetch_factor)

    def test_invalid_precision_is_rejected(self):
        with self.assertRaises(ValueError):
            TrainingSpec(precision="half")

    def test_from_dict_backward_compatible_without_new_runtime_fields(self):
        payload = {
            "epochs": 3,
            "batch_size": 4,
            "lr": 1e-4,
            "weight_decay": 0.0,
            "seed": 42,
            "device": "cpu",
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "num_workers": 0,
        }
        restored = TrainingSpec.from_dict(payload)
        self.assertEqual(restored.precision, "auto")
        self.assertIsNone(restored.pin_memory)
        self.assertIsNone(restored.persistent_workers)
        self.assertIsNone(restored.prefetch_factor)


if __name__ == "__main__":
    unittest.main()
