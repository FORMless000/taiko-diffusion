import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if torch is not None:
    from src.model.runtime import build_dataloader_runtime_kwargs, resolve_precision_runtime
    from src.model.specs import TrainingSpec


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestRuntimeHelpers(unittest.TestCase):
    def test_precision_auto_falls_back_to_fp32_on_cpu(self):
        runtime = resolve_precision_runtime("auto", torch.device("cpu"))
        self.assertEqual(runtime.requested, "auto")
        self.assertEqual(runtime.resolved, "fp32")
        self.assertFalse(runtime.autocast_enabled)
        self.assertFalse(runtime.scaler_enabled)

    def test_precision_fp16_on_cpu_falls_back_to_fp32(self):
        runtime = resolve_precision_runtime("fp16", torch.device("cpu"))
        self.assertEqual(runtime.resolved, "fp32")
        self.assertTrue("falling back" in runtime.fallback_reason.lower())

    def test_dataloader_kwargs_auto_defaults(self):
        spec = TrainingSpec(num_workers=2, pin_memory=None, persistent_workers=None, prefetch_factor=None)
        kwargs = build_dataloader_runtime_kwargs(spec, torch.device("cuda"))
        self.assertEqual(kwargs["num_workers"], 2)
        self.assertTrue(kwargs["pin_memory"])
        self.assertTrue(kwargs["persistent_workers"])
        self.assertEqual(kwargs["prefetch_factor"], 2)

    def test_dataloader_kwargs_with_zero_workers_omit_prefetch_and_persistent(self):
        spec = TrainingSpec(num_workers=0, pin_memory=None, persistent_workers=True, prefetch_factor=4)
        kwargs = build_dataloader_runtime_kwargs(spec, torch.device("cpu"))
        self.assertEqual(kwargs["num_workers"], 0)
        self.assertFalse(kwargs["pin_memory"])
        self.assertNotIn("persistent_workers", kwargs)
        self.assertNotIn("prefetch_factor", kwargs)


if __name__ == "__main__":
    unittest.main()
