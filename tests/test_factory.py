import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if torch is not None:
    from src.model.factory import build_model
    from src.model.model import TaikoTransformer
    from src.model.specs import ArchitectureSpec
    from src.model.taiko_context import TaikoContextTransformer


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestModelFactory(unittest.TestCase):
    def test_builds_default_transformer(self):
        spec = ArchitectureSpec(
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            max_len=64,
        )
        model = build_model(spec, vocab_size=17)
        self.assertIsInstance(model, TaikoTransformer)

    def test_rejects_unknown_architecture(self):
        spec = ArchitectureSpec(name="does_not_exist")
        with self.assertRaises(ValueError):
            build_model(spec, vocab_size=17)

    def test_builds_context_transformer(self):
        spec = ArchitectureSpec(
            name="taiko_context_transformer",
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            max_len=64,
            history_max_tokens=32,
            retrieval_top_k=1,
            retrieval_max_tokens_per_window=16,
        )
        model = build_model(spec, vocab_size=17)
        self.assertIsInstance(model, TaikoContextTransformer)
        self.assertEqual(model.history_max_tokens, 32)


if __name__ == "__main__":
    unittest.main()
