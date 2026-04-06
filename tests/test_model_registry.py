import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.specs import ModelSpec

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from src.model.factory import build_model
    from src.model.registry import list_architectures


class TestModelSpecSerialization(unittest.TestCase):
    def test_model_spec_round_trip(self):
        spec = ModelSpec(name="transformer_baseline", params={"d_model": 32, "dropout": 0.2})
        recovered = ModelSpec.from_dict(spec.to_dict())
        self.assertEqual(recovered, spec)


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestModelRegistry(unittest.TestCase):
    def test_build_model_uses_registered_architecture(self):
        spec = ModelSpec(
            name="transformer_baseline",
            params={
                "input_dim": 128,
                "d_model": 32,
                "nhead": 4,
                "num_encoder_layers": 1,
                "num_decoder_layers": 1,
                "dim_feedforward": 64,
            },
        )
        model = build_model(spec, vocab_size=10, input_shape=(192, 128))

        self.assertIn("transformer_baseline", list_architectures())
        self.assertEqual(model.audio_embed.proj.in_features, 128)
        self.assertEqual(model.output_head.proj.out_features, 10)


if __name__ == "__main__":
    unittest.main()
