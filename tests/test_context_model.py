import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if torch is not None:
    from src.model.taiko_context import TaikoContextTransformer


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestContextModel(unittest.TestCase):
    def test_logits_change_when_history_changes(self):
        torch.manual_seed(0)
        model = TaikoContextTransformer(
            vocab_size=16,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            max_len=128,
            history_max_tokens=32,
            retrieval_top_k=1,
            retrieval_max_tokens_per_window=16,
        )

        audio = torch.randn(1, 192, 128)
        memory = model.encode_audio(audio)
        history_a = torch.tensor([[5, 2, 6, 2, 1, 3, 4]], dtype=torch.long)
        history_b = torch.tensor([[7, 2, 8, 2, 1, 3, 4]], dtype=torch.long)
        segment_ids = torch.tensor([[0, 0, 0, 0, 2, 2, 2]], dtype=torch.long)

        logits_a = model.decode_with_memory(memory=memory, input_ids=history_a, segment_ids=segment_ids)
        logits_b = model.decode_with_memory(memory=memory, input_ids=history_b, segment_ids=segment_ids)

        self.assertFalse(torch.allclose(logits_a[:, -1, :], logits_b[:, -1, :]))


if __name__ == "__main__":
    unittest.main()
