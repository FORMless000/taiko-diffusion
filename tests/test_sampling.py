import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if torch is not None:
    from src.model.generation import SamplingConfig, TaikoBeatmapGenerator


class _DummyModel:
    def eval(self):
        return self

    def encode_audio(self, audio):
        return audio

    def decode_with_memory(self, memory, input_ids, decoder_attention_mask=None, difficulty_values=None, density_values=None, beatmap_id_values=None):
        vocab = 11
        logits = torch.zeros((input_ids.size(0), input_ids.size(1), vocab), device=input_ids.device)
        return logits


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestSamplingSafeguards(unittest.TestCase):
    def test_class_aware_candidates_limit_ts_and_keep_events(self):
        token_to_id = {
            "PAD": 0,
            "BOS": 1,
            "EOS": 2,
            "DON": 3,
            "KAT": 4,
            "TS_1": 5,
            "TS_2": 6,
            "TS_3": 7,
            "TS_4": 8,
            "TS_5": 9,
            "TS_6": 10,
        }
        id_to_token = {v: k for k, v in token_to_id.items()}

        gen = TaikoBeatmapGenerator(_DummyModel(), token_to_id, id_to_token, device=torch.device("cpu"))
        logits = torch.tensor([0.1, 0.1, 0.3, 0.8, 0.7, 5.0, 4.9, 4.8, 4.7, 4.6, 4.5], dtype=torch.float32)

        candidates = gen._class_aware_candidate_ids(
            logits,
            top_k=2,
            ts_top_k=1,
            min_event_candidates=2,
        )

        candidate_tokens = [id_to_token[int(i)] for i in candidates.tolist()]
        ts_count = sum(1 for t in candidate_tokens if t.startswith("TS_"))
        event_count = sum(1 for t in candidate_tokens if not t.startswith("TS_") and t not in {"PAD", "BOS"})

        self.assertLessEqual(ts_count, 1)
        self.assertGreaterEqual(event_count, 2)

    def test_sampling_is_deterministic_for_fixed_seed(self):
        token_to_id = {
            "PAD": 0,
            "BOS": 1,
            "EOS": 2,
            "DON": 3,
            "KAT": 4,
            "TS_1": 5,
            "TS_2": 6,
        }
        id_to_token = {v: k for k, v in token_to_id.items()}

        gen = TaikoBeatmapGenerator(_DummyModel(), token_to_id, id_to_token, device=torch.device("cpu"))
        logits_last = torch.tensor([[0.1, 0.1, 0.2, 0.9, 0.8, 1.5, 1.4]], dtype=torch.float32)
        generated_ids = torch.tensor([1, 3], dtype=torch.long)
        cfg = SamplingConfig(temperature=0.9, top_p=0.82, top_k=2, ts_top_k=1, min_event_candidates=2)

        torch.manual_seed(123)
        first = gen._sample_next_token(logits_last, generated_ids, cfg)
        torch.manual_seed(123)
        second = gen._sample_next_token(logits_last, generated_ids, cfg)

        self.assertEqual(int(first.item()), int(second.item()))


if __name__ == "__main__":
    unittest.main()
