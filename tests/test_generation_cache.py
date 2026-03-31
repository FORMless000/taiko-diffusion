import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if torch is not None:
    from src.model.generation import TaikoBeatmapGenerator


class _DummyModel:
    def eval(self):
        return self

    def encode_audio(self, audio):
        return audio

    def decode_with_memory(self, memory, input_ids, decoder_attention_mask=None, difficulty_values=None, density_values=None, beatmap_id_values=None):
        vocab = 7
        return torch.zeros((input_ids.size(0), input_ids.size(1), vocab), device=input_ids.device)


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestGeneratorCache(unittest.TestCase):
    def test_cache_hit_and_miss_by_key(self):
        token_to_id = {"PAD": 0, "BOS": 1, "EOS": 2, "DON": 3, "KAT": 4, "TS_1": 5, "TS_2": 6}
        id_to_token = {v: k for k, v in token_to_id.items()}

        gen = TaikoBeatmapGenerator(_DummyModel(), token_to_id, id_to_token, device=torch.device("cpu"), audio_cache_size=4)

        calls = {"count": 0}

        def compute_once():
            calls["count"] += 1
            return ["value"]

        key_a = ("song_a", 0.0, 180.0, 4)
        key_b = ("song_b", 0.0, 180.0, 4)

        _ = gen._cache_get_or_compute(key_a, compute_once)
        _ = gen._cache_get_or_compute(key_a, compute_once)
        _ = gen._cache_get_or_compute(key_b, compute_once)

        self.assertEqual(calls["count"], 2)


if __name__ == "__main__":
    unittest.main()
