import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if torch is not None:
    from src.model.generation import TaikoBeatmapGenerator
else:
    TaikoBeatmapGenerator = object


class _ContextDummyModel:
    supports_long_context = True
    history_max_tokens = 32
    retrieval_top_k = 0
    retrieval_max_tokens_per_window = 16
    retrieval_exclude_last_n_windows = 0
    use_motif_retrieval = False

    def __init__(self):
        self.decode_calls = []

    def eval(self):
        return self

    def encode_audio(self, audio):
        return torch.ones((audio.size(0), audio.size(1), 8), device=audio.device)

    def _pool_audio_memory(self, memory):
        pooled = memory.mean(dim=1)
        return torch.nn.functional.normalize(pooled, dim=-1)

    def decode_with_memory(self, memory, input_ids, decoder_attention_mask=None, difficulty_values=None, density_values=None, beatmap_id_values=None, segment_ids=None):
        self.decode_calls.append(input_ids.detach().cpu().clone())
        vocab = 4
        logits = torch.zeros((input_ids.size(0), input_ids.size(1), vocab), device=input_ids.device)
        last_token = int(input_ids[0, -1].item())
        if last_token == 1:
            logits[:, -1, 3] = 10.0
        else:
            logits[:, -1, 2] = 10.0
        return logits


class _GeneratorWithStubAudio(TaikoBeatmapGenerator):
    def preprocess_audio(self, audio_path, offset_ms, bpm, meter=4):
        del audio_path, offset_ms, bpm, meter
        return np.zeros((2, 192, 128), dtype=np.float32)


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestContextGeneration(unittest.TestCase):
    def test_generation_carries_history_across_windows(self):
        token_to_id = {"PAD": 0, "BOS": 1, "EOS": 2, "DON": 3}
        id_to_token = {v: k for k, v in token_to_id.items()}
        model = _ContextDummyModel()
        gen = _GeneratorWithStubAudio(model, token_to_id, id_to_token, device=torch.device("cpu"), max_len=4)

        song = gen.generate_song_structure("unused.wav", 0.0, 180.0)

        self.assertEqual(song[0]["pred_tokens"], ["DON"])
        self.assertEqual(song[1]["pred_tokens"], ["DON"])
        self.assertEqual(int(model.decode_calls[0].shape[1]), 1)
        self.assertGreater(int(model.decode_calls[2].shape[1]), 1)


if __name__ == "__main__":
    unittest.main()
