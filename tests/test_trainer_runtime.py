import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if torch is not None:
    from src.model.trainer import train_one_epoch, validate_one_epoch


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestTrainerRuntimeStats(unittest.TestCase):
    def _make_batch(self):
        return {
            "audio": torch.zeros((2, 192, 128), dtype=torch.float32),
            "input_ids": torch.tensor([[1, 3, 4], [1, 4, 3]], dtype=torch.long),
            "labels": torch.tensor([[3, 4, 2], [4, 3, 2]], dtype=torch.long),
            "decoder_attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long),
            "difficulty_values": torch.tensor([0.5, 0.6], dtype=torch.float32),
            "density_values": torch.tensor([0.4, 0.5], dtype=torch.float32),
            "beatmap_id_values": torch.tensor([0.2, 0.3], dtype=torch.float32),
        }

    def test_train_and_validate_report_throughput_stats(self):
        class _DummyModel(torch.nn.Module):
            def __init__(self, vocab_size=8):
                super().__init__()
                self.proj = torch.nn.Linear(4, vocab_size)

            def forward(
                self,
                audio,
                input_ids,
                decoder_attention_mask=None,
                difficulty_values=None,
                density_values=None,
                beatmap_id_values=None,
                segment_ids=None,
            ):
                del audio, decoder_attention_mask, difficulty_values, density_values, beatmap_id_values, segment_ids
                features = torch.nn.functional.one_hot(input_ids % 4, num_classes=4).float()
                return self.proj(features)

        model = _DummyModel(vocab_size=8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        dataloader = [self._make_batch(), self._make_batch()]

        train_stats = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=torch.device("cpu"),
        )
        self.assertGreater(train_stats["samples_per_sec"], 0.0)
        self.assertGreater(train_stats["tokens_per_sec"], 0.0)
        self.assertGreater(train_stats["avg_batch_time_sec"], 0.0)

        val_stats = validate_one_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            device=torch.device("cpu"),
        )
        self.assertGreater(val_stats["samples_per_sec"], 0.0)
        self.assertGreater(val_stats["tokens_per_sec"], 0.0)
        self.assertGreater(val_stats["avg_batch_time_sec"], 0.0)


if __name__ == "__main__":
    unittest.main()
