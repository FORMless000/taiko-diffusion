import io
import math
import sys
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import librosa  # noqa: F401
    import torch
except ImportError:
    librosa = None
    torch = None

if torch is not None:
    from src.model.specs import ModelSpec
    from src.training import OptimizationConfig, SplitConfig, TrainingRunConfig, load_checkpoint, train_from_raw_osz


def _write_wav(path: Path, duration_sec: float = 4.0, sample_rate: int = 22050) -> None:
    n_samples = int(duration_sec * sample_rate)
    amplitude = 0.1
    frames = bytearray()
    for idx in range(n_samples):
        value = int(32767 * amplitude * math.sin(2.0 * math.pi * 440.0 * idx / sample_rate))
        frames.extend(int(value).to_bytes(2, byteorder="little", signed=True))

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(frames))


def _build_osu_text(audio_filename: str, beatmap_id: int) -> str:
    return f"""osu file format v14

[General]
AudioFilename:{audio_filename}
Mode:1

[Metadata]
Title:Song {beatmap_id}
Artist:Artist
Creator:Unit Test
Version:Taiko
BeatmapID:{beatmap_id}
BeatmapSetID:{beatmap_id}

[Difficulty]
HPDrainRate:5
CircleSize:5
OverallDifficulty:5
ApproachRate:5
SliderMultiplier:1.4
SliderTickRate:1

[TimingPoints]
0,500,4,1,0,100,1,0

[HitObjects]
256,192,500,1,0,0:0:0:0:
256,192,1000,1,8,0:0:0:0:
256,192,1500,1,4,0:0:0:0:
256,192,2000,1,12,0:0:0:0:
"""


def _create_osz_archive(path: Path, beatmap_id: int) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        audio_name = f"{beatmap_id}.wav"
        osu_name = f"{beatmap_id}.osu"
        audio_path = tmp_root / audio_name
        osu_path = tmp_root / osu_name

        _write_wav(audio_path)
        osu_path.write_text(_build_osu_text(audio_name, beatmap_id), encoding="utf-8")

        with zipfile.ZipFile(path, "w") as archive:
            archive.write(audio_path, arcname=audio_name)
            archive.write(osu_path, arcname=osu_name)


@unittest.skipIf(torch is None or librosa is None, "training dependencies are not installed in this environment")
class TestTrainingResume(unittest.TestCase):
    def test_end_to_end_training_and_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            raw_paths = []
            for beatmap_id in range(1001, 1006):
                osz_path = raw_dir / f"{beatmap_id}.osz"
                _create_osz_archive(osz_path, beatmap_id)
                raw_paths.append(str(osz_path))

            run_dir = root / "run"
            model_spec = ModelSpec(
                name="transformer_baseline",
                params={
                    "input_dim": 128,
                    "d_model": 16,
                    "nhead": 4,
                    "num_encoder_layers": 1,
                    "num_decoder_layers": 1,
                    "dim_feedforward": 32,
                    "dropout": 0.0,
                    "max_len": 64,
                },
            )
            base_optimization = OptimizationConfig(
                batch_size=2,
                num_epochs=1,
                learning_rate=1e-3,
                num_workers=0,
                pin_memory=False,
                use_amp=False,
            )
            split_config = SplitConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=7)

            first_config = TrainingRunConfig(
                run_dir=str(run_dir),
                raw_osz_paths=raw_paths,
                model_spec=model_spec,
                optimization=base_optimization,
                split=split_config,
                device="cpu",
            )
            first_result = train_from_raw_osz(first_config)

            latest_checkpoint = Path(first_result["latest_checkpoint"])
            self.assertTrue(latest_checkpoint.exists())
            self.assertEqual(len(first_result["history"]["train_loss"]), 1)

            resumed_config = TrainingRunConfig(
                run_dir=str(run_dir),
                raw_osz_paths=raw_paths,
                model_spec=model_spec,
                optimization=OptimizationConfig(
                    batch_size=1,
                    num_epochs=2,
                    learning_rate=1e-3,
                    num_workers=0,
                    pin_memory=False,
                    use_amp=False,
                ),
                split=split_config,
                resume_checkpoint=str(latest_checkpoint),
                device="cpu",
            )
            second_result = train_from_raw_osz(resumed_config)
            self.assertEqual(len(second_result["history"]["train_loss"]), 2)

            resumed_checkpoint = load_checkpoint(second_result["latest_checkpoint"])
            self.assertEqual(resumed_checkpoint["epoch"], 2)
            self.assertGreater(resumed_checkpoint["global_step"], 0)


if __name__ == "__main__":
    unittest.main()
