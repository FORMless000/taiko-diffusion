import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.osutaiko_parser import parse_osu_file_to_jsons, parse_unpacked_taiko_charts
from src.preprocessing.osutaiko_reconstructor import reconstruct_osu


TEST_OSU_TEXT = """osu file format v14

[General]
AudioFilename:test.wav
Mode:1

[Metadata]
Title:Test Song
Artist:Test Artist
Creator:Unit Test
Version:Taiko
BeatmapID:1
BeatmapSetID:1

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
"""


class TestParserReconstructor(unittest.TestCase):
    def test_parse_and_reconstruct_round_trip_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            osu_path = root / "test.osu"
            out_dir = root / "parsed"
            reconstructed_path = root / "test.reconstructed.osu"

            osu_path.write_text(TEST_OSU_TEXT, encoding="utf-8")

            parse_osu_file_to_jsons(osu_path, out_dir, include_bpm_events=True)

            notes_path = out_dir / "test.notes.json"
            timing_path = out_dir / "test.timing.json"
            metadata_path = out_dir / "test.metadata.json"

            self.assertTrue(notes_path.exists())
            self.assertTrue(timing_path.exists())
            self.assertTrue(metadata_path.exists())

            reconstruct_osu(
                notes_path=notes_path,
                timing_path=timing_path,
                metadata_path=metadata_path,
                out_path=reconstructed_path,
            )

            reconstructed_text = reconstructed_path.read_text(encoding="utf-8")
            self.assertIn("[TimingPoints]", reconstructed_text)
            self.assertIn("[HitObjects]", reconstructed_text)
            self.assertIn("256,192,500,1,0", reconstructed_text)

    def test_bulk_parser_skips_non_taiko_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            chart_dir = root / "123"
            chart_dir.mkdir(parents=True, exist_ok=True)
            (chart_dir / "audio.wav").write_bytes(b"RIFF")

            taiko_path = chart_dir / "taiko.osu"
            taiko_path.write_text(TEST_OSU_TEXT, encoding="utf-8")

            mania_text = TEST_OSU_TEXT.replace("Mode:1", "Mode:3")
            mania_path = chart_dir / "mania.osu"
            mania_path.write_text(mania_text, encoding="utf-8")

            result = parse_unpacked_taiko_charts(root, include_bpm_events=False, overwrite=False)

            self.assertEqual(result["parsed_count"], 1)
            self.assertGreaterEqual(result["skipped_count"], 1)
            self.assertTrue((chart_dir / "parsed" / "taiko.notes.json").exists())
            self.assertFalse((chart_dir / "parsed" / "mania.notes.json").exists())


if __name__ == "__main__":
    unittest.main()
