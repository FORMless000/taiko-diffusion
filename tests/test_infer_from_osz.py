import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.infer_from_osz import select_top_difficulty_chart, song_output_to_notes_json


class TestInferFromOsz(unittest.TestCase):
    def _write_chart_bundle(
        self,
        parsed_dir: Path,
        stem: str,
        *,
        version: str,
        od: float,
        playable_notes: int,
        ms_per_beat: float = 400.0,
    ) -> None:
        metadata = {
            "format": 2,
            "source_osu": f"{stem}.osu",
            "general": {"AudioFilename": "audio.ogg", "Mode": "1"},
            "metadata": {
                "Title": "Song",
                "Artist": "Artist",
                "Creator": "Mapper",
                "Version": version,
                "BeatmapID": "123",
                "BeatmapSetID": "456",
            },
            "difficulty": {
                "OverallDifficulty": str(od),
                "SliderMultiplier": "1.4",
                "SliderTickRate": "1",
            },
        }
        timing = {
            "format": 2,
            "source_osu": f"{stem}.osu",
            "slider_multiplier": 1.4,
            "slider_tick_rate": 1.0,
            "timing_points": [
                {
                    "offset": 1000,
                    "ms_per_beat": ms_per_beat,
                    "meter": 4,
                    "sample_set": 1,
                    "sample_index": 0,
                    "volume": 100,
                    "uninherited": 1,
                    "effects": 0,
                }
            ],
        }
        notes = {
            "format": 2,
            "mode": 1,
            "source_osu": f"{stem}.osu",
            "notes": [
                {
                    "type": "bpmchange",
                    "time": 1000,
                    "raw_time": 1000,
                    "sv": 1.0,
                    "volume": 100,
                    "bpm": 150.0,
                    "meter": 4,
                }
            ],
        }
        for i in range(playable_notes):
            notes["notes"].append(
                {
                    "type": "don",
                    "time": 1100 + i * 100,
                    "raw_time": 1100 + i * 100,
                    "sv": 1.0,
                    "volume": 100,
                    "bpm": None,
                    "meter": None,
                }
            )

        (parsed_dir / f"{stem}.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (parsed_dir / f"{stem}.timing.json").write_text(json.dumps(timing), encoding="utf-8")
        (parsed_dir / f"{stem}.notes.json").write_text(json.dumps(notes), encoding="utf-8")

    def test_select_top_difficulty_chart_prefers_highest_od(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            unpacked_dir = Path(tmpdir) / "2034220"
            parsed_dir = unpacked_dir / "parsed"
            parsed_dir.mkdir(parents=True, exist_ok=True)
            (unpacked_dir / "audio.ogg").write_bytes(b"fake-audio")

            self._write_chart_bundle(
                parsed_dir,
                "chart_easy",
                version="Kantan",
                od=2.0,
                playable_notes=40,
            )
            self._write_chart_bundle(
                parsed_dir,
                "chart_hard",
                version="Oni",
                od=8.5,
                playable_notes=38,
            )

            selected = select_top_difficulty_chart(unpacked_dir)
            self.assertEqual(selected.stem, "chart_hard")
            self.assertGreater(selected.overall_difficulty, 8.0)

    def test_song_output_to_notes_json_resolves_timestamps(self):
        song_output = [
            {
                "seq_idx": 0,
                "pred_tokens": ["TS_24", "DON", "TS_12", "KAT"],
            },
            {
                "seq_idx": 1,
                "pred_tokens": ["TS_48", "BIGDON"],
            },
        ]
        notes_json = song_output_to_notes_json(
            song_output,
            source_osu="demo.osu",
            offset_ms=1000.0,
            bpm=150.0,
            meter=4,
            sv_default=1.0,
            volume_default=100,
        )
        notes = notes_json["notes"]
        self.assertGreaterEqual(len(notes), 4)
        self.assertEqual(notes[0]["type"], "bpmchange")
        playable = [n for n in notes if n["type"] != "bpmchange"]
        self.assertEqual([n["type"] for n in playable], ["don", "kat", "bigdon"])
        self.assertTrue(playable[0]["time"] >= 1000.0)
        self.assertTrue(playable[2]["time"] > playable[1]["time"])


if __name__ == "__main__":
    unittest.main()

