import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import pysrt
except ModuleNotFoundError:  # pragma: no cover - dependency might be missing in test envs
    pysrt = None

try:
    subtitle_utils = importlib.import_module("subtitle_utils")
except ModuleNotFoundError:  # pragma: no cover - dependency chain might be missing
    subtitle_utils = None


@unittest.skipIf(subtitle_utils is None or pysrt is None, "subtitle_utils dependencies are unavailable")
class SubtitleUtilsTests(unittest.TestCase):
    def test_export_plain_text_from_srt_writes_single_line(self):
        subs = pysrt.SubRipFile()
        subs.append(pysrt.SubRipItem(index=1, text="Hello\nworld", start=pysrt.SubRipTime(seconds=0), end=pysrt.SubRipTime(seconds=1)))
        subs.append(pysrt.SubRipItem(index=2, text="Another line", start=pysrt.SubRipTime(seconds=1), end=pysrt.SubRipTime(seconds=2)))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "out.txt"
            text = subtitle_utils.export_plain_text_from_srt(subs, output_path)

            self.assertEqual(text, "Hello world Another line")
            self.assertEqual(output_path.read_text(encoding="utf-8"), text)

    @patch("subtitle_utils.string_width", side_effect=lambda text, *_: len(text) * 100)
    def test_split_segments_respects_max_width(self, mock_width):
        segments = [{
            "text": "Hello brave new world",
            "start": 0.0,
            "end": 4.0,
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.5},
                {"word": "brave", "start": 0.5, "end": 1.0},
                {"word": "new", "start": 1.0, "end": 1.5},
                {"word": "world", "start": 1.5, "end": 2.0},
            ],
        }]

        result = subtitle_utils.split_segments(segments, max_width_px=200)
        self.assertGreaterEqual(len(result), 2)
        combined = " ".join(segment["text"].replace("\n", " ") for segment in result)
        self.assertIn("Hello brave", combined)

    @patch("subtitle_utils.string_width", side_effect=lambda text, *_: len(text) * 10)
    def test_split_string_to_max_lines_balances_lines(self, mock_width):
        lines = subtitle_utils.split_string_to_max_lines("one two three four five", max_width=30, max_lines=2)
        self.assertLessEqual(len(lines), 2)
        self.assertEqual(" ".join(lines).replace("  ", " ").strip(), "one two three four five")

    def test_adjust_times_expands_large_gaps(self):
        segments = [
            {"start": 0.0, "end": 0.5},
            {"start": 4.0, "end": 5.0},
        ]
        adjusted = subtitle_utils.adjust_times([dict(seg) for seg in segments], extra_end_time=1.0)
        self.assertAlmostEqual(adjusted[0]["end"], 1.5)

    def test_adjust_times_clamps_small_gaps(self):
        segments = [
            {"start": 0.0, "end": 0.5},
            {"start": 1.0, "end": 2.0},
        ]
        adjusted = subtitle_utils.adjust_times([dict(seg) for seg in segments], extra_end_time=1.0)
        self.assertEqual(adjusted[0]["end"], segments[1]["start"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
