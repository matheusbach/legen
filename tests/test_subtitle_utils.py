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

    def test_format_speaker_prefix_none(self):
        self.assertEqual(subtitle_utils.format_speaker_prefix(None), "")

    def test_format_speaker_prefix_unknown(self):
        self.assertEqual(subtitle_utils.format_speaker_prefix("UNKNOWN"), "")

    def test_format_speaker_prefix_empty(self):
        self.assertEqual(subtitle_utils.format_speaker_prefix(""), "")

    def test_format_speaker_prefix_speaker_00(self):
        self.assertEqual(subtitle_utils.format_speaker_prefix("SPEAKER_00"), "[SPEAKER_00] ")

    def test_format_speaker_prefix_speaker_07(self):
        self.assertEqual(subtitle_utils.format_speaker_prefix("SPEAKER_07"), "[SPEAKER_07] ")

    def test_save_srt_with_speaker_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            srt_path = Path(tmpdir) / "out.srt"
            segments = [
                {"start": 0.0, "end": 1.0, "text": "Olá mundo", "speaker": "SPEAKER_00"},
                {"start": 1.5, "end": 2.5, "text": "Tudo bem?", "speaker": "SPEAKER_01"},
            ]
            subtitle_utils.SaveSegmentsToSrt(segments, srt_path)
            content = srt_path.read_text(encoding="utf-8")
            self.assertIn("[SPEAKER_00] Olá mundo", content)
            self.assertIn("[SPEAKER_01] Tudo bem?", content)

    def test_save_srt_without_speaker_has_no_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            srt_path = Path(tmpdir) / "out.srt"
            segments = [
                {"start": 0.0, "end": 1.0, "text": "Olá mundo"},
            ]
            subtitle_utils.SaveSegmentsToSrt(segments, srt_path)
            content = srt_path.read_text(encoding="utf-8")
            self.assertNotIn("[SPEAKER", content)
            self.assertIn("Olá mundo", content)

    def test_save_srt_unknown_speaker_has_no_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            srt_path = Path(tmpdir) / "out.srt"
            segments = [
                {"start": 0.0, "end": 1.0, "text": "Olá mundo", "speaker": "UNKNOWN"},
            ]
            subtitle_utils.SaveSegmentsToSrt(segments, srt_path)
            content = srt_path.read_text(encoding="utf-8")
            self.assertNotIn("[SPEAKER", content)

    def test_save_srt_labels_only_when_speaker_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            srt_path = Path(tmpdir) / "out.srt"
            segments = [
                {"start": 0.0, "end": 1.0, "text": "Primeira", "speaker": "SPEAKER_00"},
                {"start": 1.0, "end": 2.0, "text": "Continuação", "speaker": "SPEAKER_00"},
                {"start": 2.0, "end": 3.0, "text": "Resposta", "speaker": "SPEAKER_01"},
                {"start": 3.0, "end": 4.0, "text": "Nova intervenção", "speaker": "SPEAKER_00"},
            ]

            subtitle_utils.SaveSegmentsToSrt(segments, srt_path)
            content = srt_path.read_text(encoding="utf-8")

            self.assertEqual(content.count("[SPEAKER_00]"), 2)
            self.assertEqual(content.count("[SPEAKER_01]"), 1)
            self.assertIn("[SPEAKER_00] Primeira", content)
            self.assertIn("Continuação", content)
            self.assertNotIn("[SPEAKER_00] Continuação", content)

    def test_save_srt_relabels_after_unknown_speaker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            srt_path = Path(tmpdir) / "out.srt"
            segments = [
                {"start": 0.0, "end": 1.0, "text": "Conhecida", "speaker": "SPEAKER_00"},
                {"start": 1.0, "end": 2.0, "text": "Continuação", "speaker": "SPEAKER_00"},
                {"start": 2.0, "end": 3.0, "text": "Sem atribuição", "speaker": "UNKNOWN"},
                {"start": 3.0, "end": 4.0, "text": "Retorno", "speaker": "SPEAKER_00"},
            ]

            subtitle_utils.SaveSegmentsToSrt(segments, srt_path)
            content = srt_path.read_text(encoding="utf-8")

            self.assertEqual(content.count("[SPEAKER_00]"), 2)
            self.assertNotIn("[UNKNOWN]", content)
            self.assertNotIn("[SPEAKER_00] Continuação", content)

    def test_export_plain_text_from_srt_separates_speaker_turns(self):
        subs = pysrt.SubRipFile()
        subs.extend([
            pysrt.SubRipItem(
                index=1,
                text="[SPEAKER_00] Primeira frase.",
                start=pysrt.SubRipTime(seconds=0),
                end=pysrt.SubRipTime(seconds=1),
            ),
            pysrt.SubRipItem(
                index=2,
                text="Continuação do turno.",
                start=pysrt.SubRipTime(seconds=1),
                end=pysrt.SubRipTime(seconds=2),
            ),
            pysrt.SubRipItem(
                index=3,
                text="[SPEAKER_01] Resposta.",
                start=pysrt.SubRipTime(seconds=2),
                end=pysrt.SubRipTime(seconds=3),
            ),
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "out.txt"
            text = subtitle_utils.export_plain_text_from_srt(subs, output_path)

            expected = (
                "[SPEAKER_00] Primeira frase. Continuação do turno.\n"
                "[SPEAKER_01] Resposta."
            )
            self.assertEqual(text, expected)
            self.assertEqual(output_path.read_text(encoding="utf-8"), expected)

    def test_export_plain_text_from_srt_collapses_legacy_repeated_prefixes(self):
        subs = pysrt.SubRipFile()
        subs.extend([
            pysrt.SubRipItem(
                index=1,
                text="[SPEAKER_00] Uma.",
                start=pysrt.SubRipTime(seconds=0),
                end=pysrt.SubRipTime(seconds=1),
            ),
            pysrt.SubRipItem(
                index=2,
                text="[SPEAKER_00] Duas.",
                start=pysrt.SubRipTime(seconds=1),
                end=pysrt.SubRipTime(seconds=2),
            ),
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            text = subtitle_utils.export_plain_text_from_srt(
                subs,
                Path(tmpdir) / "out.txt",
            )

            self.assertEqual(text, "[SPEAKER_00] Uma. Duas.")

    def test_export_plain_text_from_srt_preserves_label_after_unlabeled_cue(self):
        subs = pysrt.SubRipFile()
        subs.extend([
            pysrt.SubRipItem(index=1, text="[SPEAKER_00] A"),
            pysrt.SubRipItem(index=2, text="B"),
            pysrt.SubRipItem(index=3, text="[SPEAKER_00] D"),
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            text = subtitle_utils.export_plain_text_from_srt(
                subs,
                Path(tmpdir) / "out.txt",
            )

            self.assertEqual(text, "[SPEAKER_00] A B\n[SPEAKER_00] D")

    def test_export_plain_text_from_srt_applies_prefix_only_to_next_cue(self):
        subs = pysrt.SubRipFile()
        subs.extend([
            pysrt.SubRipItem(index=1, text="[SPEAKER_01] Previous"),
            pysrt.SubRipItem(index=2, text="[SPEAKER_00]"),
            pysrt.SubRipItem(index=3, text="A"),
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            text = subtitle_utils.export_plain_text_from_srt(
                subs,
                Path(tmpdir) / "out.txt",
            )

            self.assertEqual(text, "[SPEAKER_01] Previous\n[SPEAKER_00] A")

    def test_export_plain_text_from_srt_discards_pending_prefix_after_explicit_cue(self):
        subs = pysrt.SubRipFile()
        subs.extend([
            pysrt.SubRipItem(index=1, text="[SPEAKER_01] Previous"),
            pysrt.SubRipItem(index=2, text="[SPEAKER_00]"),
            pysrt.SubRipItem(index=3, text="[SPEAKER_02] Replacement"),
            pysrt.SubRipItem(index=4, text="Tail"),
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            text = subtitle_utils.export_plain_text_from_srt(
                subs,
                Path(tmpdir) / "out.txt",
            )

            self.assertEqual(text, "[SPEAKER_01] Previous\n[SPEAKER_02] Replacement Tail")

    @patch("subtitle_utils.string_width", side_effect=lambda text, *_: len(text) * 100)
    def test_split_segments_preserves_speaker(self, mock_width):
        segments = [{
            "text": "Hello brave new world",
            "start": 0.0,
            "end": 4.0,
            "speaker": "SPEAKER_00",
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.5},
                {"word": "brave", "start": 0.5, "end": 1.0},
                {"word": "new", "start": 1.0, "end": 1.5},
                {"word": "world", "start": 1.5, "end": 2.0},
            ],
        }]

        result = subtitle_utils.split_segments(segments, max_width_px=200)
        self.assertGreaterEqual(len(result), 2)
        for seg in result:
            self.assertEqual(seg.get("speaker"), "SPEAKER_00")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
