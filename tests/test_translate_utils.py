import importlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import pysrt
except ModuleNotFoundError:  # pragma: no cover
    pysrt = None

try:
    translate_utils = importlib.import_module("translate_utils")
except ModuleNotFoundError:  # pragma: no cover - dependency chain might be missing
    translate_utils = None


@unittest.skipIf(translate_utils is None, "translate_utils dependencies are unavailable")
class TranslateUtilsTests(unittest.TestCase):
    def test_join_sentences_respects_sentence_endings(self):
        lines = ["Hello world.", "Another line!", "Trailing"]
        chunks = translate_utils.join_sentences(lines, max_chars=200)

        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200)
            self.assertTrue(chunk.endswith(translate_utils.separator))

    def test_unjoin_sentences_preserves_line_count(self):
        original_lines = ["Hello world.", "Another line."]
        original_chunk = translate_utils.join_sentences(original_lines, max_chars=200)[0]
        modified_chunk = translate_utils.separator_unjoin.join(["Hola mundo.", "Otra linea."])

        rebuilt = translate_utils.unjoin_sentences(original_chunk, modified_chunk, translate_utils.separator_unjoin)
        self.assertEqual(len(rebuilt), len(original_lines))
        self.assertTrue(all(isinstance(item, str) for item in rebuilt))

    def test_unjoin_sentences_handles_missing_translation(self):
        original_lines = ["Only line."]
        original_chunk = translate_utils.join_sentences(original_lines, max_chars=200)[0]
        rebuilt = translate_utils.unjoin_sentences(original_chunk, None, translate_utils.separator_unjoin)
        self.assertEqual(rebuilt, original_lines or ' ')

    def test_cli_translates_single_file_with_custom_output_dir(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        input_file = Path(tmp_dir.name) / "sample.srt"
        input_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
        output_dir = Path(tmp_dir.name) / "out"

        with mock.patch.object(translate_utils, "translate_srt_file", autospec=True) as translate_mock:
            exit_code = translate_utils.main([
                "-i",
                str(input_file),
                "-o",
                str(output_dir),
                "--translate",
                "es",
            ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(translate_mock.call_count, 1)
        called_source, called_dest, called_lang = translate_mock.call_args[0]
        self.assertEqual(called_source, input_file.resolve())
        self.assertEqual(called_dest, (output_dir / "sample_es.srt").resolve())
        self.assertEqual(called_lang, "es")
        self.assertEqual(translate_mock.call_args.kwargs["translate_engine"], "google")
        self.assertEqual(translate_mock.call_args.kwargs["gemini_api_keys"], [])

    def test_cli_auto_selects_gemini_when_api_key_provided(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        input_file = Path(tmp_dir.name) / "sample.srt"
        input_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")

        with mock.patch.object(translate_utils, "translate_srt_file", autospec=True) as translate_mock:
            exit_code = translate_utils.main([
                "-i",
                str(input_file),
                "--translate",
                "es",
                "--gemini_api_key",
                "dummy-key",
            ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(translate_mock.call_count, 1)
        self.assertEqual(translate_mock.call_args.kwargs["translate_engine"], "gemini")
        self.assertEqual(translate_mock.call_args.kwargs["gemini_api_keys"], ["dummy-key"])

    def test_cli_requires_directory_output_for_multiple_inputs(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        base = Path(tmp_dir.name)
        for idx in range(2):
            (base / f"file{idx}.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")

        with self.assertRaises(SystemExit):
            translate_utils.main([
                "-i",
                str(base),
                "-o",
                str(base / "output.srt"),
                "--translate",
                "es",
            ])

    def test_cli_requires_gemini_api_key(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        input_file = Path(tmp_dir.name) / "sample.srt"
        input_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")

        with self.assertRaises(SystemExit):
            translate_utils.main([
                "-i",
                str(input_file),
                "--translate",
                "es",
                "--translate_engine",
                "gemini",
            ])

    def test_cli_gemini_model_default(self):
        parser = translate_utils.build_cli_parser()
        args = parser.parse_args(["-i", "/tmp/x.srt", "--translate", "es"])
        self.assertEqual(args.gemini_model, "gemini-3.1-flash-lite")

    def test_cli_accepts_gemini_model_flag(self):
        parser = translate_utils.build_cli_parser()
        args = parser.parse_args([
            "-i", "/tmp/x.srt",
            "--translate", "es",
            "--gemini_model", "gemini-2.5-flash",
        ])
        self.assertEqual(args.gemini_model, "gemini-2.5-flash")

    def test_cli_forwards_gemini_model_to_translate_srt_file(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        input_file = Path(tmp_dir.name) / "sample.srt"
        input_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")

        with mock.patch.object(translate_utils, "translate_srt_file", autospec=True) as translate_mock:
            exit_code = translate_utils.main([
                "-i", str(input_file),
                "--translate", "es",
                "--gemini_api_key", "dummy-key",
                "--gemini_model", "gemini-3.1-flash-lite",
            ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(translate_mock.call_args.kwargs.get("gemini_model"), "gemini-3.1-flash-lite")

    @unittest.skipIf(pysrt is None, "pysrt unavailable")
    def test_strip_speaker_prefixes_removes_prefixes_and_returns_list(self):
        subs = pysrt.SubRipFile()
        subs.append(pysrt.SubRipItem(index=1, text="[SPEAKER_00] Hello world", start=pysrt.SubRipTime(0), end=pysrt.SubRipTime(1)))
        subs.append(pysrt.SubRipItem(index=2, text="[SPEAKER_01] Hi there", start=pysrt.SubRipTime(1), end=pysrt.SubRipTime(2)))
        subs.append(pysrt.SubRipItem(index=3, text="No prefix here", start=pysrt.SubRipTime(2), end=pysrt.SubRipTime(3)))

        prefixes = translate_utils._strip_speaker_prefixes(subs)

        self.assertEqual(prefixes, ["[SPEAKER_00] ", "[SPEAKER_01] ", ""])
        self.assertEqual(subs[0].text, "Hello world")
        self.assertEqual(subs[1].text, "Hi there")
        self.assertEqual(subs[2].text, "No prefix here")

    @unittest.skipIf(pysrt is None, "pysrt unavailable")
    def test_restore_speaker_prefixes_prepends_prefix(self):
        subs = pysrt.SubRipFile()
        subs.append(pysrt.SubRipItem(index=1, text="Olá mundo", start=pysrt.SubRipTime(0), end=pysrt.SubRipTime(1)))
        subs.append(pysrt.SubRipItem(index=2, text="Tudo bem?", start=pysrt.SubRipTime(1), end=pysrt.SubRipTime(2)))
        subs.append(pysrt.SubRipItem(index=3, text="Sem prefixo", start=pysrt.SubRipTime(2), end=pysrt.SubRipTime(3)))

        prefixes = ["[SPEAKER_00] ", "[SPEAKER_01] ", ""]

        translate_utils._restore_speaker_prefixes(subs, prefixes)

        self.assertEqual(subs[0].text, "[SPEAKER_00] Olá mundo")
        self.assertEqual(subs[1].text, "[SPEAKER_01] Tudo bem?")
        self.assertEqual(subs[2].text, "Sem prefixo")

    @unittest.skipIf(pysrt is None, "pysrt unavailable")
    def test_strip_and_restore_round_trip_preserves_text(self):
        original_texts = [
            "[SPEAKER_00] Hello world",
            "[SPEAKER_01] Hi there",
            "No prefix here",
        ]
        subs = pysrt.SubRipFile()
        for i, text in enumerate(original_texts):
            subs.append(pysrt.SubRipItem(index=i + 1, text=text, start=pysrt.SubRipTime(i), end=pysrt.SubRipTime(i + 1)))

        prefixes = translate_utils._strip_speaker_prefixes(subs)
        # Simulate translation: each text gets translated (here we just uppercase as stand-in)
        for sub in subs:
            sub.text = sub.text.upper()
        translate_utils._restore_speaker_prefixes(subs, prefixes)

        self.assertEqual(subs[0].text, "[SPEAKER_00] HELLO WORLD")
        self.assertEqual(subs[1].text, "[SPEAKER_01] HI THERE")
        self.assertEqual(subs[2].text, "NO PREFIX HERE")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
