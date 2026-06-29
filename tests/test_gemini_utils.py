import importlib
import tempfile
import unittest
from pathlib import Path

try:
    gemini_utils = importlib.import_module("gemini_utils")
except ModuleNotFoundError:  # pragma: no cover - dependency chain might be missing
    gemini_utils = None


@unittest.skipIf(gemini_utils is None, "gemini_utils dependencies are unavailable")
class GeminiUtilsTests(unittest.TestCase):
    def test_normalize_api_keys_handles_iterables_and_duplicates(self):
        keys = [" key1 ", "key2,key3", "key1", None, "\nkey4\n"]
        expected = ["key1", "key2", "key3", "key4"]
        self.assertEqual(gemini_utils.normalize_api_keys(keys), expected)

    def test_normalize_api_keys_accepts_single_string(self):
        self.assertEqual(gemini_utils.normalize_api_keys("key"), ["key"])

    def test_normalize_api_keys_empty_input(self):
        self.assertEqual(gemini_utils.normalize_api_keys(None), [])
        self.assertEqual(gemini_utils.normalize_api_keys([]), [])

    def test_generate_tltw_rotates_keys_and_writes_output(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        subtitle_file = Path(tmp_dir.name) / "input.srt"
        subtitle_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello world\n", encoding="utf-8")
        output_file = Path(tmp_dir.name) / "summary.md"

        calls: list[str] = []

        def fake_request(**kwargs):
            calls.append(kwargs["api_key"])
            if kwargs["api_key"] == "bad-key":
                raise RuntimeError("boom")
            return "TLTW summary"

        config = gemini_utils.GeminiSummaryConfig(
            api_keys=["bad-key", "good-key"],
            subtitle_file=subtitle_file,
            output_file=output_file,
            language="en",
        )

        result = gemini_utils.generate_tltw(config, request_func=fake_request)

        self.assertEqual(result, "TLTW summary")
        self.assertEqual(calls, ["bad-key", "good-key"])
        self.assertTrue(output_file.exists())
        self.assertEqual(output_file.read_text(encoding="utf-8"), "TLTW summary")

    def test_generate_tltw_chunks_and_final_synthesis(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        subtitle_file = Path(tmp_dir.name) / "input.srt"
        long_text = "".join(["line" * 1000 for _ in range(3)])  # force chunking
        subtitle_file.write_text(long_text, encoding="utf-8")
        output_file = Path(tmp_dir.name) / "summary.md"

        calls = []

        def fake_request(prompt_builder=None, **kwargs):
            # record prompt identity and payload length
            calls.append({
                "api_key": kwargs["api_key"],
                "payload_len": len(kwargs["subtitle_text"]),
                "prompt": prompt_builder("en") if prompt_builder else None,
            })
            return f"summary-{len(calls)}"

        config = gemini_utils.GeminiSummaryConfig(
            api_keys=["k1"],
            subtitle_file=subtitle_file,
            output_file=output_file,
            language="en",
            chunk_chars=1_000,  # small to force multiple chunks
            max_output_tokens=200,
            final_max_output_tokens=400,
        )

        result = gemini_utils.generate_tltw(config, request_func=fake_request)

        self.assertTrue(result.startswith("summary-"))
        # Expect chunk summaries plus final synthesis: >=3 calls
        self.assertGreaterEqual(len(calls), 3)
        # Final output written
        self.assertTrue(output_file.exists())
        self.assertEqual(output_file.read_text(encoding="utf-8"), result)

    def test_gemini_translation_config_default_model(self):
        cfg = gemini_utils.GeminiTranslationConfig(
            api_keys=["k1"],
            input_file=Path("/tmp/in.srt"),
            output_file=Path("/tmp/out.srt"),
            target_language="pt",
        )
        self.assertEqual(cfg.model_name, "gemini-3.1-flash-lite")

    def test_gemini_translation_config_custom_model(self):
        cfg = gemini_utils.GeminiTranslationConfig(
            api_keys=["k1"],
            input_file=Path("/tmp/in.srt"),
            output_file=Path("/tmp/out.srt"),
            target_language="pt",
            model_name="gemini-2.5-flash",
        )
        self.assertEqual(cfg.model_name, "gemini-2.5-flash")

    def test_translate_with_gemini_passes_model_name(self):
        from unittest import mock

        cfg = gemini_utils.GeminiTranslationConfig(
            api_keys=["k1"],
            input_file=Path("/tmp/in.srt"),
            output_file=Path("/tmp/out.srt"),
            target_language="pt",
            model_name="gemini-3.1-flash-lite",
        )

        with mock.patch.object(gemini_utils, "MultiKeyGeminiTranslator") as mock_translator:
            mock_translator.return_value.translate.return_value = None
            try:
                gemini_utils.translate_with_gemini(cfg)
            except Exception:
                pass
            self.assertIn("model_name", mock_translator.call_args.kwargs)
            self.assertEqual(mock_translator.call_args.kwargs["model_name"], "gemini-3.1-flash-lite")

    def test_gemini_summary_config_default_model(self):
        cfg = gemini_utils.GeminiSummaryConfig(
            api_keys=["k1"],
            subtitle_file=Path("/tmp/in.srt"),
            output_file=Path("/tmp/out.md"),
            language="pt",
        )
        self.assertEqual(cfg.model, "gemini-3.1-flash-lite")

    def test_send_tltw_request_uses_google_genai_client(self):
        from unittest import mock

        from google import genai

        fake_response = mock.MagicMock()
        fake_response.text = "TLTW via new SDK"
        fake_response.candidates = [mock.MagicMock(finish_reason=None)]

        fake_client = mock.MagicMock()
        fake_client.models.generate_content.return_value = fake_response

        with mock.patch.object(genai, "Client", return_value=fake_client) as client_mock:
            result = gemini_utils._send_tltw_request(
                api_key="k1",
                subtitle_text="Hello world",
                language="en",
                model="gemini-3.1-flash-lite",
                max_output_tokens=100,
                request_timeout=500,
                stream_output=False,
                show_progress=False,
                prompt_builder=lambda lang: "Summarize. No final marker.",
            )

        self.assertEqual(result, "TLTW via new SDK")
        client_mock.assert_called_once()
        client_kwargs = client_mock.call_args.kwargs
        self.assertEqual(client_kwargs.get("api_key"), "k1")
        fake_client.models.generate_content.assert_called_once()
        gen_kwargs = fake_client.models.generate_content.call_args.kwargs
        self.assertEqual(gen_kwargs.get("model"), "gemini-3.1-flash-lite")
        self.assertIn("contents", gen_kwargs)
        self.assertIn("config", gen_kwargs)

    def test_send_tltw_request_streams_and_accumulates_chunks(self):
        from unittest import mock

        from google import genai

        chunk1 = mock.MagicMock()
        chunk1.text = "Hello "
        chunk2 = mock.MagicMock()
        chunk2.text = "world"
        final = mock.MagicMock()
        final.text = "Hello world"
        final.candidates = [mock.MagicMock(finish_reason=None)]

        fake_client = mock.MagicMock()
        fake_client.models.generate_content_stream.return_value = iter([chunk1, chunk2, final])

        with mock.patch.object(genai, "Client", return_value=fake_client):
            result = gemini_utils._send_tltw_request(
                api_key="k1",
                subtitle_text="Hello world",
                language="en",
                model="gemini-3.1-flash-lite",
                max_output_tokens=100,
                request_timeout=10,
                stream_output=True,
                show_progress=False,
                prompt_builder=lambda lang: "Summarize. No final marker.",
            )

        self.assertEqual(result, "Hello world")
        fake_client.models.generate_content_stream.assert_called_once()
        fake_client.models.generate_content.assert_not_called()

    def test_send_tltw_request_detects_max_tokens_truncation(self):
        from unittest import mock

        from google import genai
        from google.genai import types

        truncated_response = mock.MagicMock()
        truncated_response.text = "partial output"
        truncated_response.candidates = [
            mock.MagicMock(finish_reason=types.FinishReason.MAX_TOKENS)
        ]

        completed_response = mock.MagicMock()
        completed_response.text = "partial output continued and finished"
        completed_response.candidates = [
            mock.MagicMock(finish_reason=types.FinishReason.STOP)
        ]

        fake_client = mock.MagicMock()
        # First call truncates, second (continuation) completes.
        fake_client.models.generate_content_stream.side_effect = [
            iter([truncated_response]),
            iter([completed_response]),
        ]

        with mock.patch.object(genai, "Client", return_value=fake_client):
            result = gemini_utils._send_tltw_request(
                api_key="k1",
                subtitle_text="Hello world",
                language="en",
                model="gemini-3.1-flash-lite",
                max_output_tokens=50,
                request_timeout=10,
                stream_output=True,
                show_progress=False,
                max_rounds=3,
                prompt_builder=lambda lang: "Summarize. No final marker.",
            )

        self.assertIn("partial output", result)
        self.assertIn("continued", result)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
