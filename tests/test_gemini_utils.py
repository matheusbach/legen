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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
