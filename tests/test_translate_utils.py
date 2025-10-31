import importlib
import unittest

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
