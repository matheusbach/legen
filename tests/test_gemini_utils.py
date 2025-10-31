import importlib
import unittest

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
