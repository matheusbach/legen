import tempfile
import unittest
from pathlib import Path

import utils


class UtilsTests(unittest.TestCase):
    def test_format_time_formats_components(self):
        self.assertEqual(utils.format_time(3661), "1h 1m 1s")
        self.assertEqual(utils.format_time(0), "0s")

    def test_check_other_extensions_finds_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base = tmp_path / "video.mp4"
            sibling = tmp_path / "video.mkv"
            unrelated = tmp_path / "other.mp4"

            base.write_text("base", encoding="utf-8")
            sibling.write_text("sibling", encoding="utf-8")
            unrelated.write_text("unrelated", encoding="utf-8")

            matches = utils.check_other_extensions(base, [".mp4", ".mkv"])
            self.assertIn(sibling, matches)
            self.assertIn(base, matches)
            self.assertNotIn(unrelated, matches)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
