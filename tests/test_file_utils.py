import os
import tempfile
import time
import unittest
from pathlib import Path

import file_utils


class FileUtilsTests(unittest.TestCase):
    def test_validate_files_filters_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            valid = tmp_path / "valid.txt"
            empty = tmp_path / "empty.txt"
            missing = tmp_path / "missing.txt"

            valid.write_text("data", encoding="utf-8")
            empty.write_text("", encoding="utf-8")

            result = file_utils.validate_files([valid, empty, missing])
            self.assertEqual(result, [valid])

    def test_tempfile_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            final_path = tmp_path / "final.txt"

            temp_file = file_utils.TempFile(final_path, file_ext=".txt")
            temp_path = temp_file.getpath()
            self.assertTrue(temp_path.exists())

            temp_path.write_text("temporary", encoding="utf-8")
            temp_file.save()

            self.assertTrue(final_path.exists())
            self.assertEqual(final_path.read_text(encoding="utf-8"), "temporary")

            temp_file.destroy()  # should not raise even if nothing to delete

    def test_copy_file_if_different(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src = tmp_path / "src.txt"
            dst = tmp_path / "dst.txt"

            src.write_text("content", encoding="utf-8")
            file_utils.copy_file_if_different(src, dst, silent=True)

            self.assertTrue(dst.exists())
            self.assertEqual(dst.read_text(encoding="utf-8"), "content")

    def test_update_folder_times_tracks_newest_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            older = folder / "older.txt"
            newer = folder / "newer.txt"

            older.write_text("old", encoding="utf-8")
            time.sleep(0.01)
            newer.write_text("new", encoding="utf-8")

            original_mtime = folder.stat().st_mtime
            newest = file_utils.update_folder_times(folder)

            self.assertGreaterEqual(newest, int(newer.stat().st_mtime))
            self.assertGreaterEqual(int(folder.stat().st_mtime), int(original_mtime))

    def test_check_existing_path_accepts_files_and_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            file_path = folder / "file.txt"
            file_path.write_text("data", encoding="utf-8")

            self.assertEqual(file_utils.check_existing_path(str(folder)), str(folder))
            self.assertEqual(file_utils.check_existing_path(str(file_path)), str(file_path))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
