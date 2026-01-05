import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import download_utils


class DownloadUtilsTests(unittest.TestCase):
    class FakeProcess:
        def __init__(self, output_lines, on_wait=None, returncode=0):
            self.stdout = io.StringIO("".join(output_lines))
            self._returncode = returncode
            self.returncode = returncode
            self._on_wait = on_wait

        def wait(self):
            if self._on_wait:
                self._on_wait()
            return self._returncode

        def kill(self):
            pass

    @patch("download_utils.shutil.which", return_value="/usr/bin/yt-dlp")
    def test_resolve_downloader_found(self, mock_which):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Ensure the "local yt-dlp next to sys.executable" fast-path is not taken.
            with patch("download_utils.sys.executable", new=str(Path(tmpdir) / "python")):
                self.assertEqual(download_utils._resolve_downloader(), "yt-dlp")
        mock_which.assert_called_once_with("yt-dlp")

    @patch("download_utils.shutil.which", return_value=None)
    def test_resolve_downloader_missing(self, mock_which):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("download_utils.sys.executable", new=str(Path(tmpdir) / "python")):
                with self.assertRaises(FileNotFoundError):
                    download_utils._resolve_downloader()
        mock_which.assert_called_once_with("yt-dlp")

    @patch("download_utils._append_downloaded_suffix_to_subtitles")
    @patch("download_utils.subprocess.Popen")
    @patch("download_utils.shutil.which", return_value="/usr/bin/yt-dlp")
    def test_download_urls_resume_and_skip_with_subtitles(self, mock_which, mock_popen, mock_suffix):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            mp4_path = output_dir / "video.mp4"
            (output_dir / "video.mp4.part").write_bytes(b"partial")

            lines = [
                f"[download] Destination: {mp4_path}\n",
                "[download] Resuming download at byte 12345\n",
                "[download]  10.0% of 10.00MiB at 1.00MiB/s ETA 00:10\n",
                "[download] 100% of 10.00MiB in 00:10\n",
            ]

            def on_wait():
                mp4_path.write_bytes(b"data")

            fake_process = self.FakeProcess(lines, on_wait=on_wait)
            captured = {}

            def fake_popen(command, stdout, stderr, text):
                captured["command"] = command
                return fake_process

            mock_popen.side_effect = fake_popen

            result = download_utils.download_urls(
                [" https://example.com/video "],
                output_dir,
                download_remote_subs=True,
            )

            self.assertEqual(result, [mp4_path])
            self.assertTrue(mp4_path.exists())
            self.assertIn("--no-warnings", captured["command"])
            self.assertIn("--progress", captured["command"])
            self.assertIn("--newline", captured["command"])
            self.assertNotIn("--quiet", captured["command"])
            mock_suffix.assert_called_once_with(mp4_path)

    @patch("download_utils._append_downloaded_suffix_to_subtitles")
    @patch("download_utils.subprocess.Popen")
    @patch("download_utils.shutil.which", return_value="/usr/bin/yt-dlp")
    def test_download_urls_force_overwrite_with_subtitles(self, mock_which, mock_popen, mock_suffix):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            mp4_path = output_dir / "video.mp4"

            lines = [
                f"[download] Destination: {mp4_path}\n",
                "[download] 100% of 10.00MiB in 00:10\n",
            ]

            def on_wait():
                mp4_path.write_bytes(b"data")

            fake_process = self.FakeProcess(lines, on_wait=on_wait)
            captured = {}

            def fake_popen(command, stdout, stderr, text):
                captured["command"] = command
                return fake_process

            mock_popen.side_effect = fake_popen

            result = download_utils.download_urls(
                ["https://example.com/video"],
                output_dir,
                overwrite=True,
                download_remote_subs=True,
            )

            self.assertEqual(result, [mp4_path])
            self.assertIn("--force-overwrites", captured["command"])
            self.assertNotIn("--no-overwrites", captured["command"])
            mock_suffix.assert_called_once_with(mp4_path)

    @patch("download_utils._append_downloaded_suffix_to_subtitles")
    @patch("download_utils.subprocess.Popen")
    @patch("download_utils.shutil.which", return_value="/usr/bin/yt-dlp")
    def test_download_urls_no_subtitles_by_default(self, mock_which, mock_popen, mock_suffix):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            mp4_path = output_dir / "video.mp4"

            lines = [
                f"[download] Destination: {mp4_path}\n",
                "[download] 100% of 10.00MiB in 00:10\n",
            ]

            def on_wait():
                mp4_path.write_bytes(b"data")

            fake_process = self.FakeProcess(lines, on_wait=on_wait)
            captured = {}

            def fake_popen(command, stdout, stderr, text):
                captured["command"] = command
                return fake_process

            mock_popen.side_effect = fake_popen

            result = download_utils.download_urls(["https://example.com/video"], output_dir)

            self.assertEqual(result, [mp4_path])
            self.assertNotIn("--embed-subs", captured["command"])
            self.assertNotIn("--sub-langs", captured["command"])
            mock_suffix.assert_not_called()

    @patch("download_utils.shutil.which", return_value="/usr/bin/yt-dlp")
    def test_download_urls_no_urls(self, mock_which):
        with self.assertRaises(ValueError):
            download_utils.download_urls([], Path("/tmp"))

    @patch("download_utils.subprocess.Popen")
    @patch("download_utils.shutil.which", return_value="/usr/bin/yt-dlp")
    def test_download_urls_multi_stream_prints_title_once(self, mock_which, mock_popen):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            mp4_path = output_dir / "video.mp4"

            lines = [
                "[download] Destination: video.f137.mp4\n",
                "[download] 100% of 10.00MiB in 00:10\n",
                "[download] Destination: video.f140.m4a\n",
                "[download] 100% of 3.00MiB in 00:05\n",
            ]

            def on_wait():
                mp4_path.write_bytes(b"data")

            fake_process = self.FakeProcess(lines, on_wait=on_wait)
            mock_popen.return_value = fake_process

            with patch("sys.stdout", new_callable=io.StringIO) as fake_stdout:
                result = download_utils.download_urls([
                    "https://example.com/video"
                ], output_dir)

            output = fake_stdout.getvalue()
            self.assertEqual(result, [mp4_path])
            self.assertEqual(output.count("Downloading: video"), 1)

    @patch("download_utils.subprocess.Popen")
    @patch("download_utils.shutil.which", return_value="/usr/bin/yt-dlp")
    def test_download_urls_subprocess_failure(self, mock_which, mock_popen):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_process = self.FakeProcess([], returncode=1)
            mock_popen.return_value = fake_process

            with self.assertRaises(RuntimeError) as exc:
                download_utils.download_urls(["https://example.com/video"], Path(tmpdir))
            self.assertIn("failed with exit code", str(exc.exception))

    @patch("download_utils.subprocess.Popen")
    @patch("download_utils.shutil.which", return_value="/usr/bin/yt-dlp")
    def test_download_urls_no_output_files(self, mock_which, mock_popen):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_process = self.FakeProcess([])
            mock_popen.return_value = fake_process

            with self.assertRaises(RuntimeError) as exc:
                download_utils.download_urls(["https://example.com/video"], Path(tmpdir))
            self.assertIn("without producing MP4 files", str(exc.exception))

    @patch("download_utils.subprocess.run")
    def test_append_downloaded_suffix_no_streams(self, mock_run):
        with tempfile.TemporaryDirectory() as tmpdir:
            media_path = Path(tmpdir) / "video.mp4"
            media_path.write_bytes(b"data")

            mock_run.return_value = subprocess.CompletedProcess(
                ["ffprobe"], 0, stdout=json.dumps({"streams": []})
            )

            download_utils._append_downloaded_suffix_to_subtitles(media_path)
            mock_run.assert_called_once()

    def test_append_downloaded_suffix_updates_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            media_path = Path(tmpdir) / "video.mp4"
            media_path.write_bytes(b"original")

            streams = {
                "streams": [
                    {"tags": {"title": "English"}},
                    {"tags": {"language": "es"}},
                ]
            }

            def fake_run(command, check=True, capture_output=False, text=False, **kwargs):
                if command[0] == "ffprobe":
                    return subprocess.CompletedProcess(command, 0, stdout=json.dumps(streams))
                if command[0] == "ffmpeg":
                    self.assertIn("-metadata:s:s:0", command)
                    idx0 = command.index("-metadata:s:s:0")
                    self.assertEqual(command[idx0 + 1], "title=English [downloaded]")

                    self.assertIn("-metadata:s:s:1", command)
                    idx1 = command.index("-metadata:s:s:1")
                    self.assertEqual(command[idx1 + 1], "title=es [downloaded]")

                    output = command[-1]
                    if output.startswith("file:"):
                        Path(output[5:]).write_bytes(b"muxed")
                    else:
                        Path(output).write_bytes(b"muxed")
                    return subprocess.CompletedProcess(command, 0)
                raise AssertionError(f"Unexpected command: {command}")

            with patch("download_utils.subprocess.run", side_effect=fake_run) as mock_run:
                download_utils._append_downloaded_suffix_to_subtitles(media_path)

            self.assertEqual(media_path.read_bytes(), b"muxed")
            self.assertEqual(mock_run.call_count, 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
