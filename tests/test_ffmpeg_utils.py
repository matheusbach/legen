import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ffmpeg_utils


class DummyTqdm:
    def __init__(self, *_, **__):
        self.n = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, value):
        self.n += value


class DummyProgress:
    def __init__(self, cmd):
        self.cmd = cmd

    def run_command_with_progress(self):
        yield 100


class FfmpegUtilsTests(unittest.TestCase):
    def test_existing_subtitles_mapped_after_generated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_media = Path(tmpdir) / "input.mp4"
            input_media.write_bytes(b"media")

            srt_file = Path(tmpdir) / "legen.srt"
            srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\ntext\n", encoding="utf-8")

            output_media = Path(tmpdir) / "output.mp4"
            captured_cmd = {}

            def fake_run(command, capture_output=False, text=False, **kwargs):
                if "-show_streams" in command:
                    return subprocess.CompletedProcess(command, 0, stdout="DISPOSITION:attached_pic=0")
                raise AssertionError(f"Unexpected command: {command}")

            with patch("ffmpeg_utils.subprocess.run", side_effect=fake_run), \
                 patch("ffmpeg_utils.tqdm", side_effect=lambda *a, **k: DummyTqdm()), \
                 patch("ffmpeg_utils.FfmpegProgress", side_effect=lambda cmd: DummyProgress(cmd)) as mock_progress:
                ffmpeg_utils.insert_subtitle(
                    input_media_path=input_media,
                    subtitles_path=[srt_file],
                    burn_subtitles=False,
                    output_video_path=output_media,
                    codec_video="h264",
                    codec_audio="aac",
                )

            cmd = mock_progress.call_args[0][0]
            map_entries = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-map"]

            self.assertIn("0:s?", map_entries)
            generated_indexes = [idx for idx, value in enumerate(map_entries) if value.endswith(":s") and value != "0:s?"]
            self.assertTrue(generated_indexes)
            downloaded_index = map_entries.index("0:s?")
            self.assertTrue(all(idx < downloaded_index for idx in generated_indexes))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
