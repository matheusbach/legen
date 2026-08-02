import importlib
import io
import re
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


try:
    whisperx_utils = importlib.import_module("whisperx_utils")
except ModuleNotFoundError:  # pragma: no cover - dependency chain might be missing
    whisperx_utils = None


class _FakeWhisperModel:
    def __init__(self):
        self.model = SimpleNamespace(
            feature_extractor=SimpleNamespace(sampling_rate=16000),
        )
        self.kwargs = None

    def transcribe(self, **kwargs):
        self.kwargs = kwargs
        kwargs["progress_callback"](100.0)
        return {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "language": "xx",
        }


@unittest.skipIf(whisperx_utils is None, "whisperx_utils dependencies are unavailable")
class ProgressAdapterTests(unittest.TestCase):
    def test_formats_float_progress_and_never_decreases(self):
        callback = whisperx_utils._make_progress_callback("WhisperX transcription")
        output = io.StringIO()

        with redirect_stdout(output):
            callback(12.5)
            callback(8.0)
            callback(100.0)

        progress = [
            float(value)
            for value in re.findall(r"(\d+(?:\.\d+)?)%", output.getvalue())
        ]
        self.assertEqual(progress, [12.5, 12.5, 100.0])

    def test_ignores_non_numeric_progress(self):
        callback = whisperx_utils._make_progress_callback("WhisperX transcription")
        output = io.StringIO()

        with redirect_stdout(output):
            callback("not-a-number")

        self.assertEqual(output.getvalue(), "")


@unittest.skipIf(whisperx_utils is None, "whisperx_utils dependencies are unavailable")
class UpstreamKeywordTests(unittest.TestCase):
    def test_transcription_uses_progress_callback_not_on_progress(self):
        model = _FakeWhisperModel()

        with patch.object(whisperx_utils.wx_audio, "load_audio", return_value=object()), \
             patch.object(whisperx_utils.subtitle_utils, "format_segments", return_value=[]), \
             patch.object(whisperx_utils.subtitle_utils, "SaveSegmentsToSrt"):
            whisperx_utils.transcribe_audio(
                model,
                Path("input.wav"),
                Path("output.srt"),
                lang="xx",
            )

        self.assertIn("progress_callback", model.kwargs)
        self.assertNotIn("on_progress", model.kwargs)

    def test_alignment_uses_progress_callback_not_on_progress(self):
        model = _FakeWhisperModel()
        align_kwargs = {}

        def fake_align(**kwargs):
            align_kwargs.update(kwargs)
            kwargs["progress_callback"](100.0)
            return {"segments": [], "language": "en"}

        with patch.object(whisperx_utils.wx_audio, "load_audio", return_value=object()), \
             patch.object(whisperx_utils.alignment, "DEFAULT_ALIGN_MODELS_HF", {"en"}), \
             patch.object(whisperx_utils.alignment, "DEFAULT_ALIGN_MODELS_TORCH", set()), \
             patch.object(whisperx_utils.alignment, "load_align_model", return_value=(object(), {})), \
             patch.object(whisperx_utils.alignment, "align", side_effect=fake_align), \
             patch.object(whisperx_utils.subtitle_utils, "format_segments", return_value=[]), \
             patch.object(whisperx_utils.subtitle_utils, "SaveSegmentsToSrt"):
            whisperx_utils.transcribe_audio(
                model,
                Path("input.wav"),
                Path("output.srt"),
                lang="en",
            )

        self.assertIn("progress_callback", align_kwargs)
        self.assertNotIn("on_progress", align_kwargs)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
