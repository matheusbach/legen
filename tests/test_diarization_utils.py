import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    diarization_utils = importlib.import_module("diarization_utils")
except ModuleNotFoundError:  # pragma: no cover - dependency chain might be missing
    diarization_utils = None


@unittest.skipIf(diarization_utils is None, "diarization_utils dependencies are unavailable")
class DiarizationVersionDetectionTests(unittest.TestCase):
    def test_pyannote_version_parses_four_zero_seven(self):
        with mock.patch("pyannote.audio.__version__", "4.0.7"):
            self.assertEqual(diarization_utils._pyannote_version(), (4, 0))

    def test_community1_supported_on_v4(self):
        with mock.patch.object(diarization_utils, "_pyannote_version", return_value=(4, 0)):
            self.assertTrue(diarization_utils.community1_supported())

    def test_community1_unsupported_on_v3(self):
        with mock.patch.object(diarization_utils, "_pyannote_version", return_value=(3, 4)):
            self.assertFalse(diarization_utils.community1_supported())

    def test_active_cache_dir_changes_with_version(self):
        with mock.patch.object(diarization_utils, "_pyannote_version", return_value=(4, 0)):
            self.assertEqual(diarization_utils.active_cache_dir(), diarization_utils.COMMUNITY1_CACHE_DIR)
        with mock.patch.object(diarization_utils, "_pyannote_version", return_value=(3, 4)):
            self.assertEqual(diarization_utils.active_cache_dir(), diarization_utils.LEGACY_CACHE_DIR)


@unittest.skipIf(diarization_utils is None, "diarization_utils dependencies are unavailable")
class CacheValidTests(unittest.TestCase):
    def test_cache_valid_returns_false_for_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(diarization_utils.cache_valid(Path(tmp)))

    def test_cache_valid_returns_false_when_partial_community1(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / "segmentation").mkdir(parents=True)
            (cache / "segmentation" / "pytorch_model.bin").write_bytes(b"")
            (cache / "config.yaml").write_text("dependencies: pyannote.audio: 4.0.0\n", encoding="utf-8")
            with mock.patch.object(diarization_utils, "active_cache_dir", return_value=cache), \
                 mock.patch.object(diarization_utils, "community1_supported", return_value=True):
                self.assertFalse(diarization_utils.cache_valid())

    def test_cache_valid_returns_true_when_complete_community1(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            for rel, size in diarization_utils.COMMUNITY1_SIZES.items():
                p = cache / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x" * size)
            with mock.patch.object(diarization_utils, "active_cache_dir", return_value=cache), \
                 mock.patch.object(diarization_utils, "community1_supported", return_value=True):
                self.assertTrue(diarization_utils.cache_valid())

    def test_cache_valid_returns_false_when_sizes_mismatch_community1(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            for rel, size in diarization_utils.COMMUNITY1_SIZES.items():
                p = cache / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x" * (size + 1))  # wrong size
            with mock.patch.object(diarization_utils, "active_cache_dir", return_value=cache), \
                 mock.patch.object(diarization_utils, "community1_supported", return_value=True):
                self.assertFalse(diarization_utils.cache_valid())

    def test_cache_valid_returns_true_when_complete_legacy_31(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / "segmentation").mkdir(parents=True)
            (cache / "embedding").mkdir(parents=True)
            (cache / "segmentation" / "pytorch_model.bin").write_bytes(
                b"x" * diarization_utils.LEGACY_SIZES["segmentation/pytorch_model.bin"]
            )
            (cache / "embedding" / "pytorch_model.bin").write_bytes(
                b"x" * diarization_utils.LEGACY_SIZES["embedding/pytorch_model.bin"]
            )
            # legacy path needs our hand-written config.yaml
            (cache / "config.yaml").write_text("version: 3.1.0\n", encoding="utf-8")
            with mock.patch.object(diarization_utils, "active_cache_dir", return_value=cache), \
                 mock.patch.object(diarization_utils, "community1_supported", return_value=False):
                self.assertTrue(diarization_utils.cache_valid())


@unittest.skipIf(diarization_utils is None, "diarization_utils dependencies are unavailable")
class LegacyConfigWriterTests(unittest.TestCase):
    def test_write_legacy_config_yaml_points_at_local_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            seg_path = cache / "segmentation" / "pytorch_model.bin"
            emb_path = cache / "embedding" / "pytorch_model.bin"
            seg_path.parent.mkdir(parents=True)
            emb_path.parent.mkdir(parents=True)
            seg_path.write_bytes(b"")
            emb_path.write_bytes(b"")

            config_path = diarization_utils._write_legacy_config_yaml(cache)
            text = config_path.read_text(encoding="utf-8")

            self.assertIn("pyannote.audio.pipelines.SpeakerDiarization", text)
            self.assertIn("AgglomerativeClustering", text)
            self.assertIn(seg_path.resolve().as_posix(), text)
            self.assertIn(emb_path.resolve().as_posix(), text)


@unittest.skipIf(diarization_utils is None, "diarization_utils dependencies are unavailable")
class EnsureModelTests(unittest.TestCase):
    def test_ensure_skips_download_when_cache_valid_community1(self):
        with mock.patch.object(diarization_utils, "cache_valid", return_value=True), \
             mock.patch.object(diarization_utils, "community1_supported", return_value=True), \
             mock.patch.object(diarization_utils, "_download_file") as dl_mock:
            result = diarization_utils.ensure_diarization_model()
            self.assertEqual(result, diarization_utils.COMMUNITY1_CACHE_DIR)
            dl_mock.assert_not_called()

    def test_ensure_downloads_community1_when_invalid(self):
        with mock.patch.object(diarization_utils, "cache_valid", side_effect=[False, True]), \
             mock.patch.object(diarization_utils, "community1_supported", return_value=True), \
             mock.patch.object(diarization_utils, "_download_file") as dl_mock:
            result = diarization_utils.ensure_diarization_model()
            self.assertEqual(result, diarization_utils.COMMUNITY1_CACHE_DIR)
            # 5 files to download: config.yaml, segmentation, embedding, plda, xvec_transform
            self.assertEqual(dl_mock.call_count, len(diarization_utils.COMMUNITY1_FILES))

    def test_ensure_falls_back_to_legacy_31_when_unsupported(self):
        with mock.patch.object(diarization_utils, "cache_valid", side_effect=[False, True]), \
             mock.patch.object(diarization_utils, "community1_supported", return_value=False), \
             mock.patch.object(diarization_utils, "_download_file") as dl_mock, \
             mock.patch.object(diarization_utils, "_write_legacy_config_yaml") as cfg_mock:
            result = diarization_utils.ensure_diarization_model()
            self.assertEqual(result, diarization_utils.LEGACY_CACHE_DIR / "config.yaml")
            self.assertEqual(dl_mock.call_count, 2)  # segmentation + embedding
            cfg_mock.assert_called_once_with(diarization_utils.LEGACY_CACHE_DIR)


@unittest.skipIf(diarization_utils is None, "diarization_utils dependencies are unavailable")
class DiarizationOutputConverterTests(unittest.TestCase):
    def test_diarization_to_dataframe_handles_modern_output(self):
        # Modern pyannote 4.x returns a DiarizeOutput with .speaker_diarization
        class FakeAnnotation:
            def itertracks(self, yield_label=True):
                yield (type("Seg", (), {"start": 0.0, "end": 1.5})(), "label", "SPEAKER_00")
                yield (type("Seg", (), {"start": 2.0, "end": 3.5})(), "label", "SPEAKER_01")

        class FakeModernOutput:
            speaker_diarization = FakeAnnotation()

        df = diarization_utils._diarization_to_dataframe(FakeModernOutput())
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df["speaker"]), ["SPEAKER_00", "SPEAKER_01"])
        self.assertAlmostEqual(df["start"].iloc[0], 0.0)
        self.assertAlmostEqual(df["end"].iloc[1], 3.5)

    def test_diarization_to_dataframe_handles_legacy_output(self):
        # Legacy pyannote 3.x returns an Annotation directly
        class FakeAnnotation:
            def itertracks(self, yield_label=True):
                yield (type("Seg", (), {"start": 1.0, "end": 2.0})(), "label", "SPEAKER_00")

        df = diarization_utils._diarization_to_dataframe(FakeAnnotation())
        self.assertEqual(len(df), 1)
        self.assertEqual(df["speaker"].iloc[0], "SPEAKER_00")


@unittest.skipIf(diarization_utils is None, "diarization_utils dependencies are unavailable")
class ProgressBarVisibilityTests(unittest.TestCase):
    def test_disables_when_not_tty(self):
        with mock.patch("sys.stdout.isatty", return_value=False), \
             mock.patch("os.get_terminal_size", return_value=os.terminal_size((80, 24))):
            self.assertFalse(diarization_utils._should_show_progress_bar())

    def test_disables_when_terminal_narrow(self):
        with mock.patch("sys.stdout.isatty", return_value=True), \
             mock.patch("os.get_terminal_size", return_value=os.terminal_size((40, 24))):
            self.assertFalse(diarization_utils._should_show_progress_bar())

    def test_enables_on_wide_tty(self):
        with mock.patch("sys.stdout.isatty", return_value=True), \
             mock.patch("os.get_terminal_size", return_value=os.terminal_size((80, 24))):
            self.assertTrue(diarization_utils._should_show_progress_bar())

    def test_handles_missing_terminal(self):
        with mock.patch("sys.stdout.isatty", return_value=True), \
             mock.patch("os.get_terminal_size", side_effect=OSError):
            self.assertFalse(diarization_utils._should_show_progress_bar())


@unittest.skipIf(diarization_utils is None, "diarization_utils dependencies are unavailable")
class DiarizationHookTests(unittest.TestCase):
    def _make_bar(self):
        bar = mock.MagicMock(name="progress_bar")
        bar.n = 0
        bar.total = None
        return bar

    def test_hook_updates_bar_on_segmentation_progress(self):
        bar = self._make_bar()
        hook = diarization_utils._make_diarization_hook(bar)
        hook("segmentation", None, completed=5, total=20)
        self.assertEqual(bar.n, 5)
        self.assertEqual(bar.total, 20)
        bar.refresh.assert_called()

    def test_hook_sets_chunks_postfix_on_segmentation_done(self):
        import numpy as np
        bar = self._make_bar()
        hook = diarization_utils._make_diarization_hook(bar)
        seg = mock.MagicMock()
        seg.data = np.zeros((42, 100, 3))
        hook("segmentation", seg)
        bar.set_postfix_str.assert_called_with("chunks: 42")

    def test_hook_sets_max_speakers_postfix(self):
        import numpy as np
        bar = self._make_bar()
        hook = diarization_utils._make_diarization_hook(bar)
        count = mock.MagicMock()
        count.data = np.array([[0], [2], [1], [3], [2]])
        hook("speaker_counting", count)
        bar.set_postfix_str.assert_called_with("max speakers/frame: 3")

    def test_hook_sets_candidates_postfix_on_embeddings_done(self):
        import numpy as np
        bar = self._make_bar()
        hook = diarization_utils._make_diarization_hook(bar)
        emb = np.zeros((10, 4, 512))
        hook("embeddings", emb)
        bar.set_postfix_str.assert_called_with("candidates: 4")

    def test_hook_sets_speakers_postfix_at_discrete_diarization(self):
        bar = self._make_bar()
        hook = diarization_utils._make_diarization_hook(bar)

        class FakeAnnotation:
            def itertracks(self, yield_label=True):
                yield (None, None, "SPEAKER_00")
                yield (None, None, "SPEAKER_01")
                yield (None, None, "SPEAKER_02")
                yield (None, None, "SPEAKER_00")  # duplicate

        hook("discrete_diarization", FakeAnnotation())
        bar.set_postfix_str.assert_called_with("speakers: 3")

    def test_hook_sets_speakers_for_sliding_window_feature(self):
        import numpy as np

        bar = self._make_bar()
        bar.n = 10
        bar.total = 20
        hook = diarization_utils._make_diarization_hook(bar)

        class FakeSlidingWindowFeature:
            data = np.zeros((10, 100, 3))

        hook("discrete_diarization", FakeSlidingWindowFeature())
        bar.set_postfix_str.assert_called_with("speakers: 3")
        self.assertEqual(bar.n, bar.total)

    def test_hook_completes_bar_at_discrete_diarization(self):
        bar = self._make_bar()
        bar.n = 10
        bar.total = 20
        hook = diarization_utils._make_diarization_hook(bar)

        class FakeAnnotation:
            def itertracks(self, yield_label=True):
                yield (None, None, "SPEAKER_00")

        hook("discrete_diarization", FakeAnnotation())
        self.assertEqual(bar.n, 20)
        bar.refresh.assert_called()

    def test_hook_handles_kwargs_only_step_name(self):
        bar = self._make_bar()
        hook = diarization_utils._make_diarization_hook(bar)
        # simulate pyannote calling without positional args (defensive path)
        hook(step_name="embeddings", completed=7, total=14)
        self.assertEqual(bar.n, 7)
        self.assertEqual(bar.total, 14)
        bar.refresh.assert_called()

    def test_hook_sets_description_to_current_step(self):
        bar = self._make_bar()
        hook = diarization_utils._make_diarization_hook(bar)
        hook("segmentation", None, completed=0, total=10)
        bar.set_description_str.assert_any_call("Diarizing: segmentation")

    def test_hook_updates_description_when_step_changes(self):
        bar = self._make_bar()
        hook = diarization_utils._make_diarization_hook(bar)
        hook("segmentation", None, completed=0, total=770)
        hook("speaker_counting", mock.MagicMock())
        hook("embeddings", None, completed=0, total=73)
        called_with = [c.args[0] for c in bar.set_description_str.call_args_list]
        self.assertIn("Diarizing: segmentation", called_with)
        self.assertIn("Diarizing: speaker_counting", called_with)
        self.assertIn("Diarizing: embeddings", called_with)

    def test_hook_does_not_repeat_description_within_same_step(self):
        bar = self._make_bar()
        hook = diarization_utils._make_diarization_hook(bar)
        hook("segmentation", None, completed=0, total=770)
        hook("segmentation", None, completed=385, total=770)
        hook("segmentation", None, completed=770, total=770)
        segmentation_calls = [
            c for c in bar.set_description_str.call_args_list
            if c.args and c.args[0] == "Diarizing: segmentation"
        ]
        self.assertEqual(len(segmentation_calls), 1)


@unittest.skipIf(diarization_utils is None, "diarization_utils dependencies are unavailable")
class DiarizeAudioTests(unittest.TestCase):
    def test_diarize_audio_invokes_pipeline_and_assigns_speakers(self):
        fake_diarization = mock.MagicMock(name="diarization_output")
        fake_df = mock.MagicMock(name="diarize_df")
        with mock.patch.object(diarization_utils, "ensure_diarization_model", return_value=Path("/fake/model")), \
             mock.patch.object(diarization_utils, "community1_supported", return_value=True), \
             mock.patch.object(diarization_utils, "_load_pipeline") as load_mock, \
             mock.patch.object(diarization_utils, "_diarization_to_dataframe", return_value=fake_df) as convert_mock, \
             mock.patch("whisperx.audio.load_audio", return_value=__import__("numpy").zeros(16000, dtype="float32")), \
             mock.patch("whisperx.diarize.assign_word_speakers", return_value={"segments": []}) as assign_mock:
            pipeline_inst = load_mock.return_value
            pipeline_inst.return_value = fake_diarization
            audio = __import__("numpy").zeros(16000, dtype="float32")
            result = {"segments": []}
            out = diarization_utils.diarize_audio(audio, result, device="cpu", min_speakers=2)
            # ensure_pipeline was called exactly once
            load_mock.assert_called_once()
            # pipeline was applied with audio dict containing waveform + sample_rate
            call_args = pipeline_inst.call_args
            audio_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("audio_data")
            self.assertIsInstance(audio_arg, dict)
            self.assertIn("waveform", audio_arg)
            self.assertIn("sample_rate", audio_arg)
            self.assertEqual(call_args.kwargs.get("min_speakers"), 2)
            # _diarization_to_dataframe was fed the raw pipeline output
            convert_mock.assert_called_once_with(fake_diarization)
            # assign_word_speakers was invoked with the converted df and the result
            assign_mock.assert_called_once_with(fake_df, result)
            # return value is the assign return value
            self.assertIs(out, assign_mock.return_value)

    def test_diarize_audio_passes_hook_to_pipeline(self):
        fake_diarization = mock.MagicMock(name="diarization_output")
        fake_df = mock.MagicMock(name="diarize_df")
        fake_tqdm = mock.MagicMock(name="tqdm_instance")
        with mock.patch.object(diarization_utils, "ensure_diarization_model", return_value=Path("/fake/model")), \
             mock.patch.object(diarization_utils, "community1_supported", return_value=True), \
             mock.patch.object(diarization_utils, "_load_pipeline") as load_mock, \
             mock.patch.object(diarization_utils, "_diarization_to_dataframe", return_value=fake_df), \
             mock.patch.object(diarization_utils, "_should_show_progress_bar", return_value=True), \
             mock.patch.object(diarization_utils, "tqdm", return_value=fake_tqdm) as tqdm_mock, \
             mock.patch("whisperx.audio.load_audio", return_value=__import__("numpy").zeros(16000, dtype="float32")), \
             mock.patch("whisperx.diarize.assign_word_speakers", return_value={"segments": []}):
            pipeline_inst = load_mock.return_value
            pipeline_inst.return_value = fake_diarization
            audio = __import__("numpy").zeros(16000, dtype="float32")
            result = {"segments": []}
            diarization_utils.diarize_audio(audio, result, device="cpu")
            # tqdm was instantiated with the expected kwargs
            tqdm_mock.assert_called_once()
            call_kwargs = tqdm_mock.call_args.kwargs
            self.assertEqual(call_kwargs.get("desc"), "Diarizing")
            self.assertIn("bar_format", call_kwargs)
            # the hook kwarg was passed to the initial pipeline call
            self.assertIn("hook", pipeline_inst.call_args.kwargs)
            # the bar was closed in the finally block
            fake_tqdm.close.assert_called_once()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
