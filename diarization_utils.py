"""Speaker diarization utilities for the LeGen pipeline.

Auto-selects between two pretrained `pyannote` pipelines based on the
installed `pyannote.audio` version:

- ``pyannote.audio >= 4.0``: ``pyannote/speaker-diarization-community-1``
  (SOTA 2025, uses VBxClustering + PLDA + improved segmentation/embedding).
- ``pyannote.audio < 4.0``: ``pyannote/speaker-diarization-3.1`` (legacy)
  with a custom local `config.yaml` using `AgglomerativeClustering`.

Both strategies download their model files from ModelScope
(`modelscope.cn/api/v1/models/pyannote/...`) — no Hugging Face token, no
`HF_TOKEN`, no `--diarize_model_path` flag. Cache lives in
`~/.cache/legen/models/`.

We deliberately bypass `whisperx.diarize.DiarizationPipeline` because that
wrapper uses the old ``use_auth_token=`` keyword argument that was renamed
to ``token=`` in pyannote 4.x; calling it directly would raise a
``TypeError``. We still reuse `whisperx.diarize.assign_word_speakers` for
the word↔speaker overlap assignment (it's pure pandas/numpy, no pyannote API).
"""

from __future__ import annotations

import os
import sys
import urllib.request
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
from tqdm import tqdm
import torch

from utils import time_task


# === ModelScope URLs (token-free, public) ==============================
# community-1 pipeline (recommended, needs pyannote.audio >= 4.0)
_COMMUNITY1_FILE_BASE = (
    "https://modelscope.cn/api/v1/models/pyannote/speaker-diarization-community-1/repo"
    "?Revision=master&FilePath="
)
COMMUNITY1_FILES: dict[str, str] = {
    "config.yaml": f"{_COMMUNITY1_FILE_BASE}config.yaml",
    "segmentation/pytorch_model.bin": f"{_COMMUNITY1_FILE_BASE}segmentation/pytorch_model.bin",
    "embedding/pytorch_model.bin": f"{_COMMUNITY1_FILE_BASE}embedding/pytorch_model.bin",
    "plda/plda.npz": f"{_COMMUNITY1_FILE_BASE}plda/plda.npz",
    "plda/xvec_transform.npz": f"{_COMMUNITY1_FILE_BASE}plda/xvec_transform.npz",
}
COMMUNITY1_SIZES: dict[str, int] = {
    "config.yaml": 444,
    "segmentation/pytorch_model.bin": 5906507,
    "embedding/pytorch_model.bin": 26646242,
    "plda/plda.npz": 133852,
    "plda/xvec_transform.npz": 134376,
}
COMMUNITY1_CACHE_DIR = Path.home() / ".cache" / "legen" / "models" / "diarization-community-1"

# 3.1 legacy fallback (pyannote.audio < 4.0)
MODELSCOPE_SEG_URL = (
    "https://modelscope.cn/api/v1/models/pyannote/segmentation-3.0/repo"
    "?Revision=master&FilePath=pytorch_model.bin"
)
MODELSCOPE_EMB_URL = (
    "https://modelscope.cn/api/v1/models/pyannote/wespeaker-voxceleb-resnet34-LM/repo"
    "?Revision=master&FilePath=pytorch_model.bin"
)
LEGACY_SIZES = {
    "segmentation/pytorch_model.bin": 5905440,
    "embedding/pytorch_model.bin": 26645418,
}
LEGACY_CACHE_DIR = Path.home() / ".cache" / "legen" / "models" / "diarization-3.1"


# === Download helpers ==================================================
MAX_DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT = 30


def _pyannote_version() -> tuple[int, int]:
    """Return ``(major, minor)`` of the installed pyannote.audio, or ``(0, 0)``."""
    try:
        import pyannote.audio
        version_str = getattr(pyannote.audio, "__version__", "0.0.0")
        parts = version_str.split(".")
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except Exception:  # noqa: BLE001
        return (0, 0)


def community1_supported() -> bool:
    """Return True if the installed pyannote.audio can load community-1."""
    return _pyannote_version() >= (4, 0)


def active_cache_dir() -> Path:
    return COMMUNITY1_CACHE_DIR if community1_supported() else LEGACY_CACHE_DIR


def cache_valid(cache_dir: Optional[Path] = None) -> bool:
    """Return True if the cache directory holds a complete, non-truncated model.

    We validate file sizes (not SHA256) because ModelScope does not publish
    official digests; size is sufficient to detect a truncated download.
    """
    cache_dir = cache_dir or active_cache_dir()
    is_community1 = community1_supported()
    expected = COMMUNITY1_SIZES if is_community1 else LEGACY_SIZES
    for rel, expected_size in expected.items():
        path = cache_dir / rel
        if not path.is_file() or path.stat().st_size != expected_size:
            return False
    # 3.1 path also needs our hand-written config.yaml
    if not is_community1:
        if not (cache_dir / "config.yaml").is_file():
            return False
    return True


def _download_file(url: str, dest: Path) -> None:
    """Stream-download `url` to `dest` with a tqdm progress bar.

    Retries up to `MAX_DOWNLOAD_RETRIES` times on network errors. Raises the
    last exception if all attempts fail.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")

    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "legen/0.20"})
            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as response:
                total = int(response.headers.get("Content-Length", 0)) or None
                with open(tmp_path, "wb") as out, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                    leave=False,
                ) as bar:
                    while True:
                        chunk = response.read(64 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        bar.update(len(chunk))
            tmp_path.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < MAX_DOWNLOAD_RETRIES:
                print(
                    f"  Download attempt {attempt} failed ({exc}); retrying...",
                    flush=True,
                )

    raise RuntimeError(
        f"Could not download {url} after {MAX_DOWNLOAD_RETRIES} attempts: {last_exc}"
    ) from last_exc


def _write_legacy_config_yaml(cache_dir: Path) -> Path:
    """Generate the 3.1 pipeline `config.yaml` with absolute local paths.

    The upstream `speaker-diarization-3.1` config references sub-models by
    HF name (gated), so we author our own config pointing at the locally
    cached `.bin` files so `Pipeline.from_pretrained` short-circuits at the
    `os.path.isfile` branch and never touches the Hub.
    """
    config_path = cache_dir / "config.yaml"
    seg_path = (cache_dir / "segmentation" / "pytorch_model.bin").resolve()
    emb_path = (cache_dir / "embedding" / "pytorch_model.bin").resolve()

    config_text = f"""version: 3.1.0

pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: AgglomerativeClustering
    embedding: {emb_path.as_posix()}
    embedding_batch_size: 32
    embedding_exclude_overlap: true
    segmentation: {seg_path.as_posix()}
    segmentation_batch_size: 32

params:
  clustering:
    method: centroid
    min_cluster_size: 12
    threshold: 0.7045654963945799
  segmentation:
    min_duration_off: 0.0
"""
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


def _ensure_community1() -> Path:
    """Download community-1 model files from ModelScope. Returns the cache dir.

    Unlike the 3.1 path, we keep the upstream `config.yaml` untouched (with
    its `$model/{subfolder}` placeholders) because pyannote 4.x's
    `expand_subfolders` resolves those to local file paths automatically.
    """
    if cache_valid(COMMUNITY1_CACHE_DIR):
        return COMMUNITY1_CACHE_DIR

    COMMUNITY1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(
        "Downloading speaker diarization model (~33 MB) from ModelScope...",
        flush=True,
    )
    for rel_path, url in COMMUNITY1_FILES.items():
        _download_file(url, COMMUNITY1_CACHE_DIR / rel_path)

    if not cache_valid(COMMUNITY1_CACHE_DIR):
        raise RuntimeError(
            "Community-1 diarization model cache is invalid after download. "
            "Please try again or check your network connection."
        )
    print("Speaker diarization model ready.", flush=True)
    return COMMUNITY1_CACHE_DIR


def _ensure_legacy_3_1() -> Path:
    """Download 3.1 sub-model files and synthesize a config.yaml. Returns the
    config.yaml path (not a directory) so `Pipeline.from_pretrained` finds it
    via the `os.path.isfile` branch.
    """
    if cache_valid(LEGACY_CACHE_DIR):
        return LEGACY_CACHE_DIR / "config.yaml"

    LEGACY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(
        "Downloading speaker diarization model v3.1 (~33 MB) from ModelScope...",
        flush=True,
    )
    _download_file(MODELSCOPE_SEG_URL, LEGACY_CACHE_DIR / "segmentation" / "pytorch_model.bin")
    _download_file(MODELSCOPE_EMB_URL, LEGACY_CACHE_DIR / "embedding" / "pytorch_model.bin")
    _write_legacy_config_yaml(LEGACY_CACHE_DIR)

    if not cache_valid(LEGACY_CACHE_DIR):
        raise RuntimeError(
            "Speaker diarization v3.1 cache is invalid after download. "
            "Please try again or check your network connection."
        )
    print("Speaker diarization model ready.", flush=True)
    return LEGACY_CACHE_DIR / "config.yaml"


def ensure_diarization_model() -> Path:
    """Ensure the diarization model is cached locally and return the path to
    feed to `Pipeline.from_pretrained`.

    - On pyannote.audio >= 4.0: returns the cache *directory* (which contains
      the upstream `config.yaml` with `$model/` placeholders).
    - On pyannote.audio < 4.0: returns the path to our hand-written
      `config.yaml` file with absolute local paths.
    """
    if community1_supported():
        return _ensure_community1()
    return _ensure_legacy_3_1()


def _load_pipeline(model_path: Path, device: Union[str, torch.device]):
    """Instantiate a `pyannote.audio` Pipeline from a local path, accounting
    for the kwargs rename `use_auth_token` → `token` in pyannote 4.x."""
    from pyannote.audio import Pipeline

    if community1_supported():
        pipeline = Pipeline.from_pretrained(model_path)
    else:
        # pyannote 3.x accepts `use_auth_token=`; 4.x renamed it to `token=`.
        try:
            pipeline = Pipeline.from_pretrained(model_path, use_auth_token=None)
        except TypeError:
            pipeline = Pipeline.from_pretrained(model_path, token=None)

    if hasattr(pipeline, "to"):
        pipeline.to(device)
    return pipeline


def _diarization_to_dataframe(diarization):
    """Convert a pyannote diarization output into the pandas DataFrame shape
    expected by `whisperx.diarize.assign_word_speakers`.

    Handles both shape (pyannote 3.x returns an Annotation directly, pyannote
    4.x returns a `DiarizeOutput` with a `.speaker_diarization` attribute).
    """
    import pandas as pd

    annotation = getattr(diarization, "speaker_diarization", diarization)
    df = pd.DataFrame(
        annotation.itertracks(yield_label=True),
        columns=["segment", "label", "speaker"],
    )
    df["start"] = df["segment"].apply(lambda x: x.start)
    df["end"] = df["segment"].apply(lambda x: x.end)
    return df


def _should_show_progress_bar() -> bool:
    """Return True if the diarization progress bar should render.

    The bar only renders on interactive TTYs with at least 50 columns
    of terminal width. In notebooks, CI logs, or narrow terminals the
    hook still runs (so internal state is consistent) but the bar is
    hidden via tqdm's `disable=True`.
    """
    if not sys.stdout.isatty():
        return False
    try:
        return os.get_terminal_size().columns >= 50
    except OSError:
        return False


def _make_diarization_hook(progress_bar):
    """Build a pyannote pipeline hook that drives `progress_bar`.

    Handles the three call patterns emitted by pyannote.audio:
      1. `hook("step", artifact, completed=N, total=M)` — progress update
      2. `hook("step", artifact)` — step completed; derive postfix info
      3. `hook(completed=0, total=N)` (curried via partial) — initial 0%

    Updates `progress_bar.set_description_str` to the current step name so the
    user can see which pyannote phase is running (each step has its own total,
    so without the step name the bar appears to "jump back" at every boundary).
    """
    last_step = [None]

    def hook(*args, **kwargs):
        step_name = args[0] if args else kwargs.get("step_name")
        step_artefact = args[1] if len(args) > 1 else kwargs.get("step_artefact")
        completed = kwargs.get("completed")
        total = kwargs.get("total")

        if step_name != last_step[0]:
            last_step[0] = step_name
            progress_bar.set_description_str(
                f"Diarizing: {step_name}" if step_name else "Diarizing"
            )
            progress_bar.refresh()

        if step_name == "segmentation":
            if completed is not None and total is not None:
                progress_bar.n = completed
                progress_bar.total = total
                progress_bar.set_postfix_str("")
                progress_bar.refresh()
            elif step_artefact is not None:
                num_chunks = step_artefact.data.shape[0]
                progress_bar.set_postfix_str(f"chunks: {num_chunks}")
        elif step_name == "speaker_counting" and step_artefact is not None:
            try:
                peak = int(np.nanmax(step_artefact.data))
            except (ValueError, TypeError):
                peak = 0
            progress_bar.set_postfix_str(f"max speakers/frame: {peak}")
        elif step_name == "embeddings":
            if completed is not None and total is not None:
                progress_bar.n = completed
                progress_bar.total = total
                progress_bar.set_postfix_str("")
                progress_bar.refresh()
            elif step_artefact is not None:
                candidates = step_artefact.shape[1]
                progress_bar.set_postfix_str(f"candidates: {candidates}")
        elif step_name == "discrete_diarization" and step_artefact is not None:
            if hasattr(step_artefact, "itertracks"):
                speakers = {
                    label
                    for _, _, label in step_artefact.itertracks(yield_label=True)
                }
                speaker_count = len(speakers)
            else:
                # pyannote 4.x emits a 2-D SlidingWindowFeature before Annotation conversion.
                try:
                    data = np.asarray(getattr(step_artefact, "data", None))
                    speaker_count = int(data.shape[1]) if data.ndim == 2 else 0
                except (TypeError, ValueError):
                    speaker_count = 0
            progress_bar.set_postfix_str(f"speakers: {speaker_count}")
            if progress_bar.total and progress_bar.n < progress_bar.total:
                progress_bar.n = progress_bar.total
                progress_bar.refresh()
    return hook


def diarize_audio(
    audio,
    result,
    *,
    device: Union[str, torch.device] = "cpu",
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
):
    """Run speaker diarization and attach speaker labels to transcript segments.

    Parameters
    ----------
    audio : np.ndarray or str
        Audio waveform (16 kHz mono) or path to a wav file. Accepts whatever
        `pyannote.audio.Pipeline.__call__` accepts.
    result : dict
        WhisperX/Whisper transcription result with `segments` (and optionally
        `words` per segment). Mutated in place: each segment (and each word,
        when timings are available) receives a `speaker` field.
    device : str or torch.device
        Torch device for the diarization pipeline. Falls back to CPU if the
        GPU runs out of VRAM.
    min_speakers, max_speakers : int, optional
        Hints for the clustering step. When known, supplying these greatly
        improves accuracy.

    Returns
    -------
    dict
        The same `result` object, with `speaker` labels assigned.
    """
    # PyTorch >= 2.6 defaults `torch.load(weights_only=True)`. The pyannote
    # checkpoints predate this change; relax the policy so loading works.
    # pyannote 4.x already passes `weights_only=False` explicitly but we set
    # the env var defensively for the 3.1 fallback path.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    model_path = ensure_diarization_model()

    if isinstance(device, str):
        device = torch.device(device)

    from whisperx.audio import load_audio as _load_audio_file, SAMPLE_RATE

    # Pyannote 4.x's I/O defaults to torchcodec (which needs CUDA libs even
    # for CPU decoding). Bypass that by always passing audio as a waveform
    # dict — exactly what `whisperx.diarize.DiarizationPipeline` used to do.
    if isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio:
        audio_data = audio
    elif isinstance(audio, str):
        waveform = _load_audio_file(audio)
        audio_data = {
            "waveform": torch.from_numpy(waveform[None, :]),
            "sample_rate": SAMPLE_RATE,
        }
    else:  # assume np.ndarray or torch.Tensor — wrap with channel dim
        waveform = audio
        if isinstance(waveform, torch.Tensor):
            audio_data = {
                "waveform": waveform[None, :] if waveform.dim() == 1 else waveform,
                "sample_rate": SAMPLE_RATE,
            }
        else:
            audio_data = {
                "waveform": torch.from_numpy(waveform[None, :]),
                "sample_rate": SAMPLE_RATE,
            }

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"std\(\): degrees of freedom is <= 0",
            category=UserWarning,
        )
        with time_task("Running diarization...", end="\n"):
            show_bar = _should_show_progress_bar()
            bar_format = "{desc} {percentage:3.0f}% | {n_fmt}/{total_fmt} | ETA: {remaining} | ⏱: {elapsed}"
            progress_bar = tqdm(
                desc="Diarizing",
                bar_format=bar_format,
                disable=not show_bar,
                dynamic_ncols=True,
                leave=True,
            )
            hook = _make_diarization_hook(progress_bar)
            try:
                try:
                    pipeline = _load_pipeline(model_path, device)
                    diarization = pipeline(
                        audio_data,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                        hook=hook,
                    )
                except RuntimeError as exc:
                    msg = str(exc).lower()
                    if "out of memory" not in msg and "cuda" not in msg:
                        raise
                    print("  GPU out of memory for diarization; falling back to CPU.", flush=True)
                    pipeline = _load_pipeline(model_path, torch.device("cpu"))
                    diarization = pipeline(
                        audio_data,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                        hook=hook,
                    )
            finally:
                progress_bar.close()

        diarize_df = _diarization_to_dataframe(diarization)
        from whisperx.diarize import assign_word_speakers
        result = assign_word_speakers(diarize_df, result)

    return result
