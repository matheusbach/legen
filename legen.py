from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import warnings
import logging

# Filter warnings early
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"pyannote\.database.*")
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=UserWarning, message="torchaudio._backend.list_audio_backends has been deprecated",)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from inspect import currentframe, getframeinfo
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

import download_utils
import ffmpeg_utils
import file_utils
import subtitle_utils
import translate_utils
from gemini_utils import GeminiSummaryConfig, generate_tltw, normalize_api_keys
from utils import audio_extensions, check_other_extensions, split_lang_suffix, time_task, video_extensions

# Fix for matplotlib backend issue in some environments (e.g. Colab)
if os.environ.get("MPLBACKEND") == "module://matplotlib_inline.backend_inline":
    os.environ.pop("MPLBACKEND")

VERSION = "0.20.0"
version = f"v{VERSION}"
__version__ = VERSION
__all__ = [
    "VERSION",
    "__version__",
    "build_parser",
    "looks_like_url",
    "main",
    "normalize_subtitle_formats",
]

SUPPORTED_SUBTITLE_FORMATS = {"srt", "txt"}


def looks_like_url(value: str) -> bool:
    if not value:
        return False
    parsed = urlparse(str(value))
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def normalize_subtitle_formats(raw_value):
    if isinstance(raw_value, (list, tuple)):
        values = raw_value
    else:
        values = [raw_value]

    formats = []
    for value in values:
        if value is None:
            continue
        parts = str(value).replace(",", " ").split()
        for part in parts:
            fmt = part.strip().lower()
            if fmt:
                formats.append(fmt)

    if not formats:
        formats = ["srt"]

    # maintain user order while removing duplicates
    seen = {}
    for fmt in formats:
        seen.setdefault(fmt, None)

    return list(seen.keys())

# Terminal colors
default = "\033[1;0m"
gray = "\033[1;37m"
wblue = "\033[1;36m"
blue = "\033[1;34m"
yellow = "\033[1;33m"
green = "\033[1;32m"
red = "\033[1;31m"


def _print_banner() -> None:
    banner = f"""
{blue}888              {gray} .d8888b.                   
{blue}888              {gray}d88P  Y88b                  
{blue}888              {gray}888    888                  
{blue}888      .d88b.  {gray}888         .d88b.  88888b. 
{blue}888     d8P  Y8b {gray}888  88888 d8P  Y8b 888 "88b
{blue}888     88888888 {gray}888    888 88888888 888  888
{blue}888     Y8b.     {gray}Y88b  d88P Y8b.     888  888
{blue}88888888 "Y8888  {gray} "Y8888P88  "Y8888  888  888

legen {version} - github.com/matheusbach/legen{default}
python {sys.version}
"""
    print(banner)
    time.sleep(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="LeGen", description="Uses AI to locally transcribes speech from media files, generating subtitle files, translates the generated subtitles, inserts them into the mp4 container, and burns them directly into video",
                                     argument_default=True, allow_abbrev=True, add_help=True, usage='LeGen -i INPUT_PATH [other options]')
    parser.add_argument("-i", "--input_path",
                        help="Local media path or URL to download before processing", required=True, type=str)
    parser.add_argument("--norm", default=False, action="store_true",
                        help="Normalize folder times and run vidqa on input_path before starting processing files")
    parser.add_argument("-ts:e", "--transcription_engine", type=str, default="whisperx",
                        help="Transcription engine. Possible values: whisperx (default), whisper")
    parser.add_argument("-ts:m", "--transcription_model", type=str, default="large-v3-turbo",
                        help="Path or name of the Whisper transcription model. A larger model will consume more resources and be slower, but with better transcription quality. Possible values: tiny, base, small, medium, large, turbo, large-v3-turbo (default) ...")
    parser.add_argument("-ts:d", "--transcription_device", type=str, default="auto",
                        help="Device to run the transcription through Whisper. Possible values: auto (default), cpu, cuda")
    parser.add_argument("-ts:c", "--transcription_compute_type", type=str, default="auto",
                        help="Quantization for the neural network. Possible values: auto (default), int8, int8_float32, int8_float16, int8_bfloat16, int16, float16, bfloat16, float32")
    parser.add_argument("-ts:v", "--transcription_vad", type=str.lower, default="silero", choices=["pyannote", "silero"],
                        help="Voice activity detector to segment audio before transcription when using whisperx. Defaults to silero (CPU friendly).")
    parser.add_argument("-ts:b", "--transcription_batch", type=int, default=4,
                        help="Number of simultaneous segments being transcribed. Higher values will speed up processing. If you have low RAM/VRAM, long duration media files or have buggy subtitles, reduce this value to avoid issues. Only works using transcription_engine whisperx. (default: 4)")
    parser.add_argument("--translate", type=str, default="none",
                        help="Translate subtitles to language code if not the same as origin. (default: don't translate)")
    parser.add_argument("--translate_engine", type=str, default="google",
                        help="Translation engine to use: google (default) or gemini")
    parser.add_argument("--gemini_api_key", action="append", default=[], type=str,
                        help="Gemini API key. Repeat or separate by comma/line break to add multiple keys (required if --translate_engine=gemini)")
    parser.add_argument("--tltw", action="store_true", default=False,
                        help="Generate a Gemini 'Too Long To Watch' summary from subtitles (requires --gemini_api_key)")
    parser.add_argument("--output_tltw", type=Path, default=None,
                        help="Directory to save TLTW summaries. Defaults to the softsubs output folder")
    parser.add_argument("--input_lang", type=str, default="auto",
                        help="Indicates (forces) the language of the voice in the input media (default: auto)")
    parser.add_argument("-c:v", "--codec_video", type=str, default="h264", metavar="VIDEO_CODEC",
                        help="Target video codec. Can be used to set acceleration via GPU or another video API [codec_api], if supported (ffmpeg -encoders). Ex: h264, libx264, h264_vaapi, h264_nvenc, hevc, libx265 hevc_vaapi, hevc_nvenc, hevc_cuvid, hevc_qsv, hevc_amf (default: h264)")
    parser.add_argument("-c:a", "--codec_audio", type=str, default="aac", metavar="AUDIO_CODEC",
                        help="Target audio codec. (default: aac). Ex: aac, libopus, mp3, vorbis")
    parser.add_argument("-o:s", "--output_softsubs", default=None, type=Path,
                        help="Path to the folder or output file for the video files with embedded softsub (embedded in the mp4 container and .srt files). (default: softsubs_ + input_path)")
    parser.add_argument("-o:h", "--output_hardsubs", default=None, type=Path,
                        help="Output folder path for video files with burned-in captions and embedded in the mp4 container. (default: hardsubs_ + input_path)")
    parser.add_argument("-o:d", "--output_downloads", default=None, type=Path,
                        help="Destination folder for videos downloaded from URL inputs. (default: ./downloads)")
    parser.add_argument("--overwrite", default=False, action="store_true",
                        help="Overwrite existing files in output directories")
    parser.add_argument("-dl:rs", "--download_remote_subs", default=False, action="store_true",
                        help="When using a URL input, also download and embed remote subtitle tracks (disabled by default)")
    parser.add_argument("--disable_srt", default=False, action="store_true",
                        help="Disable .srt file generation and don't insert subtitles in mp4 container of output_softsubs")
    parser.add_argument("--subtitle_formats", type=str, default="srt",
                        help="Subtitle formats to export (separate multiple options with comma or space). Supported: srt, txt")
    parser.add_argument("--disable_softsubs", default=False, action="store_true",
                        help="Don't insert subtitles in mp4 container of output_softsubs. This option continues generating .srt files")
    parser.add_argument("--disable_hardsubs", default=False, action="store_true",
                        help="Disable subtitle burn in output_hardsubs")
    parser.add_argument("--copy_files", default=False, action="store_true",
                        help="Copy other (non-video) files present in input directory to output directories. Only generate the subtitles and videos")
    parser.add_argument(
        "--process_input_subs",
        "--process_srt_inputs",
        dest="process_input_subs",
        default=False,
        action="store_true",
        help="Also process existing .srt subtitle files found in the input path (translate/TLTW). When a subtitle matches a media filename, it is used instead of transcription.",
    )
    return parser


def patch_torch_hub():
    """
    Monkeypatch torch.hub.load to add retries for transient errors (like 503).
    """
    try:
        import torch.hub
        import time
        from urllib.error import HTTPError
        
        original_load = torch.hub.load

        def retrying_load(*args, **kwargs):
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    return original_load(*args, **kwargs)
                except Exception as e:
                    # Check for 503 or other transient errors
                    is_transient = False
                    error_str = str(e)
                    if isinstance(e, HTTPError) and e.code in [500, 502, 503, 504]:
                        is_transient = True
                    elif "503" in error_str or "504" in error_str or "Connection reset" in error_str:
                        is_transient = True
                    
                    if is_transient and attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"Download failed with {e}, retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    raise e
        
        torch.hub.load = retrying_load
    except ImportError:
        pass


def main(argv: Sequence[str] | None = None) -> int:
    _print_banner()
    parser = build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv if argv is not None else None)

    input_path_raw = args.input_path

    if looks_like_url(input_path_raw):
        download_destination = args.output_downloads or Path.cwd() / "downloads"
        args.output_downloads = Path(download_destination).expanduser().resolve()
        try:
            with time_task(message_start=f"\nDownloading media with yt-dlp to {gray}{args.output_downloads}{default}", end="\n"):
                downloaded_files = download_utils.download_urls(
                    [input_path_raw],
                    args.output_downloads,
                    overwrite=args.overwrite,
                    download_remote_subs=args.download_remote_subs,
                )
        except FileNotFoundError as exc:
            parser.error(str(exc))
        except RuntimeError as exc:
            parser.error(str(exc))

        print(f"Downloaded {len(downloaded_files)} file(s) with {gray}yt-dlp{default}. Continuing with local processing from {gray}{args.output_downloads}{default}.")
        args.input_path = args.output_downloads
    else:
        candidate_path = Path(input_path_raw).expanduser().resolve()
        if not candidate_path.exists():
            parser.error(f"Input path '{input_path_raw}' is neither an existing file/folder nor a downloadable URL.")
        args.input_path = candidate_path

    if args.output_downloads is not None:
        args.output_downloads = Path(args.output_downloads).expanduser().resolve()

    requested_formats = normalize_subtitle_formats(args.subtitle_formats)
    unsupported_formats = [fmt for fmt in requested_formats if fmt not in SUPPORTED_SUBTITLE_FORMATS]
    if unsupported_formats:
        parser.error(f"Unsupported subtitle format(s): {', '.join(unsupported_formats)}. Supported formats: {', '.join(sorted(SUPPORTED_SUBTITLE_FORMATS))}")

    if args.disable_srt:
        requested_formats = [fmt for fmt in requested_formats if fmt != "srt"]

    if not requested_formats and not args.disable_srt:
        requested_formats = ["srt"]

    args.subtitle_formats = requested_formats
    args.disable_srt = "srt" not in args.subtitle_formats
    args.export_txt = "txt" in args.subtitle_formats

    def export_txt_if_requested(source_path: Path, target_path: Path):
        if not args.export_txt:
            return
        if source_path is None:
            return
        if not args.overwrite and file_utils.file_is_valid(target_path):
            return
        subtitle_utils.export_plain_text_from_srt(source_path, target_path)

    args.gemini_api_keys = normalize_api_keys(args.gemini_api_key)
    args.gemini_api_key = args.gemini_api_keys[0] if args.gemini_api_keys else None

    translate_engine_explicit = any(str(item).startswith("--translate_engine") for item in raw_argv)
    if (
        not translate_engine_explicit
        and args.translate
        and str(args.translate).lower() != "none"
        and args.translate_engine == "google"
        and args.gemini_api_keys
    ):
        # If the user provided Gemini keys but did not explicitly choose an engine,
        # prefer Gemini translation.
        args.translate_engine = "gemini"

    if args.tltw and not args.gemini_api_keys:
        parser.error("Gemini API key is required for TLTW summaries. Provide --gemini_api_key.")

    if args.translate_engine == "gemini" and not args.gemini_api_keys:
        parser.error("Gemini API key is required when --translate_engine=gemini. Provide --gemini_api_key.")

    if args.translate_engine == "gemini" and args.translate.lower() == 'pt':
        args.translate = 'pt-BR'
    if args.translate_engine == "google" and args.translate.lower() in ['pt', 'pt-br', 'pt-pt']:
        args.translate = 'pt'

    if not args.output_softsubs:
        if args.input_path.is_file():
            args.output_softsubs = Path(args.input_path.parent, "softsubs")
        else:
            args.output_softsubs = compatibility_path if (compatibility_path := Path(args.input_path.parent, "legen_srt_" + args.input_path.name)).exists() else Path(args.input_path.parent, "softsubs_" + args.input_path.name)
    if not args.output_hardsubs:
        if args.input_path.is_file():
            args.output_hardsubs = Path(args.input_path.parent, "hardsubs")
        else:
            args.output_hardsubs = compatibility_path if (compatibility_path := Path(args.input_path.parent, "legen_burned_" + args.input_path.name)).exists() else Path(args.input_path.parent, "hardsubs_" + args.input_path.name)

    if args.output_tltw:
        args.output_tltw = Path(args.output_tltw).expanduser().resolve()
    else:
        args.output_tltw = args.output_softsubs

    device_info = None
    resolved_compute_type = None

    if args.transcription_device == "auto":
        # centralized, more robust device detection
        from device_utils import select_torch_device

        try:
            device_info = select_torch_device(
                preferred="auto",
                model_name=args.transcription_model,
                compute_type=args.transcription_compute_type,
            )
        except Exception as exc:
            print(f"{yellow}Device detection failed ({exc}). Falling back to CPU.{default}")
            torch_device = "cpu"
        else:
            torch_device = device_info.backend
            resolved_compute_type = device_info.resolved_compute_type
            for message in device_info.messages:
                print(message)
            for issue in device_info.issues:
                print(f"{yellow}{issue}{default}")
            for note in device_info.notes:
                print(f"{gray}{note}{default}")

        # If we have PyTorch and CUDA was selected, try to enable TF32 where
        # possible for a small speedup on Ampere+ hardware. Keep this best-effort
        # and ignore any failures.
        try:
            import torch

            if torch_device.startswith("cuda"):
                matmul_backend = getattr(torch.backends.cuda, "matmul", None)
                if matmul_backend is not None and hasattr(matmul_backend, "fp32_precision"):
                    matmul_backend.fp32_precision = "tf32"
                elif matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
                    matmul_backend.allow_tf32 = True

                cudnn_conv_backend = getattr(torch.backends.cudnn, "conv", None)
                if cudnn_conv_backend is not None and hasattr(cudnn_conv_backend, "fp32_precision"):
                    cudnn_conv_backend.fp32_precision = "tf32"
                elif hasattr(torch.backends.cudnn, "allow_tf32"):
                    torch.backends.cudnn.allow_tf32 = True
        except Exception:
            # ignore failures in optional PyTorch tuning
            pass
    else:
        torch_device = str(args.transcription_device).lower()

    if resolved_compute_type:
        transcription_compute_type = resolved_compute_type
    elif args.transcription_compute_type in {"auto", "default"}:
        if torch_device.startswith("cuda"):
            transcription_compute_type = "float16"
        elif torch_device == "mps":
            transcription_compute_type = "float16"
        else:
            transcription_compute_type = "float32"
    else:
        transcription_compute_type = args.transcription_compute_type

    args.transcription_model = "large-v3" if args.transcription_model == "large" else args.transcription_model

    if args.norm:
        vidqa_executable = "vidqa"
        local_vidqa = Path(sys.executable).parent / "vidqa"
        if local_vidqa.exists() and os.access(local_vidqa, os.X_OK):
            vidqa_executable = str(local_vidqa)

        with time_task(message_start=f"Running {wblue}vidqa{default} and updating folder modifiation times in {gray}{args.input_path}{default}", end="\n"):
            subprocess.run([vidqa_executable, "-i", args.input_path, "-m", "unique", "-fd",
                            Path(Path(getframeinfo(currentframe()).filename).resolve().parent, "vidqa_data")])
            file_utils.update_folder_times(args.input_path)

    with time_task(message_start=f"\nLoading {args.transcription_engine} model: {wblue}{args.transcription_model}{default} ({transcription_compute_type}) on {wblue}{torch_device}{default}", end="\n"):
        if args.transcription_engine == 'whisperx':
            patch_torch_hub()
            import whisperx_legen_fork as whisperx
            import whisperx_utils

            whisper_model = whisperx.load_model(
                whisper_arch=args.transcription_model,
                device=torch_device,
                compute_type=transcription_compute_type,
                vad_method=args.transcription_vad,
                asr_options={"repetition_penalty": 1, "prompt_reset_on_temperature": 0.5, "no_repeat_ngram_size": 2,},
            )
        elif args.transcription_engine == 'whisper':
            import whisper

            import whisper_utils
            whisper_model = whisper.load_model(
                name=args.transcription_model, device=torch_device, in_memory=True)
        else:
            raise ValueError(f'Unsupported transcription engine {args.transcription_engine}. Supported values: whisperx, whisper')

    with time_task(message="⌛ Processing files for"):
        # Pre-scan subtitle inputs so we can (a) use them instead of transcription
        # for matching media, and (b) process standalone subtitle files.
        srt_index: dict[tuple[Path, str], list[Path]] = {}
        srt_linked_to_media: set[Path] = set()

        if args.process_input_subs and args.input_path.is_dir():
            media_keys: set[tuple[Path, str]] = set()
            for media_path in args.input_path.rglob("*"):
                if not media_path.is_file():
                    continue
                ext = media_path.suffix.lower()
                if ext not in (video_extensions | audio_extensions):
                    continue
                rel_media = media_path.relative_to(args.input_path)
                media_keys.add((rel_media.parent, rel_media.stem))

            for srt_path in args.input_path.rglob("*.srt"):
                if not srt_path.is_file():
                    continue
                rel_srt = srt_path.relative_to(args.input_path)
                base_stem, _ = split_lang_suffix(rel_srt.stem)
                key = (rel_srt.parent, base_stem)
                srt_index.setdefault(key, []).append(srt_path)
                if key in media_keys:
                    srt_linked_to_media.add(srt_path)

            # deterministic selection order
            for key in list(srt_index.keys()):
                srt_index[key] = sorted(srt_index[key])

        path: Path
        if args.input_path.is_file():
            files_iterator = [args.input_path]
        else:
            files_iterator = (item for item in sorted(sorted(Path(args.input_path).rglob('*'), key=lambda x: x.stat().st_mtime), key=lambda x: len(x.parts)) if item.is_file())

        for path in files_iterator:
            if args.input_path.is_file():
                rel_path = Path(path.name)
            else:
                rel_path = path.relative_to(args.input_path)
            with time_task(message_start=f"\nProcessing {yellow}{rel_path.as_posix()}{default}", end="\n", message="⌚ Done in"):
                try:
                    if path.suffix.lower() in video_extensions:
                        file_type = "video"
                    elif path.suffix.lower() in audio_extensions:
                        file_type = "audio"
                    elif args.process_input_subs and path.suffix.lower() == ".srt":
                        file_type = "subtitle"
                    else:
                        file_type = "other"

                    if file_type == "video" or file_type == "audio":
                        origin_media_path = path
                        dupe_filename = len(check_other_extensions(path, list(video_extensions | audio_extensions))) > 1
                        posfix_extension = path.suffix.lower().replace('.', '_') if dupe_filename else ''

                        softsub_video_dir = Path(args.output_softsubs, rel_path.parent)
                        burned_video_dir = Path(args.output_hardsubs, rel_path.parent)
                        softsub_video_path = Path(args.output_softsubs, rel_path.stem + posfix_extension + ".mp4")
                        hardsub_video_path = Path(burned_video_dir, rel_path.stem + posfix_extension + ".mp4")
                        subtitle_translated_path = Path(
                            softsub_video_dir, rel_path.stem + posfix_extension + f"_{args.translate.lower()}.srt")
                        subtitles_path = []

                        # If enabled, prefer an existing subtitle file matching this media name
                        linked_srt: Path | None = None
                        linked_srt_lang: str | None = None
                        if args.process_input_subs and args.input_path.is_dir():
                            candidates = srt_index.get((rel_path.parent, rel_path.stem), [])
                            if candidates:
                                linked_srt = candidates[0]
                                _, linked_srt_lang = split_lang_suffix(linked_srt.stem)

                        if linked_srt is not None:
                            if args.input_lang != "auto":
                                audio_language = args.input_lang
                            else:
                                audio_language = linked_srt_lang or "auto"
                            subtitle_transcribed_path = Path(
                                softsub_video_dir, rel_path.stem + posfix_extension + f"_{audio_language.lower()}.srt")
                            transcribed_srt_temp = file_utils.TempFile(
                                subtitle_transcribed_path, file_ext=".srt")

                            if file_utils.file_is_valid(subtitle_transcribed_path) and not args.overwrite:
                                print(f"Existing subtitle file {gray}{subtitle_transcribed_path}{default}. Skipping copy.")
                            else:
                                file_utils.copy_file_if_different(linked_srt, transcribed_srt_temp.getpath(), silent=True)
                                if not args.disable_srt:
                                    transcribed_srt_temp.save()
                        else:
                            if args.input_lang == "auto":
                                audio_short_extracted = file_utils.TempFile(
                                    None, file_ext=".wav")
                                ffmpeg_utils.extract_short_wav(
                                    origin_media_path, audio_short_extracted.getpath())
                                print("Detecting audio language: ", end='', flush=True)
                                if args.transcription_engine == 'whisperx':
                                    audio_language = whisperx_utils.detect_language(
                                        whisper_model, audio_short_extracted.getpath())
                                if args.transcription_engine == 'whisper':
                                    audio_language = whisper_utils.detect_language(
                                        whisper_model, audio_short_extracted.getpath())
                                print(f"{gray}{audio_language}{default}")

                                audio_short_extracted.destroy()
                            else:
                                audio_language = args.input_lang
                                print(f"Forced input audio language: {gray}{audio_language}{default}")
                            subtitle_transcribed_path = Path(
                                softsub_video_dir, rel_path.stem + posfix_extension + f"_{audio_language.lower()}.srt")
                            transcribed_srt_temp = file_utils.TempFile(
                                subtitle_transcribed_path, file_ext=".srt")
                            if (file_utils.file_is_valid(subtitle_transcribed_path)) or ((args.disable_hardsubs or file_utils.file_is_valid(hardsub_video_path)) and (args.disable_srt or file_utils.file_is_valid(subtitle_transcribed_path))) and not args.overwrite:
                                print("Transcription is unnecessary. Skipping.")
                            else:
                                audio_extracted = file_utils.TempFile(None, file_ext=".wav")
                                ffmpeg_utils.extract_audio_wav(
                                    origin_media_path, audio_extracted.getpath())
                                if args.transcription_engine == 'whisperx':
                                    print(f"{wblue}Transcribing{default} with {gray}WhisperX{default}")
                                    whisperx_utils.transcribe_audio(
                                        whisper_model, audio_extracted.getpath(), transcribed_srt_temp.getpath(), audio_language, device=torch_device, batch_size=args.transcription_batch)
                                if args.transcription_engine == 'whisper':
                                    print(f"{wblue}Transcribing{default} with {gray}Whisper{default}")
                                    whisper_utils.transcribe_audio(
                                        model=whisper_model, audio_path=audio_extracted.getpath(), srt_path=transcribed_srt_temp.getpath(), lang=audio_language, disable_fp16=False if transcription_compute_type == "float16" or transcription_compute_type == "fp16" else True)

                                audio_extracted.destroy()
                                if not args.disable_srt:
                                    transcribed_srt_temp.save()
                        transcribed_srt_source_path = transcribed_srt_temp.getvalidpath()
                        if transcribed_srt_source_path:
                            subtitles_path.append(transcribed_srt_source_path)
                        if args.translate == "none":
                            translated_srt_source_path = None
                        elif audio_language != "auto" and args.translate == audio_language:
                            print("Translation is unnecessary because input and output language are the same. Skipping.")
                            translated_srt_source_path = None
                        elif (args.disable_hardsubs or file_utils.file_is_valid(hardsub_video_path)) and (args.disable_srt or (file_utils.file_is_valid(subtitle_translated_path) and file_utils.file_is_valid(subtitle_transcribed_path) and file_utils.file_is_valid(subtitle_translated_path))) and not args.overwrite:
                            print("Translation is unnecessary. Skipping.")
                            subtitles_path.insert(0, subtitle_translated_path)
                            translated_srt_source_path = subtitle_translated_path if file_utils.file_is_valid(subtitle_translated_path) else None
                        elif file_utils.file_is_valid(subtitle_translated_path):
                            print("Translated file found. Skipping translation.")
                            subtitles_path.insert(0, subtitle_translated_path)
                            translated_srt_source_path = subtitle_translated_path
                        elif transcribed_srt_temp.getvalidpath():
                            translated_srt_temp = file_utils.TempFile(
                                subtitle_translated_path, file_ext=".srt")

                            print(f"{wblue}Translating{default} with {gray}{args.translate_engine.capitalize()}{default} to {gray}{args.translate}{default}")
                            translate_utils.translate_srt_file(
                                transcribed_srt_temp.getvalidpath(),
                                translated_srt_temp.getpath(),
                                args.translate,
                                translate_engine=args.translate_engine,
                                gemini_api_keys=args.gemini_api_keys,
                                overwrite=args.overwrite
                            )
                            if not args.disable_srt:
                                translated_srt_temp.save()

                            translated_srt_source_path = translated_srt_temp.getvalidpath()
                            subtitles_path.insert(0, translated_srt_source_path)
                        else:
                            translated_srt_source_path = None
                        if args.export_txt and transcribed_srt_source_path:
                            export_txt_if_requested(transcribed_srt_source_path, subtitle_transcribed_path.with_suffix(".txt"))
                        if args.export_txt and translated_srt_source_path:
                            export_txt_if_requested(translated_srt_source_path, subtitle_translated_path.with_suffix(".txt"))
                        if args.tltw:
                            summary_source_path = translated_srt_source_path or transcribed_srt_source_path
                            summary_language = "auto-detect" if audio_language == "auto" else audio_language
                            if summary_source_path and translated_srt_source_path and summary_source_path == translated_srt_source_path and args.translate.lower() != "none":
                                summary_language = args.translate

                            if summary_source_path:
                                summary_output_dir = Path(args.output_tltw, rel_path.parent)
                                summary_filename = f"{rel_path.stem + posfix_extension}_tltw_{str(summary_language).lower()}.md"
                                summary_output_path = summary_output_dir / summary_filename

                                if file_utils.file_is_valid(summary_output_path) and not args.overwrite:
                                    print(f"Existing TLTW summary {gray}{summary_output_path}{default}. Skipping.")
                                else:
                                    print(f"{wblue}Generating TLTW summary{default} with {gray}Gemini{default}")
                                    generate_tltw(
                                        GeminiSummaryConfig(
                                            api_keys=args.gemini_api_keys,
                                            subtitle_file=summary_source_path,
                                            output_file=summary_output_path,
                                            language=summary_language,
                                        )
                                    )
                            else:
                                print("No subtitles available for TLTW summary. Skipping.")
                        if not args.disable_softsubs:
                            if file_utils.file_is_valid(softsub_video_path) and not args.overwrite:
                                print(f"Existing video file {gray}{softsub_video_path}{default}. Skipping subtitle insert")
                            else:
                                video_softsubs_temp = file_utils.TempFile(
                                    softsub_video_path, file_ext=".mp4")

                                print(f"{wblue}Inserting subtitle{default} in mp4 container using {gray}FFmpeg{default}")
                                ffmpeg_utils.insert_subtitle(input_media_path=origin_media_path, subtitles_path=subtitles_path,
                                                            burn_subtitles=False, output_video_path=video_softsubs_temp.getpath(),
                                                            codec_video=args.codec_video, codec_audio=args.codec_audio)
                                video_softsubs_temp.save()
                        if not args.disable_hardsubs:
                            if file_utils.file_is_valid(hardsub_video_path) and not args.overwrite:
                                print(f"Existing video file {gray}{hardsub_video_path}{default}. Skipping subtitle burn")
                            else:
                                video_hardsubs_temp = file_utils.TempFile(
                                    hardsub_video_path, file_ext=".mp4")
                                print(f"{wblue}Inserting subtitle{default} in mp4 container and {wblue}burning{default} using {gray}FFmpeg{default}")
                                ffmpeg_utils.insert_subtitle(input_media_path=origin_media_path, subtitles_path=subtitles_path,
                                                            burn_subtitles=True, output_video_path=video_hardsubs_temp.getpath(),
                                                            codec_video=args.codec_video, codec_audio=args.codec_audio)
                                video_hardsubs_temp.save()
                    elif file_type == "subtitle":
                        # Standalone subtitle files (not linked to any media) can be translated and summarized.
                        if args.input_path.is_dir() and path in srt_linked_to_media:
                            print("Subtitle matches a media file; it will be used during media processing. Skipping standalone processing.")
                            continue

                        # Derive base stem and language from filename suffix convention: name_lang.srt
                        base_stem, lang_from_suffix = split_lang_suffix(rel_path.stem)
                        if args.input_lang != "auto":
                            subtitle_language = args.input_lang
                        else:
                            subtitle_language = lang_from_suffix or "auto"

                        softsub_dir = Path(args.output_softsubs, rel_path.parent)
                        subtitle_transcribed_path = Path(softsub_dir, f"{base_stem}_{subtitle_language.lower()}.srt")
                        subtitle_translated_path = Path(softsub_dir, f"{base_stem}_{args.translate.lower()}.srt")

                        transcribed_srt_temp = file_utils.TempFile(subtitle_transcribed_path, file_ext=".srt")
                        if file_utils.file_is_valid(subtitle_transcribed_path) and not args.overwrite:
                            print(f"Existing subtitle file {gray}{subtitle_transcribed_path}{default}. Skipping copy.")
                            transcribed_srt_source_path = subtitle_transcribed_path
                        else:
                            file_utils.copy_file_if_different(path, transcribed_srt_temp.getpath(), silent=True)
                            if not args.disable_srt:
                                transcribed_srt_temp.save()
                            transcribed_srt_source_path = transcribed_srt_temp.getvalidpath()

                        if args.translate == "none":
                            translated_srt_source_path = None
                        elif subtitle_language != "auto" and args.translate == subtitle_language:
                            print("Translation is unnecessary because input and output language are the same. Skipping.")
                            translated_srt_source_path = None
                        elif file_utils.file_is_valid(subtitle_translated_path) and not args.overwrite:
                            print("Translated file found. Skipping translation.")
                            translated_srt_source_path = subtitle_translated_path
                        elif transcribed_srt_source_path:
                            translated_srt_temp = file_utils.TempFile(subtitle_translated_path, file_ext=".srt")
                            print(f"{wblue}Translating{default} with {gray}{args.translate_engine.capitalize()}{default} to {gray}{args.translate}{default}")
                            translate_utils.translate_srt_file(
                                transcribed_srt_source_path,
                                translated_srt_temp.getpath(),
                                args.translate,
                                translate_engine=args.translate_engine,
                                gemini_api_keys=args.gemini_api_keys,
                                overwrite=args.overwrite,
                            )
                            if not args.disable_srt:
                                translated_srt_temp.save()
                            translated_srt_source_path = translated_srt_temp.getvalidpath()
                        else:
                            translated_srt_source_path = None

                        if args.export_txt and transcribed_srt_source_path:
                            export_txt_if_requested(transcribed_srt_source_path, subtitle_transcribed_path.with_suffix(".txt"))
                        if args.export_txt and translated_srt_source_path:
                            export_txt_if_requested(translated_srt_source_path, subtitle_translated_path.with_suffix(".txt"))

                        if args.tltw:
                            summary_source_path = translated_srt_source_path or transcribed_srt_source_path
                            summary_language = "auto-detect" if subtitle_language == "auto" else subtitle_language
                            if summary_source_path and translated_srt_source_path and summary_source_path == translated_srt_source_path and args.translate.lower() != "none":
                                summary_language = args.translate

                            if summary_source_path:
                                summary_output_dir = Path(args.output_tltw, rel_path.parent)
                                summary_filename = f"{base_stem}_tltw_{str(summary_language).lower()}.md"
                                summary_output_path = summary_output_dir / summary_filename

                                if file_utils.file_is_valid(summary_output_path) and not args.overwrite:
                                    print(f"Existing TLTW summary {gray}{summary_output_path}{default}. Skipping.")
                                else:
                                    print(f"{wblue}Generating TLTW summary{default} with {gray}Gemini{default}")
                                    generate_tltw(
                                        GeminiSummaryConfig(
                                            api_keys=args.gemini_api_keys,
                                            subtitle_file=summary_source_path,
                                            output_file=summary_output_path,
                                            language=summary_language,
                                        )
                                    )
                            else:
                                print("No subtitles available for TLTW summary. Skipping.")
                    else:
                        print("not a video file")
                        if args.copy_files:
                            if not args.disable_srt:
                                file_utils.copy_file_if_different(path, Path(
                                    args.output_softsubs, rel_path))
                            if not args.disable_hardsubs:
                                file_utils.copy_file_if_different(path, Path(
                                    args.output_hardsubs, rel_path))
                except Exception as e:  # noqa: BLE001
                    file = path.as_posix()
                    print(f"{red}ERROR !!!{default} {file}")
                    print(f"{yellow}check legen-errors.txt for details{default}")
                    current_time = time.strftime("%y/%m/%d %H:%M:%S", time.localtime())

                    error_message = f"[{current_time}] {file}: {type(e).__name__}: {str(e)}"
                    with open(Path(Path(getframeinfo(currentframe()).filename).resolve().parent, "legen-errors.txt"), "a") as f:
                        f.write(error_message + "\n")
                        f.close()

    print("Deleting temp folder")
    file_utils.delete_folder(
        Path(Path(getframeinfo(currentframe()).filename).resolve().parent, "temp"))

    print(f"{green}Tasks done!{default}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
