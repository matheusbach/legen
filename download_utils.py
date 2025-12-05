import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List


def _resolve_downloader() -> str:
    """Return the executable name for yt-dlp."""
    # First check if yt-dlp is in the same directory as the python executable
    # This handles cases where legen is installed via 'uv tool install'
    local_downloader = Path(sys.executable).parent / "yt-dlp"
    if local_downloader.exists() and os.access(local_downloader, os.X_OK):
        return str(local_downloader)

    downloader = "yt-dlp"
    if shutil.which(downloader):
        return downloader
    raise FileNotFoundError(
        "yt-dlp executable not found. Install yt-dlp before using URL downloads."
    )


def _append_downloaded_suffix_to_subtitles(media_path: Path) -> None:
    """Ensure existing subtitle streams are labeled with a [downloaded] suffix."""
    ffprobe_cmd = [
        "ffprobe",
        "-loglevel",
        "error",
        "-select_streams",
        "s",
        "-show_entries",
        "stream=index:stream_tags=language,title",
        "-of",
        "json",
        "file:" + media_path.expanduser().resolve().as_posix(),
    ]

    try:
        probe = subprocess.run(
            ffprobe_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffprobe failed while inspecting subtitles: {exc}") from exc

    try:
        data = json.loads(probe.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe returned invalid JSON for subtitle metadata") from exc

    streams = data.get("streams") or []
    if not streams:
        return

    metadata_args: List[str] = []
    update_needed = False

    for subtitle_index, stream in enumerate(streams):
        tags = stream.get("tags") or {}
        title = (tags.get("title") or "").strip()
        language = (tags.get("language") or "").strip()

        base_title = title or language or f"Subtitle {subtitle_index + 1}"
        if not base_title:
            base_title = f"Subtitle {subtitle_index + 1}"

        final_title = (
            base_title
            if base_title.endswith(" [downloaded]")
            else f"{base_title} [downloaded]"
        )

        if title != final_title:
            update_needed = True

        metadata_args.extend([
            f"-metadata:s:s:{subtitle_index}",
            f"title={final_title}",
        ])

    if not update_needed:
        return

    with tempfile.NamedTemporaryFile(
        dir=media_path.parent, suffix=media_path.suffix, delete=False
    ) as tmp:
        temp_output = Path(tmp.name)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        "file:" + media_path.as_posix(),
        "-map",
        "0",
        "-c",
        "copy",
    ]

    ffmpeg_cmd.extend(metadata_args)
    ffmpeg_cmd.extend(["-movflags", "+faststart", "file:" + temp_output.as_posix()])

    try:
        subprocess.run(ffmpeg_cmd, check=True)
    except subprocess.CalledProcessError as exc:
        temp_output.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed while tagging subtitles: {exc}") from exc

    os.replace(temp_output, media_path)


_FORMAT_SUFFIX_RE = re.compile(r"\.f\d+(?:[a-z0-9_-]+)?$", re.IGNORECASE)


def _title_from_destination(destination: str) -> str:
    """Derive a human-friendly title from a yt-dlp destination path."""
    sanitized = destination.strip().strip('"').strip("'")
    name = Path(sanitized).name
    base, _ = os.path.splitext(name)
    cleaned = _FORMAT_SUFFIX_RE.sub("", base).strip()
    return cleaned or name


def _subtitle_label_from_path(path_str: str) -> str:
    """Extract a readable subtitle label (language or filename)."""
    sanitized = path_str.strip().strip('"')
    if sanitized.startswith("file:"):
        sanitized = sanitized[5:]
    candidate = Path(sanitized)
    suffixes = candidate.suffixes
    if len(suffixes) >= 2:
        lang_suffix = suffixes[-2]
        if lang_suffix.startswith("."):
            lang = lang_suffix[1:]
        else:
            lang = lang_suffix
        lang = lang.strip()
        if lang:
            return lang
    return candidate.name


def download_urls(
    urls: Iterable[str],
    output_dir: Path,
    overwrite: bool = False,
    download_remote_subs: bool = False,
) -> List[Path]:
    """Download one or more media URLs into output_dir using yt-dlp."""
    downloader = _resolve_downloader()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    format_selector = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio/bestvideo+bestaudio/best"
    command = [
        downloader,
        "--no-warnings",
        "--newline",
        "--progress",
        "--format",
        format_selector,
        "--merge-output-format",
        "mp4",
        "-P",
        str(output_dir),
        "--continue",
    ]
    if download_remote_subs:
        command.extend([
            "--embed-subs",
            "--sub-langs",
            "all",
        ])
    if overwrite:
        command.append("--force-overwrites")
    else:
        command.append("--no-overwrites")

    url_list = [url.strip() for url in urls if url and url.strip()]
    if not url_list:
        raise ValueError("No URLs provided to download")

    command.extend(url_list)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    current_destination: Path | None = None
    current_status: str | None = None
    current_title: str | None = None
    current_url: str | None = None
    current_title_printed = False
    current_stream_index = 0
    active_stream_is_primary = False
    progress_active = False
    last_progress_len = 0
    error_lines: List[str] = []

    def current_label(fallback: str | None = None) -> str:
        if current_title:
            return current_title
        if current_url:
            return current_url
        if current_destination is not None:
            return current_destination.name
        return fallback or "download"

    def finish_progress() -> None:
        nonlocal progress_active, last_progress_len
        if progress_active:
            sys.stdout.write("\r")
            sys.stdout.write(" " * last_progress_len)
            sys.stdout.write("\r")
            sys.stdout.flush()
            last_progress_len = 0
            progress_active = False

    def update_progress(text: str) -> None:
        nonlocal progress_active, last_progress_len
        try:
            terminal_width = os.get_terminal_size().columns
        except OSError:
            terminal_width = 120

        if terminal_width > 0 and len(text) > terminal_width:
            chunks = [text[i:i + terminal_width] for i in range(0, len(text), terminal_width)]
            finish_progress()
            for chunk in chunks[:-1]:
                print(chunk, flush=True)
            text = chunks[-1]

        display_text = text
        if len(display_text) < last_progress_len:
            display_text += " " * (last_progress_len - len(display_text))

        sys.stdout.write("\r" + display_text)
        sys.stdout.flush()
        last_progress_len = len(display_text)
        progress_active = True

    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            if raw_line is None:
                continue
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue

            if "Extracting URL:" in stripped:
                url = stripped.split("Extracting URL:", 1)[1].strip()
                if url:
                    current_url = url
                    current_title = None
                    current_destination = None
                    current_status = None
                    current_title_printed = False
                    current_stream_index = 0
                    active_stream_is_primary = False
                    update_progress(f"Loading [{url}]")
                continue

            destination_prefix = "[download] Destination:"
            resume_prefix = "[download] Resuming download"
            already_downloaded_text = "has already been downloaded"

            if stripped.startswith(destination_prefix):
                destination_value = stripped[len(destination_prefix):].strip()
                target = Path(destination_value)
                if not target.is_absolute():
                    target = (output_dir / target).resolve()
                current_destination = target
                candidate_title = _title_from_destination(destination_value)
                if current_title != candidate_title:
                    current_title = candidate_title
                    current_title_printed = False
                    current_stream_index = 0

                is_primary_stream = current_stream_index == 0

                finish_progress()
                if is_primary_stream:
                    display_name = current_label(target.name)
                    if target.exists() and not overwrite:
                        print(f"Existing: {display_name}", flush=True)
                        current_status = "skip"
                    else:
                        print(f"Downloading: {display_name}", flush=True)
                        current_status = "download"
                    current_title_printed = True
                else:
                    if current_status is None:
                        current_status = "download"

                current_stream_index += 1
                active_stream_is_primary = is_primary_stream
                continue

            if stripped.startswith(resume_prefix):
                if active_stream_is_primary and current_status != "continue":
                    finish_progress()
                    display_name = current_label()
                    print(f"Downloading: {display_name}", flush=True)
                    current_status = "continue"
                continue

            if already_downloaded_text in stripped:
                remainder = stripped.split("[download]", 1)[-1].strip()
                name_part = remainder.split(already_downloaded_text, 1)[0].strip().rstrip('.')
                if not name_part and current_destination is not None:
                    name_part = current_destination.name
                if name_part:
                    title_guess = _title_from_destination(name_part)
                    if title_guess:
                        current_title = title_guess
                finish_progress()
                display_name = current_label(name_part or None)
                if not current_title_printed:
                    print(f"Existing: {display_name}", flush=True)
                    current_title_printed = True
                current_status = "skip"
                active_stream_is_primary = False
                continue

            if download_remote_subs:
                if "Writing video subtitles to:" in stripped or "Writing subtitles to:" in stripped:
                    _, target_part = stripped.split("to:", 1)
                    subtitle_target = target_part.strip()
                    finish_progress()
                    label = _subtitle_label_from_path(subtitle_target)
                    print(f"Downloading subtitle track: {label}", flush=True)
                    continue
                if stripped.lower().startswith("[download] downloading subtitle"):
                    finish_progress()
                    print(stripped, flush=True)
                    continue

            if stripped.startswith("[download]") and "Downloading webpage" in stripped:
                finish_progress()
                current_title = None
                current_title_printed = False
                current_stream_index = 0
                active_stream_is_primary = False
                current_status = None
                continue

            if stripped.startswith("[Merger] Merging formats into"):
                finish_progress()
                current_title = None
                current_title_printed = False
                current_stream_index = 0
                active_stream_is_primary = False
                current_status = None
                continue

            if stripped.startswith("[download]"):
                if "%" in stripped:
                    update_progress(stripped)
                    if "100%" in stripped:
                        finish_progress()
                continue

            if "ERROR" in stripped.upper() or stripped.lower().startswith("error:") or stripped.startswith("[error]"):
                finish_progress()
                print(stripped, flush=True)
                error_lines.append(stripped)

        return_code = process.wait()
        finish_progress()
    except Exception:
        process.kill()
        finish_progress()
        raise

    if return_code != 0:
        if not error_lines:
            print(f"{downloader} reported an error. Review the output above for details.", flush=True)
        raise RuntimeError(
            f"{downloader} failed with exit code {return_code}. Check the terminal output for details."
        )

    downloaded_files = sorted(output_dir.rglob("*.mp4"))
    if not downloaded_files:
        raise RuntimeError(
            f"{downloader} completed without producing MP4 files. Check the URL or provide --overwrite to re-download."
        )

    if download_remote_subs:
        for media_path in downloaded_files:
            try:
                _append_downloaded_suffix_to_subtitles(media_path)
            except RuntimeError as exc:
                print(f"Warning: unable to tag downloaded subtitles in {media_path.name}: {exc}")

    return downloaded_files
