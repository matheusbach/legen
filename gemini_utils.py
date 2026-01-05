from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
import threading
import time
from typing import Callable, Iterable, List, Sequence

import pysrt

import gemini_srt_translator as gst


@dataclass(frozen=True)
class GeminiTranslationConfig:
    api_keys: Sequence[str]
    input_file: Path
    output_file: Path
    target_language: str
    batch_size: int = 500
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 50
    free_quota: bool = True
    resume: bool = False
    thinking: bool = True
    progress_log: bool = False
    thoughts_log: bool = False


class MultiKeyGeminiTranslator(gst.GeminiSRTTranslator):
    """Gemini translator that rotates across an arbitrary number of API keys."""

    def __init__(self, api_keys: Sequence[str], **kwargs) -> None:
        cleaned: List[str] = [key.strip() for key in api_keys if key and key.strip()]
        if not cleaned:
            raise ValueError("At least one Gemini API key is required.")

        primary = cleaned[0]
        secondary = cleaned[1] if len(cleaned) > 1 else None

        super().__init__(gemini_api_key=primary, gemini_api_key2=secondary, **kwargs)

        self._api_keys = cleaned
        self._api_index = 0
        self.current_api_key = primary
        self.current_api_number = 1
        self.backup_api_number = 2 if len(cleaned) > 1 else 1

    def _switch_api(self) -> bool:  # type: ignore[override]
        if len(self._api_keys) <= 1:
            return False

        previous_number = self.current_api_number
        total_keys = len(self._api_keys)

        for step in range(1, total_keys + 1):
            next_index = (self._api_index + step) % total_keys
            if next_index == self._api_index:
                continue

            next_key = self._api_keys[next_index]
            if next_key:
                self._api_index = next_index
                self.current_api_key = next_key
                self.current_api_number = next_index + 1
                self.backup_api_number = previous_number
                return True

        return False


def translate_with_gemini(config: GeminiTranslationConfig) -> pysrt.SubRipFile:
    additional_instructions = (
        "CRITICAL INSTRUCTIONS:\n"
        "1. You MUST return exactly the same number of objects as the input batch.\n"
        "2. Check the input segments count and ensure your output count matches exactly.\n"
        "3. Do not skip any index. Every input object must have a corresponding output object.\n"
        "4. If a line is empty in input, keep it empty in output.\n"
        "5. If a line has content, it MUST be translated. Do not return empty strings for non-empty input.\n"
        "6. Do not merge or split subtitles.\n"
    )

    translator = MultiKeyGeminiTranslator(
        api_keys=config.api_keys,
        target_language=config.target_language,
        input_file=str(config.input_file),
        output_file=str(config.output_file),
        batch_size=config.batch_size,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        free_quota=config.free_quota,
        resume=config.resume,
        thinking=config.thinking,
        progress_log=config.progress_log,
        thoughts_log=config.thoughts_log,
        description=additional_instructions,
    )

    translator.translate()

    return pysrt.open(config.output_file, encoding="utf-8")


def normalize_api_keys(keys: Iterable[str] | str | None) -> List[str]:
    if keys is None:
        return []

    if isinstance(keys, str):
        raw = [keys]
    else:
        raw = list(keys)

    candidates: List[str] = []
    for value in raw:
        if not value:
            continue
        parts = [part.strip() for part in str(value).replace("\n", ",").split(",")]
        candidates.extend(part for part in parts if part)

    unique: List[str] = []
    seen = set()
    for key in candidates:
        if key not in seen:
            unique.append(key)
            seen.add(key)

    return unique


@dataclass(frozen=True)
class GeminiSummaryConfig:
    api_keys: Sequence[str]
    subtitle_file: Path
    output_file: Path
    language: str
    model: str = "gemini-2.5-flash"
    # TLTW tends to be more useful with longer outputs. If the model truncates,
    # we auto-continue (see _send_tltw_request).
    max_output_tokens: int = 16364
    final_max_output_tokens: int | None = None
    request_timeout: int = 500
    truncate_chars: int | None = None
    # With modern Gemini context windows, chunking is often unnecessary and can
    # reduce coherence. Keep it disabled by default; enable explicitly for very
    # large inputs or when you want extra robustness.
    chunk_chars: int | None = None

    # Inference tuning (lower temperature => more stable outputs)
    temperature: float = 0.15
    top_p: float = 0.9
    top_k: int = 40

    # Auto-continue when the model hits output limits
    max_rounds: int = 10
    continuation_tail_chars: int = 800

    # CLI/console progress
    show_progress: bool = True
    progress_update_interval: float = 0.5
    progress_preview_chars: int = 80
    stream_output: bool = True


def _mask_api_key(key: str) -> str:
    cleaned = (key or "").strip()
    if len(cleaned) <= 6:
        return "***"
    return "***" + cleaned[-4:]


def _strip_ansi(text: str) -> str:
    # Minimal ANSI stripper to keep progress lines tidy in some terminals.
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\x1b":
            # Skip CSI sequences.
            while i < len(text) and text[i] != "m":
                i += 1
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _single_line_preview(text: str, limit: int) -> str:
    cleaned = " ".join((text or "").split())
    if limit <= 0:
        return ""
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)] + "…"


class _ProgressTicker:
    def __init__(
        self,
        *,
        label: str,
        expected_seconds: float,
        update_interval: float,
        preview_supplier: Callable[[], str],
    ) -> None:
        self._label = label
        self._expected_seconds = max(1.0, float(expected_seconds))
        self._update_interval = max(0.05, float(update_interval))
        self._preview_supplier = preview_supplier
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start = 0.0

    def start(self) -> None:
        if not sys.stderr.isatty():
            return
        self._start = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not sys.stderr.isatty():
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        # tqdm is already a dependency of this project (requirements.txt).
        from tqdm import tqdm

        # We intentionally display ONLY the percentage (no bar), plus an optional postfix.
        pbar = tqdm(
            total=100,
            desc=self._label,
            file=sys.stderr,
            leave=False,
            dynamic_ncols=True,
            bar_format="{desc} {percentage:3.0f}% | {postfix}",
        )

        try:
            last_n = 0
            while not self._stop.is_set():
                elapsed = max(0.0, time.time() - self._start)
                pct = min(0.99, elapsed / self._expected_seconds)
                target_n = int(pct * 100)
                if target_n > last_n:
                    pbar.update(target_n - last_n)
                    last_n = target_n

                preview = self._preview_supplier() if self._preview_supplier else ""
                preview = _strip_ansi(preview)
                preview = _single_line_preview(preview, 80)
                pbar.set_postfix_str(preview)

                pbar.refresh()
                time.sleep(self._update_interval)

            # Finish to 100% on stop.
            if last_n < 100:
                pbar.update(100 - last_n)
                pbar.refresh()
        finally:
            pbar.close()


def _load_srt_as_text(subtitle_file: Path, truncate_chars: int | None) -> str:
    """Read the subtitle file as raw text, optionally truncating for safety."""

    content = subtitle_file.read_text(encoding="utf-8", errors="replace").strip()
    if truncate_chars and truncate_chars > 0 and len(content) > truncate_chars:
        return content[:truncate_chars]
    return content


def _build_tltw_prompt(language: str) -> str:
    # Backward-compatible default prompt (duration-aware prompt is built in generate_tltw_summary).
    return _build_tltw_prompt_with_limits(language=language, max_key_points=12)


def _build_tltw_prompt_with_limits(*, language: str, max_key_points: int, min_key_points: int | None = None) -> str:
    if max_key_points < 1:
        max_key_points = 1
    if min_key_points is None:
        min_key_points = max(1, int(math.floor(max_key_points * 0.6)))
    else:
        min_key_points = max(1, min(int(min_key_points), int(max_key_points)))

    return (
        "Generate a high-quality 'TLTW' (Too Long To Watch) summary in "
        f"{language} strictly based on the provided transcript or subtitles. "
        "The TLTW must be factual, concise, and faithful to the source.\n\n"

        "Use Markdown and follow this exact structure:\n\n"

        "# Title\n"
        "- A clear, descriptive title reflecting the main subject of the content\n\n"

        "*Tags: [tag 1, tag 2, ...]*\n"
        "- 3–9 short tags, comma-separated, singular words when possible, multi-word-expressions only when significant, lowercase when possible (e.g., `ai, subtitles, whisperx, translation`)\n"
        "- Tags must reflect topics actually present in the source\n\n"

        "## Key Points\n"
        f"- Write at most {max_key_points} bullet points\n"
        f"- Prefer {min_key_points}–{max_key_points} bullet points; you may go up to 40% smaller to avoid redundancy\n"
        "- Put the most important points first\n"
        "- Preserve technical terminology used in the source when relevant\n"
        "- Each bullet must briefly explain the point (topic + a short clarifying clause), not just label it\n"
        "- Avoid generic takeaways like 'the importance of', 'the need of', 'is crucial', unless tied to a specific situation described\n"
        "- Avoid advice/recommendations here; put them only in Actions when explicitly present\n"
        "- Do not end bullet lines with a period (no trailing '.')\n\n"

        "## Actions or Next Steps (only if applicable)\n"
        "- Bullet points describing explicit recommendations, procedures, or follow-ups mentioned in the source\n"
        "- Omit this section entirely if no actionable items are present\n"
        "- Do not end bullet lines with a period (no trailing '.')\n\n"

        "## Summary\n"
        f"- Write at most {max_key_points} chapter titles\n"
        f"- Prefer {min_key_points}–{max_key_points} chapter titles; you may go up to 40% smaller to avoid redundancy\n"
        "- Each title must be 1 short sentence\n"
        "- Write chapter-like titles, not takeaways: describe the topic, not what the viewer should learn/do\n"
        "- Prefer noun phrases and topic labels; avoid advice, conclusions, recommendations, or moral-of-the-story phrasing\n"
        "- Suppress authorship; write each line like a book chapter title, not about who said/did it\n"
        "- Each title must be a single line in the format: HH:MM:SS description\n"
        "- Use HH:MM:SS (no milliseconds)\n"
        "- Keep items in chronological order\n"
        "- Use timestamps aligned to the subtitle timeline (best possible approximation based on nearby lines)\n"
        "- Do not use bullet markers for these lines\n"
        "- Do not end lines with a period (no trailing '.')\n\n"
        "Example format:\n"
        "```\n"
        "00:00:02 chapter title description\n"
        "00:22:13 chapter title description\n"
        "```\n\n"

        "Writing style requirements:\n"
        "- Use active voice; avoid passive constructions\n"
        "- Be concise; remove filler words\n"
        "- Use an assertive, direct tone\n"
        "- Start lines with an article only when it naturally fits; prefer direct noun-phrase titles and topic labels\n\n"

        "Rules:\n"
        f"- Write exclusively in {language}\n"
        "- Do not invent, extrapolate, or assume information\n"
        "- Avoid redundancy and meta commentary\n"
        "- Keep sentences short, direct, and information-dense\n"
        "- Dont cause repetition of structures\n"
        "- Do not apologize or reference missing context\n"
        "- Do not mention the transcript or the act of summarization\n"
        "- End the document with a final line exactly equal to: <!-- END -->\n"
    )


def _estimate_srt_duration_seconds(subtitle_file: Path) -> float:
    """Best-effort estimate of the subtitle duration in seconds."""

    try:
        subs = pysrt.open(subtitle_file, encoding="utf-8")
    except Exception:
        return 0.0
    if not subs:
        return 0.0
    try:
        last_end_ms = subs[-1].end.ordinal
        return max(0.0, float(last_end_ms) / 1000.0)
    except Exception:
        return 0.0



def _build_chunk_prompt(language: str, chunk_index: int, chunk_total: int) -> str:
    return (
        f"You are summarizing chunk {chunk_index}/{chunk_total} of a transcript. "
        f"Write in {language}, strictly based on this chunk. "
        "Return Markdown with exactly one section: '## Key Points' followed by bullet points. "
        "Do not include a title or any other sections. "
        "Each bullet must include a concrete anchor from this chunk and a brief explanation (topic + short clarifying clause). "
        "Avoid generic takeaways and avoid advice unless explicitly present. "
        "Use active voice, be concise, and suppress authorship (chapter-title style). "
        "Do not end bullet lines with a period (no trailing '.')."
    )


def _send_tltw_request(
    *,
    api_key: str,
    subtitle_text: str,
    language: str,
    model: str,
    max_output_tokens: int,
    request_timeout: int,
    prompt_builder: Callable[[str], str] = _build_tltw_prompt,
    temperature: float = 0.2,
    top_p: float = 0.9,
    top_k: int = 40,
    max_rounds: int = 3,
    continuation_tail_chars: int = 800,
    show_progress: bool = False,
    progress_update_interval: float = 0.2,
    progress_preview_chars: int = 140,
    stream_output: bool = False,
) -> str:
    try:
        import google.generativeai as genai
    except Exception as exc:  # pragma: no cover - import depends on optional dependency
        raise RuntimeError("google-generativeai is required for TLTW summaries.") from exc

    genai.configure(api_key=api_key)

    END_MARKER = "<!-- END -->"
    base_prompt = prompt_builder(language)
    require_end_marker = END_MARKER in base_prompt
    model_client = genai.GenerativeModel(model)

    def _finish_reason_is_truncation(resp) -> bool:
        try:
            candidates = getattr(resp, "candidates", None) or []
            if not candidates:
                return False
            finish_reason = getattr(candidates[0], "finish_reason", None)
            if finish_reason is None:
                return False
            finish_str = str(finish_reason).lower()
            return "max" in finish_str or "token" in finish_str
        except Exception:
            return False

    last_preview: str = ""
    progress_lines_count: int = 0

    def _expected_seconds_for_call(*, include_subtitle_text: bool) -> float:
        # Heuristic only. Goal: a progress bar that feels realistic.
        base = 2.0
        input_factor = (len(subtitle_text) / 50_000.0) * (3.0 if include_subtitle_text else 0.8)
        output_factor = (max_output_tokens / 2000.0) * 1.8
        return base + input_factor + output_factor

    def _generate(
        prompt: str,
        *,
        include_subtitle_text: bool,
        phase_label: str,
        base_lines: int,
    ) -> tuple[str, bool]:
        nonlocal last_preview
        nonlocal progress_lines_count

        parts = [prompt]
        if include_subtitle_text:
            parts.append(subtitle_text)

        tqdm_active = bool(show_progress and sys.stderr.isatty())
        ticker: _ProgressTicker | None = None
        progress_lines_count = max(0, int(base_lines))

        # Prefer streaming text preview when available; otherwise show line counts.
        use_text_preview = bool(stream_output and tqdm_active)
        emit_stream_to_stderr = bool(stream_output and show_progress and not tqdm_active)

        if show_progress:
            expected = _expected_seconds_for_call(include_subtitle_text=include_subtitle_text)
            ticker = _ProgressTicker(
                label=phase_label,
                expected_seconds=expected,
                update_interval=progress_update_interval,
                preview_supplier=lambda: (
                    last_preview if use_text_preview else f"already generated {progress_lines_count} lines"
                ),
            )
            if tqdm_active:
                ticker.start()
            else:
                sys.stderr.write(f"{phase_label}...\n")
                sys.stderr.flush()

        response = None
        text = ""

        try:
            gen_kwargs = dict(
                generation_config={
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "top_k": int(top_k),
                    "max_output_tokens": int(max_output_tokens),
                },
                request_options={"timeout": request_timeout},
            )

            if stream_output:
                try:
                    response_iter = model_client.generate_content(parts, stream=True, **gen_kwargs)
                    acc = ""
                    emitted_len = 0
                    last_resp = None
                    for resp in response_iter:
                        last_resp = resp
                        chunk_text = (getattr(resp, "text", "") or "")
                        if not chunk_text:
                            continue
                        if chunk_text.startswith(acc):
                            acc = chunk_text
                        else:
                            acc += chunk_text

                        if emit_stream_to_stderr and len(acc) > emitted_len:
                            sys.stderr.write(acc[emitted_len:])
                            sys.stderr.flush()
                            emitted_len = len(acc)

                        # Update progress as text streams in.
                        progress_lines_count = base_lines + (acc.count("\n") + (1 if acc else 0))
                        last_preview = _single_line_preview(
                            acc[-max(0, progress_preview_chars * 3) :],
                            progress_preview_chars,
                        )

                    response = last_resp
                    text = acc.strip()

                    if emit_stream_to_stderr and acc and not acc.endswith("\n"):
                        sys.stderr.write("\n")
                        sys.stderr.flush()
                except Exception:
                    # Some environments/models may not support streaming.
                    use_text_preview = False
                    response = model_client.generate_content(parts, **gen_kwargs)
                    text = (getattr(response, "text", "") or "").strip()
            else:
                use_text_preview = False
                response = model_client.generate_content(parts, **gen_kwargs)
                text = (getattr(response, "text", "") or "").strip()
        except Exception as exc:  # noqa: BLE001
            if ticker is not None:
                ticker.stop()
            if show_progress and not tqdm_active:
                sys.stderr.write(f"{phase_label} failed: {exc}\n")
                sys.stderr.flush()
            raise
        finally:
            if ticker is not None:
                ticker.stop()

        if not text:
            if show_progress and not tqdm_active:
                sys.stderr.write(f"{phase_label} failed: empty response\n")
                sys.stderr.flush()
            raise RuntimeError("Empty response from Gemini while generating TLTW summary.")

        progress_lines_count = base_lines + (text.count("\n") + 1)
        last_preview = _single_line_preview(text[-max(0, progress_preview_chars * 2) :], progress_preview_chars)

        truncated = _finish_reason_is_truncation(response)
        if require_end_marker and END_MARKER not in text:
            truncated = True

        if show_progress and not tqdm_active:
            sys.stderr.write(f"{phase_label} done\n")
            sys.stderr.flush()

        return text, truncated

    if show_progress and not sys.stderr.isatty():
        sys.stderr.write(
            f"TLTW request: model={model} key={_mask_api_key(api_key)} max_output_tokens={max_output_tokens} timeout={request_timeout}s\n"
        )
        sys.stderr.flush()

    step = 1
    full_text, truncated = _generate(
        base_prompt,
        include_subtitle_text=True,
        phase_label=f"Gemini thinking (step {step})",
        base_lines=0,
    )
    if require_end_marker and END_MARKER in full_text:
        return full_text

    # If the model hit output limits, ask it to continue the same document.
    rounds = 1
    while truncated and rounds < max_rounds:
        tail = full_text[-max(0, int(continuation_tail_chars)) :]
        continuation_prompt = (
            base_prompt
            + "\n\nYou already started writing the Markdown document. "
            + "However, your previous output was cut off due to length limits. "
            + "Remember all the original instructions. "
            + "Continue it from exactly where it stopped. Do NOT repeat content. "
            + "Maintain the same structure and formatting. "
            + ("Finish by writing the final line exactly equal to: " + END_MARKER + "\n" if require_end_marker else "")
            + "Here is the last part you wrote (for alignment):\n\n"
            + "```\n"
            + tail
            + "\n```\n\n"
            + "Continue now:\n"
        )
        # Continuations include the subtitle text to keep the model grounded and
        # prevent the second half from drifting into generic filler.
        step += 1
        base_lines = full_text.count("\n") + (1 if full_text else 0)
        next_text, truncated = _generate(
            continuation_prompt,
            include_subtitle_text=True,
            phase_label=f"Gemini thinking (step {step})",
            base_lines=base_lines,
        )

        # Best-effort de-duplication: if the continuation overlaps with the tail, trim it.
        overlap_trimmed = next_text
        if tail and next_text:
            probe = tail[-200:]
            pos = next_text.find(probe)
            if pos != -1:
                overlap_trimmed = next_text[pos + len(probe) :].lstrip()

        if overlap_trimmed:
            # Do not force a newline between chunks; it can break mid-line continuity.
            left = full_text
            right = overlap_trimmed.strip()
            separator = ""
            if left and right and left[-1].isalnum() and right[0].isalnum():
                separator = " "
            full_text = (left.rstrip() + separator + right).strip()
        else:
            # If we couldn't make progress, stop to avoid looping.
            break

        if require_end_marker and END_MARKER in full_text:
            return full_text

        rounds += 1

    if require_end_marker and END_MARKER not in full_text:
        raise RuntimeError(
            "Gemini TLTW output was not finalized (missing END marker). "
            "Try increasing max_rounds and/or max_output_tokens."
        )

    return full_text


def _strip_end_marker(text: str) -> str:
    """Remove the internal END marker from the final output."""

    marker = "<!-- END -->"
    if marker not in text:
        return text

    lines = text.splitlines()
    cleaned = [line for line in lines if line.strip() != marker]
    return "\n".join(cleaned).strip()


def generate_tltw_summary(
    config: GeminiSummaryConfig,
    *,
    request_func: Callable[..., str] = _send_tltw_request,
) -> str:
    """
    Generate a structured TLTW summary from an SRT file using Gemini.

    Returns the generated summary text after writing it to ``config.output_file``.
    """

    api_keys = normalize_api_keys(config.api_keys)
    if not api_keys:
        raise ValueError("Gemini API key is required for TLTW summaries. Provide --gemini_api_key.")

    if not config.subtitle_file.exists():
        raise FileNotFoundError(f"Subtitle file not found: {config.subtitle_file}")

    subtitle_text = _load_srt_as_text(config.subtitle_file, config.truncate_chars)
    duration_seconds = _estimate_srt_duration_seconds(config.subtitle_file)
    # 10 items per 60 minutes, scaled by fractional hours (round up).
    # Keep a floor of 10 for short videos to avoid undercoverage.
    # Allow down to 40% smaller via min_key_points.
    if duration_seconds <= 0:
        max_key_points = 10
    else:
        hours = float(duration_seconds) / 3600.0
        max_key_points = int(math.ceil(hours * 10.0))
        max_key_points = max(10, max_key_points)
    min_key_points = max(1, int(math.floor(max_key_points * 0.6)))

    final_prompt_builder = lambda lang: _build_tltw_prompt_with_limits(
        language=lang,
        max_key_points=max_key_points,
        min_key_points=min_key_points,
    )

    def _split_into_chunks(text: str, limit: int | None) -> list[str]:
        if limit is None or limit <= 0 or len(text) <= limit:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + limit)
            chunks.append(text[start:end])
            start = end
        return chunks

    chunks = _split_into_chunks(subtitle_text, config.chunk_chars)

    def _run_request(payload_text: str, prompt_builder: Callable[[str], str]) -> str:
        last_error: Exception | None = None
        for key in api_keys:
            try:
                try:
                    return request_func(
                        api_key=key,
                        subtitle_text=payload_text,
                        language=config.language,
                        model=config.model,
                        max_output_tokens=config.max_output_tokens,
                        request_timeout=config.request_timeout,
                        prompt_builder=prompt_builder,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        top_k=config.top_k,
                        max_rounds=config.max_rounds,
                        continuation_tail_chars=config.continuation_tail_chars,
                        show_progress=config.show_progress,
                        progress_update_interval=config.progress_update_interval,
                        progress_preview_chars=config.progress_preview_chars,
                        stream_output=config.stream_output,
                    )
                except TypeError:
                    # Backward-compat for custom request_func implementations.
                    return request_func(
                        api_key=key,
                        subtitle_text=payload_text,
                        language=config.language,
                        model=config.model,
                        max_output_tokens=config.max_output_tokens,
                        request_timeout=config.request_timeout,
                        prompt_builder=prompt_builder,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        top_k=config.top_k,
                        max_rounds=config.max_rounds,
                        continuation_tail_chars=config.continuation_tail_chars,
                    )
            except Exception as exc:  # noqa: BLE001
                if config.show_progress:
                    sys.stderr.write(
                        f"Request failed with key={_mask_api_key(key)}: {exc}\n"
                    )
                    sys.stderr.flush()
                last_error = exc
                continue
        raise RuntimeError("Gemini TLTW summary failed with all provided API keys.") from last_error

    if len(chunks) == 1:
        summary = _run_request(chunks[0], final_prompt_builder)
        summary = _strip_end_marker(summary)
        config.output_file.parent.mkdir(parents=True, exist_ok=True)
        config.output_file.write_text(summary, encoding="utf-8")
        return summary

    # Multi-chunk: summarize each chunk, then synthesize
    chunk_summaries: list[str] = []
    total = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        chunk_prompt = lambda lang, i=idx, t=total: _build_chunk_prompt(lang, i, t)
        summary_chunk = _run_request(chunk, chunk_prompt)
        chunk_summaries.append(f"### Chunk {idx}/{total}\n{summary_chunk}")

    final_text = "\n\n".join(chunk_summaries)

    final_tokens = config.final_max_output_tokens or config.max_output_tokens

    # Final synthesis reuses request_func but with aggregated summaries
    last_error: Exception | None = None
    for key in api_keys:
        try:
            try:
                final_summary = request_func(
                    api_key=key,
                    subtitle_text=f"Summaries of all chunks follow:\n\n{final_text}",
                    language=config.language,
                    model=config.model,
                    max_output_tokens=final_tokens,
                    request_timeout=config.request_timeout,
                    prompt_builder=final_prompt_builder,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    max_rounds=config.max_rounds,
                    continuation_tail_chars=config.continuation_tail_chars,
                    show_progress=config.show_progress,
                    progress_update_interval=config.progress_update_interval,
                    progress_preview_chars=config.progress_preview_chars,
                    stream_output=config.stream_output,
                )
            except TypeError:
                final_summary = request_func(
                    api_key=key,
                    subtitle_text=f"Summaries of all chunks follow:\n\n{final_text}",
                    language=config.language,
                    model=config.model,
                    max_output_tokens=final_tokens,
                    request_timeout=config.request_timeout,
                    prompt_builder=final_prompt_builder,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    max_rounds=config.max_rounds,
                    continuation_tail_chars=config.continuation_tail_chars,
                )
            final_summary = _strip_end_marker(final_summary)
            config.output_file.parent.mkdir(parents=True, exist_ok=True)
            config.output_file.write_text(final_summary, encoding="utf-8")
            return final_summary
        except Exception as exc:  # noqa: BLE001
            if config.show_progress:
                sys.stderr.write(
                    f"Final synthesis failed with key={_mask_api_key(key)}: {exc}\n"
                )
                sys.stderr.flush()
            last_error = exc
            continue

    raise RuntimeError("Gemini TLTW summary failed during final synthesis with all provided API keys.") from last_error


def generate_tltw(
    config: GeminiSummaryConfig,
    *,
    request_func: Callable[..., str] = _send_tltw_request,
) -> str:
    """Backward/short alias for generating a TLTW summary."""

    return generate_tltw_summary(config, request_func=request_func)
