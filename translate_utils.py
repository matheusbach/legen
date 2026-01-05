from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Sequence

import deep_translator
import pysrt
import tqdm.asyncio
import subtitle_utils
from utils import format_time
from gemini_utils import (
    GeminiTranslationConfig,
    normalize_api_keys,
    translate_with_gemini,
)

# all entence endings for japanese and normal people languages
sentence_endings = ['.', '!', '?', ')', 'よ', 'ね',
                    'の', 'さ', 'ぞ', 'な', 'か', '！', '。', '」', '…']

# a good separator is a char or string that doenst change the translation quality but is near ever preserved in result at same or near position
separator = " ◌ "
separator_unjoin = separator.replace(' ', '')
chunk_max_chars = 4999


def translate_srt_file(
    srt_file_path: Path,
    translated_subtitle_path: Path,
    target_lang,
    translate_engine: str = "google",
    gemini_api_keys=None,
    overwrite: bool = False,
):
    """
    Translate SRT file using the specified engine.
    translate_engine: "google" or "gemini"
    gemini_api_keys: optional sequence of API keys required if translate_engine == "gemini"
    """
    # Load the original SRT file
    subs = pysrt.open(srt_file_path, encoding='utf-8')

    # Extract the subtitle content and store it in a list. Also rejoin all lines splited
    sub_content = [' '.join(sub.text.strip().splitlines()) for sub in subs]

    if translate_engine == "gemini":
        api_keys = normalize_api_keys(gemini_api_keys)
        if not api_keys:
            raise ValueError("Gemini API key is required for Gemini translation. Get one at https://aistudio.google.com/apikey")

        # Force cleanup of previous runs to avoid resume/progress issues
        Path(translated_subtitle_path).unlink(missing_ok=True)
        Path(str(translated_subtitle_path) + ".progress").unlink(missing_ok=True)

        subs = translate_with_gemini(
            GeminiTranslationConfig(
                api_keys=api_keys,
                input_file=srt_file_path,
                output_file=translated_subtitle_path,
                target_language=target_lang,
                resume=False
            )
        )

        return subs

    # Default: Google Translate
    # Make chunks of at maximum $chunk_max_chars to stay under Google Translate public API limits
    chunks = join_sentences(sub_content, chunk_max_chars) or []

    # Empty list to store enumerated translated chunks
    translated_chunks = [None] * len(chunks)

    tasks = []
    # Limit to 7 concomitant running tasks
    semaphore = asyncio.Semaphore(7)

    # Async chunks translate function
    async def translate_async():
        async def run_translate(index, chunk, lang):
            while True:
                try:
                    async with semaphore:
                        result = await asyncio.wait_for(translate_chunk(index, chunk, lang), 120)
                    translated_chunks[index] = result
                    break
                except Exception:
                    # Restart task
                    await asyncio.sleep(3)

        for index, chunk in enumerate(chunks):
            task = asyncio.create_task(
                run_translate(index, chunk, target_lang))
            tasks.append(task)

        for tsk in tqdm.asyncio.tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Translating", unit="chunks", unit_scale=False, leave=True, bar_format="{desc} {percentage:3.0f}% | {n_fmt}/{total_fmt} | ETA: {remaining} | ⏱: {elapsed}"):
            await tsk

    # Cria um loop de eventos e executa as tasks
    loop = asyncio.get_event_loop()
    loop.run_until_complete(translate_async())

    print('Processing translation...', end='')

    # Unjoin lines within each chunk that end with a sentence ending
    unjoined_texts = [unjoin_sentences(
        chunk, translated_chunks[i], separator_unjoin) or "" for i, chunk in enumerate(chunks)]
    unjoined_texts = [text for sublist in unjoined_texts for text in sublist]

    # Split lines as necessary targeting same number of lines as original string
    for i, segment in enumerate(unjoined_texts):
        unjoined_texts[i] = "\n".join(subtitle_utils.split_string_to_max_lines(
            text=segment, max_width=0, max_lines=len(subs[i].text.splitlines())))

    # Combine the original and translated subtitle content
    for i, sub in enumerate(subs):
        sub.text = unjoined_texts[i]

    # Save the translated SRT file
    os.makedirs(translated_subtitle_path.parent, exist_ok=True)
    subs.save(translated_subtitle_path, encoding='utf-8')

    print('\r                         ', end='\r')

    return subs

# Async chunk translate function


async def translate_chunk(index, chunk, target_lang):
    while True:
        try:
            # Translate the subtitle content of the chunk using Google Translate
            translator = deep_translator.google.GoogleTranslator(
                source='auto', target=target_lang)
            translated_chunk: str = await asyncio.wait_for(asyncio.get_event_loop().run_in_executor(None, translator.translate, chunk), 30)
            await asyncio.sleep(0)

            # if nothing is retuned, return the original chunk
            if translated_chunk is None or len(translated_chunk.replace(separator.strip(), '').split()) == 0:
                return chunk

            return translated_chunk
        except Exception as e:
            # If an error occurred, retry
            del translator
            print(
                f"\r[chunk {index}]: Exception: {e.__doc__} Retrying in 30 seconds...", flush=True)
            await asyncio.sleep(30)


def join_sentences(lines, max_chars):
    """Join sentences in chunks that stay under *max_chars* without breaking the separator mapping."""
    joined_lines = []
    current_chunk = ""

    for index, line in enumerate(lines):
        if not line:
            line = '\u3164'  # invisible char (not a simple space)

        addition = line + separator

        # if adding the current line would overflow, flush the chunk first
        if current_chunk and len(current_chunk) + len(line) + len(separator) > max_chars:
            joined_lines.append(current_chunk)
            current_chunk = ""

        if len(addition) > max_chars:
            # a single line exceeds the limit; truncate conservatively
            end_index = line.rfind(' ', 0, max_chars - (1 + len(separator)))
            if end_index == -(1 + len(separator)):
                end_index = max_chars - (1 + len(separator))
            joined_lines.append((line[:end_index] + '\u2026' + separator)[:max_chars])
            continue

        current_chunk += addition

        is_last_line = index == len(lines) - 1
        ends_sentence = any(line.endswith(ending) for ending in sentence_endings)

        if not ends_sentence and not is_last_line:
            continue

        if is_last_line:
            joined_lines.append(current_chunk)
            current_chunk = ""
            continue

        next_line = lines[index + 1] or '\u3164'
        next_addition_length = len(next_line) + len(separator)

        if len(current_chunk) + next_addition_length > max_chars:
            joined_lines.append(current_chunk)
            current_chunk = ""

    if current_chunk:
        joined_lines.append(current_chunk)

    return joined_lines


def unjoin_sentences(original_sentence: str, modified_sentence: str, separator: str):
    """
    Splits the original and modified sentences into lines based on the separator.
    Tries to match the number of lines between the original and modified sentences.
    """

    if original_sentence is None:
        return ' '

    # split by separator, remove double spaces and empty or only space strings from list
    original_lines = original_sentence.split(separator)
    original_lines = [s.strip().replace('  ', ' ').lstrip(" ,.:;)") if s.strip().replace('  ', ' ').lstrip(" ,.:;)") else s
                      for s in original_lines if s.strip()]
    original_lines = [s for s in original_lines if s]
    original_lines = [s for s in original_lines if s.strip()]

    if modified_sentence is None:
        return original_lines or ' '

    # fix strange formatation returned by google translate, case occuring
    modified_sentence = modified_sentence.replace(f"{separator_unjoin} ", f"{separator_unjoin}").replace(
        f" {separator_unjoin}", f"{separator_unjoin}").replace(
        f"{separator_unjoin}.", f".{separator_unjoin}").replace(f"{separator_unjoin},", f",{separator_unjoin}")

    # split by separator, remove double spaces and empty or only space strings from list
    modified_lines = modified_sentence.split(separator_unjoin)
    modified_lines = [s.strip().replace('  ', ' ').lstrip(" ,.:;)") if s.strip().replace('  ', ' ').lstrip(" ,.:;)") else s
                      for s in modified_lines if s.strip()]
    modified_lines = [s for s in modified_lines if s]
    modified_lines = [s for s in modified_lines if s.strip()]

    # if original lines is "silence" sign, doenst translate
    if original_lines == "..." or original_lines == "…":
        return original_lines

    # all ok, return lines
    if len(original_lines) == len(modified_lines):
        return modified_lines

    # zero words? return original sentence, removing separator
    original_word_count = sum(len(line.strip().split())
                              for line in original_lines)
    modified_word_count = len(' '.join(modified_lines).strip().split())
    if original_word_count == 0 or modified_word_count == 0:
        return original_sentence.replace(separator, ' ').replace('  ', ' ')

    # calculate proportion of words between original and translated
    modified_words_proportion = modified_word_count / original_word_count
    # list all modified words
    modified_words = ' '.join(modified_lines).replace(separator, "").replace(
        separator_unjoin, "").replace("  ", " ").strip().split(' ')

    new_modified_lines = []
    current_index = 0

    # reconstruct lines based on proportion of original and translated words
    for i in range(len(original_lines)):
        # Calculate the number of words for the current modified sentence
        num_words = int(
            round(len(original_lines[i].strip().split()) * modified_words_proportion))

        # Extract words from modified list
        generated_line = ' '.join(
            modified_words[current_index:current_index+num_words])

        # Update the current index
        current_index += num_words

        # append remaining if is the last loop
        if i == len(original_lines) - 1:
            tail = ' '.join(modified_words[current_index:])
            if tail:
                generated_line = ' '.join([generated_line, tail]).strip()
            current_index = len(modified_words)

        # Add modified sentence to the new list
        new_modified_lines.append(generated_line.replace("  ", " ").strip())

    # case it continues being shorter
    while len(new_modified_lines) < len(original_lines):
        new_modified_lines.append(new_modified_lines[-1])

    return new_modified_lines or original_lines or ' '


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="translate_utils",
        description="Translate one or more SRT files using LeGen translation helpers.",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        help="Path to an .srt file or a directory containing .srt files.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="Destination directory or .srt file. Defaults to the source folder.",
    )
    parser.add_argument(
        "--translate",
        required=True,
        help="Target language code (e.g., en, es, pt-BR).",
    )
    parser.add_argument(
        "--translate_engine",
        type=str.lower,
        choices=("google", "gemini"),
        default="google",
        help="Translation engine to use: google (default) or gemini.",
    )
    parser.add_argument(
        "--gemini_api_key",
        action="append",
        default=[],
        type=str,
        help=(
            "Gemini API key. Repeat or separate by comma/line break to add multiple keys "
            "(required for --translate_engine=gemini)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite translated files if they already exist.",
    )
    return parser


def _output_path_is_file(candidate: Path | None) -> bool:
    if candidate is None:
        return False
    if candidate.exists():
        return candidate.is_file()
    return candidate.suffix.lower() == ".srt"


def _derive_destination(source: Path, base_output: Path | None, target_language: str, input_root: Path | None = None) -> Path:
    suffix = f"_{target_language.lower()}.srt"
    if base_output is None:
        return source.with_name(f"{source.stem}{suffix}")

    if base_output.suffix.lower() == ".srt" and not base_output.is_dir():
        return base_output

    # If base_output is a directory (or intended to be one)
    if input_root and source.is_relative_to(input_root):
        rel_path = source.relative_to(input_root)
        dest_dir = base_output / rel_path.parent
        return dest_dir / f"{source.stem}{suffix}"

    return base_output / f"{source.stem}{suffix}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_cli_parser()
    raw_argv = list(argv) if argv is not None else []
    args = parser.parse_args(raw_argv if argv is not None else None)

    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"Input path '{args.input_path}' does not exist.")

    output_path = Path(args.output_path).expanduser().resolve() if hasattr(args, "output_path") and args.output_path else None
    target_language = args.translate.strip()
    if not target_language or target_language.lower() == "none":
        parser.error("Provide a valid target language via --translate (e.g., en, es, pt-BR).")

    gemini_api_keys = normalize_api_keys(getattr(args, "gemini_api_key", []))
    translate_engine_explicit = any(str(item).startswith("--translate_engine") for item in raw_argv)
    if (
        not translate_engine_explicit
        and target_language
        and target_language.lower() != "none"
        and args.translate_engine == "google"
        and gemini_api_keys
    ):
        args.translate_engine = "gemini"

    if args.translate_engine == "gemini" and not gemini_api_keys:
        parser.error("Gemini API key is required when --translate_engine=gemini.")

    input_root = None
    if input_path.is_file():
        if input_path.suffix.lower() != ".srt":
            parser.error("Input file must be an .srt file.")
        source_files = [input_path]
    elif input_path.is_dir():
        input_root = input_path
        source_files = sorted(input_path.rglob("*.srt"))
        if not source_files:
            parser.error(f"No .srt files found inside directory '{input_path}'.")
    else:
        parser.error(f"Input path '{input_path}' is neither a file nor a directory.")

    output_is_file = _output_path_is_file(output_path)
    if output_is_file and len(source_files) > 1:
        parser.error("When translating multiple files the output path must be a directory.")

    translated = 0
    skipped = 0
    target_suffix = f"_{target_language.lower()}.srt"
    for source in source_files:
        if source.name.lower().endswith(target_suffix):
            skipped += 1
            continue

        destination = _derive_destination(source, output_path, target_language, input_root)
        if destination.exists() and not getattr(args, "overwrite", False):
            print(f"Skipping existing file {destination}")
            skipped += 1
            continue

        translate_srt_file(
            source,
            destination,
            target_language,
            translate_engine=args.translate_engine,
            gemini_api_keys=gemini_api_keys,
            overwrite=getattr(args, "overwrite", False),
        )
        print(f"Translated {source} -> {destination}")
        translated += 1

    total = len(source_files)
    print(f"Finished translating {translated}/{total} file(s). {skipped} skipped.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
