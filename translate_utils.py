import asyncio
import os
from pathlib import Path

import deep_translator
import pysrt
import tqdm.asyncio
import subtitle_utils
from utils import format_time

# all entence endings for japanese and normal people languages
sentence_endings = ['.', '!', '?', ')', 'よ', 'ね',
                    'の', 'さ', 'ぞ', 'な', 'か', '！', '。', '」', '…']

# a good separator is a char or string that doenst change the translation quality but is near ever preserved in result at same or near position
separator = " ◌ "
separator_unjoin = separator.replace(' ', '')
chunk_max_chars = 4999


def translate_srt_file(srt_file_path: Path, translated_subtitle_path: Path, target_lang):
    # Load the original SRT file
    subs = pysrt.open(srt_file_path, encoding='utf-8')

    # Extract the subtitle content and store it in a list. Also rejoin all lines splited
    sub_content = [' '.join(sub.text.strip().splitlines()) for sub in subs]

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
    """
    Joins the given list of strings in a way that each part ends with a sentence ending.
    Adds a separator to all lines in the chunk.
    """
    joined_lines = []
    current_chunk = ""

    for line in lines:
        if not line or line is None:
            line = 'ㅤ'  # invisible char (not a simple space)

        if len(current_chunk) + len(line) + len(separator) <= max_chars:
            current_chunk += line + separator
            if any(line.endswith(ending) for ending in sentence_endings):
                joined_lines.append(current_chunk)
                current_chunk = ""
        else:
            if current_chunk:
                joined_lines.append(current_chunk)
                current_chunk = ""
            if len(current_chunk) + len(line) + len(separator) <= max_chars:
                current_chunk += line + separator
            else:
                # if a single line exceed max_chars, use maximum posible number of words. Discart the remaining
                end_index = line.rfind(
                    ' ', 0, max_chars - (1 + len(separator)))

                if end_index == - (1 + len(separator)):
                    end_index = max_chars - (1 + len(separator))

                joined_lines.append(
                    (line[:end_index] + '…' + separator)[:max_chars])

    # append a chunk wich doenst have a formal end with sentence endings
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
    modified_sentence.replace(f"{separator_unjoin} ", f"{separator_unjoin}").replace(f" {separator_unjoin}", f"{separator_unjoin}").replace(
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
            ' '.join([generated_line, ' '.join(
                modified_words[current_index:])])

        # Add modified sentence to the new list
        new_modified_lines.append(generated_line.replace("  ", " ").strip())

    # case it continues being shorter
    while len(new_modified_lines) < len(original_lines):
        new_modified_lines.append(new_modified_lines[-1])

    return new_modified_lines or original_lines or ' '
