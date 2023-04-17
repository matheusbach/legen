import os
import time

import deep_translator
import pysrt
from tqdm import tqdm


def translate_srt_file(srt_file_path, translated_subtitle_path, target_lang):
    # Load the original SRT file
    subs = pysrt.open(srt_file_path, encoding='utf-8')

    # Extract the subtitle content and store it in a list
    sub_content = [sub.text for sub in subs]

    translated_texts = []
    chunk_max_chars = 4000
    chunks = [[]]
    
    # make chunks of at maximum $chunk_max_chars to stay under Google Translate public API limits
    for line in sub_content:
        if sum(len(s) for s in chunks[-1]) + len(line) < chunk_max_chars:
                chunks[-1].append(line)
        else:
            chunks.append([line])
    
    for chunk in tqdm(chunks, desc="Translating", unit="chunks", unit_scale=True, leave=True, bar_format="{desc} {percentage:3.0f}% | ETA: {remaining}"):
        while True:
            try:
                # Translate the subtitle content of the chunk using Google Translate
                translated_chunk = deep_translator.GoogleTranslator(
                    source='auto', target=target_lang).translate_batch(chunk)
                translated_texts.extend(translated_chunk)
                break
            except Exception as e:
                # if an error ocurred, retry
                print(f"Error occurred: {e}. Retrying in 30 seconds...")
                time.sleep(30)

    # Combine the original and translated subtitle content
    for i, sub in enumerate(subs):
        sub.text = translated_texts[i]

    # Save the translated SRT file
    os.makedirs(os.path.dirname(translated_subtitle_path), exist_ok=True)
    subs.save(translated_subtitle_path, encoding='utf-8')

    return subs