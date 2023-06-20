import os
import time

import deep_translator
import pysrt
from tqdm import tqdm

# all entence endings for japanese and normal people languages
sentence_endings = ['.', '!', '?', ')', 'よ',
                    'ね', 'の', 'さ', 'ぞ', 'な', 'か', '！', '。', '」', '…']

# a good separator is a char or string that doenst change the translation quality but is near ever preserved in result at same or near position
separator = " ◌ "
separator_unjoin = separator.replace(' ', '')


def translate_srt_file(srt_file_path, translated_subtitle_path, target_lang):
    # Load the original SRT file
    subs = pysrt.open(srt_file_path, encoding='utf-8')

    # Extract the subtitle content and store it in a list
    sub_content = [sub.text for sub in subs]

    translated_texts = []
    chunk_max_chars = 4000
    chunks = []
    unjoined_texts = []

    # make chunks of at maximum $chunk_max_chars to stay under Google Translate public API limits
    for line in sub_content:
        chunks.append(line)
    # join lines in each chunk
    chunks = join_sentences(chunks, chunk_max_chars)

    for chunk in tqdm(chunks, desc="Translating", unit="chunks", unit_scale=True, leave=True, bar_format="{desc} {percentage:3.0f}% | ETA: {remaining}"):
        while True:
            try:
                # Translate the subtitle content of the chunk using Google Translate
                translated_texts = deep_translator.GoogleTranslator(
                    source='auto', target=target_lang).translate(chunk)
                
                # Unjoin lines within each chunk that end with a sentence ending
                unjoined_texts.extend(unjoin_sentences(chunk, translated_texts, separator_unjoin))
                break
            except Exception as e:
                # If an error occurred, retry
                print(f"Error occurred: {e}. Retrying in 30 seconds...")
                time.sleep(30)

    # Combine the original and translated subtitle content
    for i, sub in enumerate(subs):
        sub.text = unjoined_texts[i]

    # Save the translated SRT file
    os.makedirs(os.path.dirname(translated_subtitle_path), exist_ok=True)
    subs.save(translated_subtitle_path, encoding='utf-8')

    return subs


def join_sentences(lines, max_chars):
    """
    Joins the given list of strings in a way that each part ends with a sentence ending.
    Adds a separator to all lines in the chunk.
    """
    joined_lines = []
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + len(separator) <= max_chars:
            current_chunk += line + separator
            if any(line.endswith(ending) for ending in sentence_endings):
                joined_lines.append(current_chunk.strip())
                current_chunk = ""
        else:
            if current_chunk:
                joined_lines.append(current_chunk.strip())
                current_chunk = ""
            if len(current_chunk) + len(line) + len(separator) <= max_chars:
                current_chunk += line + separator
            else:
                # if a single line exceed max_chars, use maximum posible number of words. Discart the remaining
                end_index = line.rfind(' ', 0, max_chars - 1)

                if end_index == -1:
                    end_index = max_chars - 1

                joined_lines.append((line[:end_index] + '…')[:max_chars])
                    
    # append a chunk wich doenst have a formal end with sentence endings
    if current_chunk:
        joined_lines.append(current_chunk.strip())

    return joined_lines

def unjoin_sentences(original_sentence, modified_sentence, separator):
    """
    Splits the original and modified sentences into lines based on the separator.
    Tries to match the number of lines between the original and modified sentences.
    """
    
    if modified_sentence is None and original_sentence is not None:
        return original_sentence
    
    # split by separator, remove double spaces and empty or only space strings strings from list
    original_lines = original_sentence.split(separator)
    original_lines = [s.strip().replace('  ', ' ') for s in original_lines if s.strip()]
    original_lines = [s for s in original_lines if s]
    original_lines = [s for s in original_lines if s.strip()]
    # split by separator, remove double spaces and empty or only space strings from list
    modified_lines = modified_sentence.split(separator)
    modified_lines = [s.strip().replace('  ', ' ') for s in modified_lines if s.strip()]
    modified_lines = [s for s in modified_lines if s]
    modified_lines = [s for s in modified_lines if s.strip()]
    
    # if original lines is "silence" sign, doenst translate
    if original_lines == "..." or original_lines == "…":
        return original_lines

    # all ok, return lines
    if len(original_lines) == len(modified_lines):
        return modified_lines

    # zero words? return original sentence, removing separator
    original_word_count = sum(len(line.strip().split()) for line in original_lines)
    modified_word_count = len(' '.join(modified_lines).strip().split())
    if original_word_count == 0 or modified_word_count == 0:
        return original_sentence.replace(separator, ' ').replace('  ', ' ')
    
    # calculate proportion of words between original and translated
    modified_words_proportion = modified_word_count / original_word_count
    # list all modified words
    modified_words = ' '.join(modified_lines).replace(separator, "").replace(separator_unjoin, "").replace("  ", " ").strip().split(' ')
        
    new_modified_lines = []
    current_index = 0

    # reconstruct lines based on proportion of original and translated words
    for i in range(len(original_lines)):
        # Calculate the number of words for the current modified sentence
        num_words = int(round(len(original_lines[i].strip().split()) * modified_words_proportion))

        # Extract words from modified list
        generated_line = ' '.join(modified_words[current_index:current_index+num_words])
        
        # Update the current index
        current_index += num_words
        
        # append remaining if is the last loop
        if i == len(original_lines) - 1:
            ' '.join([generated_line, ' '.join(modified_words[current_index:])])

        # Add modified sentence to the new list
        new_modified_lines.append(generated_line.replace("  ", " ").strip())

    return new_modified_lines