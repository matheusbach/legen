import os
import re
import tkinter as tk
from pathlib import Path

import pysrt


def SaveSegmentsToSrt(segments: list, output_path: Path):
    # Create the subtitle file
    subs = pysrt.SubRipFile()
    sub_idx = 1

    for i in range(len(segments)):
        start_time = segments[i]["start"]
        end_time = segments[i]["end"]
        duration = end_time - start_time
        timestamp = f"{start_time:.3f} - {end_time:.3f}"
        text = segments[i]["text"]

        sub = pysrt.SubRipItem(index=sub_idx, start=pysrt.SubRipTime(seconds=start_time),
                               end=pysrt.SubRipTime(seconds=end_time), text=text)
        subs.append(sub)
        sub_idx += 1

    # make dir and save .srt
    os.makedirs(output_path.parent, exist_ok=True)
    subs.save(output_path)


def string_width(text, font_name="Jost", font_size=18):
    """
    Determines the width of a string using tkinter.
    """
    tries_remaining = 5
    
    while (tries_remaining > 0):
        tries_remaining -= 1
        try:
            root = tk.Tk()
            width = tk.font.Font(name=font_name, size=font_size,
                                weight="bold").measure(text)
            root.destroy()
            return width
        except Exception:
            pass

    # all failed, return 60% of height per char
    return len(text) * font_size * 0.60


def is_punctuation_end(word):
    """Verifica se a palavra termina com uma pontuação."""
    return any(word.endswith(punct) for punct in ['.', ',', '!', '?', ':', ';'])


def split_segments(segments, max_width_px=1440, font_name="Jost", font_size=18):
    """
    Split segments based on the max width provided.
    """
    new_segments = []
    for segment in segments:
        words = segment['words']
        current_words = []
        current_width = 0

        for word in words:
            # Calculate the width with a space after the word
            added_width = string_width(
                word['word'] + " ", font_name, font_size)
            isolated_sentence_ending = is_punctuation_end(word['word']) and not (
                current_words and is_punctuation_end(current_words[-1]['word']))
            possible_logical_break_point = len(current_words) >= 2 and len(
                current_words[-1]['word']) <= 3 and not len(current_words[-2]['word']) <= 3

            if (current_width + added_width < max_width_px) or len(current_words) == 0 or isolated_sentence_ending or possible_logical_break_point:
                current_words.append(word)
                current_width += added_width
            else:
                new_segments.append({
                    'text': ' '.join(word['word'] for word in current_words),
                    'start': next((word['start'] for word in current_words if 'start' in word), segment['start']),
                    'end': next((word['end'] for word in reversed(current_words) if 'end' in word), segment['end']),
                    'words': current_words.copy()
                })
                current_words = [word]
                current_width = added_width

        # For any remaining words
        if current_words:
            new_segments.append({
                'text': ' '.join(word['word'] for word in current_words),
                'start': next((word['start'] for word in current_words if 'start' in word), segment['start']),
                'end': next((word['end'] for word in reversed(current_words) if 'end' in word), segment['end']),
                'words': current_words
            })

    return new_segments


def split_string_to_max_lines(text, max_width=720, max_lines=2, font_name="Jost", font_size=18):
    threshold = max_width * 0.8
    total_text_width = string_width(text, font_name, font_size)

    if total_text_width <= threshold or max_lines < 2:
        return [text]

    words = text.split()
    lines = []
    current_line_words = []
    current_line_width = 0

    for i, word in enumerate(words):
        word_width = string_width(word + ' ', font_name, font_size)
        isolated_sentence_ending = is_punctuation_end(word) and not (
            current_line_words and is_punctuation_end(current_line_words[-1]))
        possible_logical_break_point = len(current_line_words) >= 2 and len(
            current_line_words[-1]) <= 3 and not len(current_line_words[-2]) <= 3

        if current_line_width + word_width < total_text_width / max_lines or len(current_line_words) == 0 or isolated_sentence_ending or possible_logical_break_point:
            current_line_words.append(word)
            current_line_width += word_width
        else:
            lines.append(' '.join(current_line_words))
            current_line_words = [word]
            current_line_width = word_width

        if len(lines) == max_lines - 1:
            remaining_words = words[i:]
            lines.append(' '.join(remaining_words))
            break

    if current_line_words and len(lines) < max_lines:
        lines.append(' '.join(current_line_words))

    return lines


def adjust_times(segments, extra_end_time=1.0):
    for i in range(len(segments) - 1):  # We don't need to check the last segment
        current_end = segments[i]['end']
        next_start = segments[i + 1]['start']

        gap = next_start - current_end

        # If the gap is more than 1.5 + extra_end_time
        if gap > 1.5 + extra_end_time:
            segments[i]['end'] = current_end + extra_end_time

        # If the gap is less than 1.5 + extra_end_time
        elif gap < 1.5 + extra_end_time:
            segments[i]['end'] = next_start

    return segments


def format_segments(segments: list, max_line_width_px: int = 380, max_lines_per_segment: int = 2):
    print('Formatting segments...', end='', flush=True)

    segments = split_segments(
        segments, max_line_width_px * max_lines_per_segment)

    for segment in segments:
        segment["text"] = "\n".join(split_string_to_max_lines(
            text=segment["text"], max_width=max_line_width_px, max_lines=max_lines_per_segment))

    segments = adjust_times(segments)
    
    print('\r                      ', end='\r', flush=True)

    return segments
