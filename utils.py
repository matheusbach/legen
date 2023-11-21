import time
from pathlib import Path

from contextlib import contextmanager


@contextmanager
def time_task(message_start=None, end=' ', message="⏱ Took"):
    if message_start:
        print(message_start, end=end, flush=True)
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    formatted_elapsed_time = format_time(elapsed_time)
    print(f"{message} {formatted_elapsed_time}", flush=True)


def format_time(elapsed_time):
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Check and format the non-zero time units
    parts = []
    if hours:
        parts.append(f"{int(hours)}h")
    if minutes:
        parts.append(f"{int(minutes)}m")
    if seconds or not parts:
        # Add seconds if it's the only non-zero unit or if all units are zero
        parts.append(f"{int(seconds)}s")

    return ' '.join(parts)


def time_func(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_elapsed_time = format_time(elapsed_time)
        print(f"Execution time of {func.__name__}: {formatted_elapsed_time}")
        return result
    return wrapper

def check_other_extensions(file_path, extensions_to_check):
    """
    Check the existence of files with the same name but different extensions
    in the same folder.

    Parameters:
    - file_path (str): The path of the file to check.
    - extensions_to_check (list): List of extensions to check.

    Returns:
    - list: List of existing file paths with different extensions.
    """
    file_path = Path(file_path)
    folder = file_path.parent
    base_name = file_path.stem

    matching_files = [
        folder / (base_name + ext)
        for ext in extensions_to_check
        if (folder / (base_name + ext)).exists()
    ]

    return matching_files

video_extensions = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".vob", ".mts", ".m2ts", ".ts", ".yuv", ".mpg", ".mp2", ".mpeg", ".mpe", ".mpv", ".m2v", ".m4v", ".3gp", ".3g2", ".nsv", ".mts"}
audio_extensions = {".aa", ".aac", ".aax", ".act", ".aiff", ".alac", ".amr", ".ape", ".au", ".awb", ".dss", ".dvf", ".flac", ".gsm", ".iklax", ".ivs", ".m4a", ".m4b", ".m4p", ".mpga", ".mmf", ".mp3", ".mpc", ".msv", ".nmf", ".ogg", ".oga", ".mogg", ".opus", ".ra", ".rm", ".raw", ".rf64", ".sln", ".tta", ".voc", ".vox", ".wav", ".wma", ".wv", ".webm", ".8svx"}