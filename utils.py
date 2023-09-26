import time

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
