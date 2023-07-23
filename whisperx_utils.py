import os
from pathlib import Path

import pysrt
import whisperx

batch_size = 4  # reduce if low on GPU mem


def transcribe_audio(model: whisperx.asr.WhisperModel, audio_path: Path, srt_path: Path, lang: str = None, disable_fp16: bool = False, device: str = "cpu"):
    audio = whisperx.load_audio(file=audio_path.as_posix())

    # Transcribe
    transcribe = model.transcribe(audio=audio, language=lang, batch_size=batch_size)

    # Align   # Disable for while dont working
    model_a, metadata = whisperx.load_align_model(language_code=lang, device="cpu")  # force load on cpu due errors on gpu
    transcribe = whisperx.align(transcript=transcribe["segments"], model=model_a, align_model_metadata=metadata, audio=audio, device="cpu", return_char_alignments=False)

    segments = transcribe['segments']

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
    os.makedirs(srt_path.parent, exist_ok=True)
    subs.save(srt_path)

    return transcribe


def detect_language(model: whisperx.asr.WhisperModel, audio_path: Path):
    audio = whisperx.load_audio(file=audio_path.as_posix())
    return model.detect_language(audio=audio)
