import os
from pathlib import Path

import pysrt
import whisper
import whisper.transcribe
import whisperx
import subtitle_utils
from utils import time_task


def transcribe_audio(model: whisper.model, audio_path: Path, srt_path: Path, lang: str = None, disable_fp16: bool = False):
    # Load audio
    audio = whisper.load_audio(file=audio_path.as_posix())
    
    # Transcribe
    with time_task():
        transcribe = model.transcribe(audio=audio, language=lang, fp16=False if disable_fp16 else True, verbose=False)

    # Align if possible
    if lang in whisperx.alignment.DEFAULT_ALIGN_MODELS_HF or lang in whisperx.alignment.DEFAULT_ALIGN_MODELS_TORCH:
        with time_task(message_start="Running alignment..."):
            try:
                model_a, metadata = whisperx.load_align_model(language_code=lang, device="cuda")
                transcribe = whisperx.align(transcript=transcribe["segments"], model=model_a, align_model_metadata=metadata, audio=audio, device="cuda", return_char_alignments=True)
            except Exception:
                model_a, metadata = whisperx.load_align_model(language_code=lang, device="cpu")  # force load on cpu due errors on gpu
                transcribe = whisperx.align(transcript=transcribe["segments"], model=model_a, align_model_metadata=metadata, audio=audio, device="cpu", return_char_alignments=True)
    else:
        print(f"Language {lang} not suported for alignment. Skipping this step")

    # Format subtitles
    segments = subtitle_utils.format_segments(transcribe['segments'])

    # Save the subtitle file
    subtitle_utils.SaveSegmentsToSrt(segments, srt_path)

    return transcribe


def detect_language(model: str, audio_path: Path):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path.as_posix())
    audio = whisper.pad_or_trim(audio)
    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    return max(probs, key=probs.get)
