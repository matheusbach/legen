import os
from pathlib import Path
import re

import pysrt
import whisperx
import whisper # only for detect language

import subtitle_utils
#import faster_whisper
#import numpy as np

def transcribe_audio(model: whisperx.asr.WhisperModel, audio_path: Path, srt_path: Path, lang: str = None, device: str = "cpu", batch_size: int = 4):
    audio = whisperx.load_audio(file=audio_path.as_posix())

    # Transcribe
    transcribe = model.transcribe(
        audio=audio, language=lang, batch_size=batch_size)

    # Align
    model_a, metadata = whisperx.load_align_model(
        language_code=lang, device="cpu")  # force load on cpu due errors on gpu
    transcribe = whisperx.align(transcript=transcribe["segments"], model=model_a,
                                align_model_metadata=metadata, audio=audio, device="cpu", return_char_alignments=True)

    segments = transcribe['segments']

    subtitle_utils.SaveSegmentsToSrt(segments, srt_path.with_name("aaaaaa.srt"))

    # Format subtitles
    segments = subtitle_utils.format_segments(segments)

    # Save the subtitle file
    subtitle_utils.SaveSegmentsToSrt(segments, srt_path)

    return transcribe


def detect_language(model: whisperx.asr.WhisperModel, audio_path: Path):
    # load audio and pad/trim it to fit 30 seconds
    # audio = whisperx.load_audio(audio_path.as_posix(), 16000)
    # segment = whisperx.asr.log_mel_spectrogram(audio[: whisperx.asr.N_SAMPLES], padding=0 if audio.shape[0]
    #                                           >= whisperx.asr.N_SAMPLES else whisperx.asr.N_SAMPLES - audio.shape[0], device="cpu")
    # encoder_output = model.model.encode(segment)
    # results = model.model.model.detect_language(encoder_output).to("cpu")
    # language_token, language_probability = results[0][0]
    # return language_token[2:-2]

    # ABOVE CODE IS BEST, BUT ITS NOT WORKING FOR NOW IN SOME SYSTEMS. SAVE FOR THE FUTURE

    audio = whisper.load_audio(audio_path.as_posix(), 16000)
    audio = whisper.pad_or_trim(audio, whisperx.asr.N_SAMPLES)
    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to("cpu")
    whisper_model = whisper.load_model("base", device="cpu", in_memory=True)
    # detect the spoken language
    _, probs = whisper_model.detect_language(mel)
    return max(probs, key=probs.get)
