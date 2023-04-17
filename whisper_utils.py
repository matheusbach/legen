import os

import pysrt
import whisper
import whisper.transcribe


def transcribe_audio(model: str, audio_path: str, srt_path: str, lang: str = None, disable_fp16: bool = False):
    # Transcribe
    transcribe = model.transcribe(
        audio=audio_path, language=lang, fp16=False if disable_fp16 else True, verbose=False)

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
    os.makedirs(os.path.dirname(srt_path), exist_ok=True)
    subs.save(srt_path)

    return transcribe


def detect_language(model: str, audio_path: str):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    return max(probs, key=probs.get)
