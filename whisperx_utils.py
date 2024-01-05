import os
from pathlib import Path

import whisperx
import whisper # only for detect language

import whisper_utils
import subtitle_utils
from utils import time_task

from tqdm import tqdm


def transcribe_audio(model: whisperx.asr.WhisperModel, audio_path: Path, srt_path: Path, lang: str = None, device: str = "cpu", batch_size: int = 4):
    audio = whisperx.load_audio(file=audio_path.as_posix(), sr=model.model.feature_extractor.sampling_rate)

    # Define a variable to keep track of progress
    progress_bar = tqdm(position=1, total=100, ascii="░▒█", desc="WhisperX", unit='', unit_scale=True, leave=False,
                        bar_format="{desc} [{bar}] {percentage:3.0f}% | {rate_fmt}{postfix} | ETA: {remaining} | ⏱: {elapsed}",
                        dynamic_ncols=True, miniters=1)  # Set miniters to 1 for dynamic updating

    # Define the progress callback function
    def progress_callback(state, current: int = None, total: int = None):
        args = state, current, total
        args = [arg for arg in args if arg is not None]

        if len(args) == 1:
            state = args[0]
        if len(args) > 1:
            total = args[-1]
            current = args[-2]
        if len(args) > 2:
            state = args[-3]

        try:
            if type(state) == 'string':
                progress_bar.desc = state
            else:
                progress_bar.desc = state.value
        except:
            progress_bar.desc = "WhisperX"
        if total:
            progress_bar.total = total
        if current:
            progress_bar.n = current
            progress_bar.last_print_n = current
        progress_bar.refresh()

    # Transcribe
    with time_task("Running WhisperX transcription engine...", end='/n'):
        transcribe = model.transcribe(audio=audio, language=lang, batch_size=batch_size, on_progress=progress_callback)

    # Align if possible
    if lang in whisperx.alignment.DEFAULT_ALIGN_MODELS_HF or lang in whisperx.alignment.DEFAULT_ALIGN_MODELS_TORCH:
        with time_task(message_start="Running alignment..."):
            try:
                model_a, metadata = whisperx.load_align_model(language_code=lang, device="cuda")
                transcribe = whisperx.align(transcript=transcribe["segments"], model=model_a, align_model_metadata=metadata, audio=audio, device="cuda", return_char_alignments=True, on_progress=progress_callback)
            except Exception:
                model_a, metadata = whisperx.load_align_model(language_code=lang, device="cpu")  # force load on cpu due errors on gpu
                transcribe = whisperx.align(transcript=transcribe["segments"], model=model_a, align_model_metadata=metadata, audio=audio, device="cpu", return_char_alignments=True, on_progress=progress_callback)
    else:
        print(f"Language {lang} not suported for alignment. Skipping this step")

    # Format subtitles
    segments = subtitle_utils.format_segments(transcribe['segments'])

    # Save the subtitle file
    subtitle_utils.SaveSegmentsToSrt(segments, srt_path)

    return transcribe


def detect_language(model: whisperx.asr.WhisperModel, audio_path: Path):
    try:
        if os.getenv("COLAB_RELEASE_TAG"):
            raise Exception("Method invalid for Google Colab") 
        audio = whisperx.load_audio(audio_path.as_posix(), model.model.feature_extractor.sampling_rate)
        audio = whisper.pad_or_trim(audio, model.model.feature_extractor.n_samples)
        mel = whisperx.asr.log_mel_spectrogram(audio, n_mels=model.model.model.n_mels)
        encoder_output = model.model.encode(mel)
        results = model.model.model.detect_language(encoder_output)
        language_token, language_probability = results[0][0]
        return language_token[2:-2]
    except:
        print("using whisper base model for detection: ", end='')
        whisper_model = whisper.load_model("base", device="cpu", in_memory=True)
        return whisper_utils.detect_language(model=whisper_model, audio_path=audio_path)
