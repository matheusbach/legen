import argparse
import os
import subprocess
import time
from inspect import currentframe, getframeinfo
from pathlib import Path

import ffmpeg_utils
import file_utils
import translate_utils
from utils import time_task, audio_extensions, video_extensions, check_other_extensions

version = "v0.16"

# Terminal colors
default = "\033[1;0m"
gray = "\033[1;37m"
wblue = "\033[1;36m"
blue = "\033[1;34m"
yellow = "\033[1;33m"
green = "\033[1;32m"
red = "\033[1;31m"

print(f"""
{blue}888              {gray} .d8888b.                   
{blue}888              {gray}d88P  Y88b                  
{blue}888              {gray}888    888                  
{blue}888      .d88b.  {gray}888         .d88b.  88888b. 
{blue}888     d8P  Y8b {gray}888  88888 d8P  Y8b 888 "88b
{blue}888     88888888 {gray}888    888 88888888 888  888
{blue}888     Y8b.     {gray}Y88b  d88P Y8b.     888  888
{blue}88888888 "Y8888  {gray} "Y8888P88  "Y8888  888  888

legen {version} - github.com/matheusbach/legen{default}
python {__import__('sys').version}
""")
time.sleep(1.5)

# Define parameters and configurations
parser = argparse.ArgumentParser(prog="LeGen", description="Uses AI to locally transcribes speech from media files, generating subtitle files, translates the generated subtitles, inserts them into the mp4 container, and burns them directly into video",
                                 argument_default=True, allow_abbrev=True, add_help=True, usage='LeGen -i INPUT_PATH [other options]')
parser.add_argument("-i", "--input_path",
                    help="Path to media files. Can be a folder containing files or an individual file", required=True, type=Path)
parser.add_argument("--norm", default=False, action="store_true",
                    help="Normalize folder times and run vidqa on input_path before starting processing files")
parser.add_argument("-ts:e", "--transcription_engine", type=str, default="whisperx",
                    help="Transcription engine. Possible values: whisperx (default), whisper")
parser.add_argument("-ts:m", "--transcription_model", type=str, default="medium",
                    help="Path or name of the Whisper transcription model. A larger model will consume more resources and be slower, but with better transcription quality. Possible values: tiny, base, small, medium (default), large, ...")
parser.add_argument("-ts:d", "--transcription_device", type=str, default="auto",
                    help="Device to run the transcription through Whisper. Possible values: auto (default), cpu, cuda")
parser.add_argument("-ts:c", "--transcription_compute_type", type=str, default="auto",
                    help="Quantization for the neural network. Possible values: auto (default), int8, int8_float32, int8_float16, int8_bfloat16, int16, float16, bfloat16, float32")
parser.add_argument("-ts:b", "--transcription_batch", type=int, default=4,
                    help="Number of simultaneous segments being transcribed. Higher values will speed up processing. If you have low RAM/VRAM, long duration media files or have buggy subtitles, reduce this value to avoid issues. Only works using transcription_engine whisperx. (default: 4)")
parser.add_argument("--translate", type=str, default="none",
                    help="Translate subtitles to language code if not the same as origin. (default: don't translate)")
parser.add_argument("--input_lang", type=str, default="auto",
                    help="Indicates (forces) the language of the voice in the input media (default: auto)")
parser.add_argument("-c:v", "--codec_video", type=str, default="h264", metavar="VIDEO_CODEC",
                    help="Target video codec. Can be used to set acceleration via GPU or another video API [codec_api], if supported (ffmpeg -encoders). Ex: h264, libx264, h264_vaapi, h264_nvenc, hevc, libx265 hevc_vaapi, hevc_nvenc, hevc_cuvid, hevc_qsv, hevc_amf (default: h264)")
parser.add_argument("-c:a", "--codec_audio", type=str, default="aac", metavar="AUDIO_CODEC",
                    help="Target audio codec. (default: aac). Ex: aac, libopus, mp3, vorbis")
parser.add_argument("-o:s", "--output_softsubs", default=None, type=Path,
                    help="Path to the folder or output file for the video files with embedded softsub (embedded in the mp4 container and .srt files). (default: softsubs_ + input_path)")
parser.add_argument("-o:h", "--output_hardsubs", default=None, type=Path,
                    help="Output folder path for video files with burned-in captions and embedded in the mp4 container. (default: hardsubs_ + input_path)")
parser.add_argument("--overwrite", default=False, action="store_true",
                    help="Overwrite existing files in output directories")
parser.add_argument("--disable_srt", default=False, action="store_true",
                    help="Disable .srt file generation and don't insert subtitles in mp4 container of output_softsubs")
parser.add_argument("--disable_softsubs", default=False, action="store_true",
                    help="Don't insert subtitles in mp4 container of output_softsubs. This option continues generating .srt files")
parser.add_argument("--disable_hardsubs", default=False, action="store_true",
                    help="Disable subtitle burn in output_hardsubs")
parser.add_argument("--copy_files", default=False, action="store_true",
                    help="Copy other (non-video) files present in input directory to output directories. Only generate the subtitles and videos")
args = parser.parse_args()

if not args.output_softsubs and not args.input_path.is_file():
    args.output_softsubs = compatibility_path if (compatibility_path := Path(args.input_path.parent, "legen_srt_" + args.input_path.name)).exists() else Path(args.input_path.parent, "softsubs_" + args.input_path.name)
if not args.output_hardsubs and not args.input_path.is_file():
    args.output_hardsubs = compatibility_path if (compatibility_path := Path(args.input_path.parent, "legen_burned_" + args.input_path.name)).exists() else Path(args.input_path.parent, "hardsubs_" + args.input_path.name)

if args.transcription_device == "auto":
    import torch
    torch_device = ("cuda" if torch.cuda.is_available() else "cpu")
else:
    torch_device = str.lower(args.transcription_device)

transcription_compute_type = args.transcription_compute_type if args.transcription_compute_type != "default" else "float16" if not torch_device == "cpu" else "float32"

args.transcription_model = "large-v3" if args.transcription_model == "large" else args.transcription_model

# ----------------------------------------------------------------------------

if args.norm:
    # normalize video using vidqa
    with time_task(message_start=f"Running {wblue}vidqa{default} and updating folder modifiation times in {gray}{args.input_path}{default}", end="\n"):
        subprocess.run(["vidqa", "-i", args.input_path, "-m", "unique", "-fd",
                        Path(Path(getframeinfo(currentframe()).filename).resolve().parent, "vidqa_data")])
        # update folder time structure
        file_utils.update_folder_times(args.input_path)

# load whisper model
with time_task(message_start=f"\nLoading {args.transcription_engine} model: {wblue}{args.transcription_model}{default} ({transcription_compute_type}) on {wblue}{torch_device}{default}", end="\n"):
    if args.transcription_engine == 'whisperx':
        import whisperx
        import whisperx_utils
    
        whisper_model = whisperx.load_model(
            whisper_arch=args.transcription_model, device=torch_device, compute_type=transcription_compute_type, asr_options={"repetition_penalty": 1, "prompt_reset_on_temperature": 0.5, "no_repeat_ngram_size": 2,})
    elif args.transcription_engine == 'whisper':
        import whisper

        import whisper_utils
        whisper_model = whisper.load_model(
            name=args.transcription_model, device=torch_device, in_memory=True)
    else:
        raise ValueError(f'Unsupported transcription engine {args.transcription_engine}. Supported values: whisperx, whisper')

with time_task(message="⌛ Processing files for"):
    path: Path
    for path in (item for item in sorted(sorted(Path(args.input_path).rglob('*'), key=lambda x: x.stat().st_mtime), key=lambda x: len(x.parts)) if item.is_file()):
        rel_path = path.relative_to(args.input_path)
        with time_task(message_start=f"\nProcessing {yellow}{rel_path.as_posix()}{default}", end="\n", message="⌚ Done in"):
            try:
                # define file type by extensions
                if path.suffix.lower() in video_extensions:
                    file_type = "video"
                elif path.suffix.lower() in audio_extensions:
                    file_type = "audio"
                else:
                    file_type = "other"

                if file_type == "video" or file_type == "audio":
                    # define paths
                    origin_media_path = path
                    dupe_filename = len(check_other_extensions(path, list(video_extensions | audio_extensions))) > 1
                    posfix_extension = path.suffix.lower().replace('.', '_') if dupe_filename else ''

                    softsub_video_dir = Path(args.output_softsubs, rel_path.parent)
                    burned_video_dir = Path(args.output_hardsubs, rel_path.parent)
                    # output video extension will be changed to .mp4
                    softsub_video_path = Path(args.output_softsubs, rel_path.stem + posfix_extension + ".mp4")
                    hardsub_video_path = Path(burned_video_dir, rel_path.stem + posfix_extension + ".mp4")
                    subtitle_translated_path = Path(
                        softsub_video_dir, rel_path.stem + posfix_extension + f"_{args.translate}.srt")
                    subtitles_path = []

                    if args.input_lang == "auto":
                        # extract audio
                        audio_short_extracted = file_utils.TempFile(
                            None, file_ext=".wav")
                        ffmpeg_utils.extract_short_wav(
                            origin_media_path, audio_short_extracted.getpath())
                        # detect language
                        print("Detecting audio language: ", end='', flush=True)
                        if args.transcription_engine == 'whisperx':
                            audio_language = whisperx_utils.detect_language(
                                whisper_model, audio_short_extracted.getpath())
                        if args.transcription_engine == 'whisper':
                            audio_language = whisper_utils.detect_language(
                                whisper_model, audio_short_extracted.getpath())
                        print(f"{gray}{audio_language}{default}")

                        audio_short_extracted.destroy()
                    else:
                        audio_language = args.input_lang
                        print(f"Forced input audio language: {gray}{audio_language}{default}")
                    # set path after get transcribed language
                    subtitle_transcribed_path = Path(
                        softsub_video_dir, rel_path.stem + posfix_extension + f"_{audio_language}.srt")
                    # create temp file for .srt
                    transcribed_srt_temp = file_utils.TempFile(
                        subtitle_transcribed_path, file_ext=".srt")
                    # skip transcription if transcribed srt for this language is existing (without overwrite neabled) or will not be used in LeGen process
                    if (file_utils.file_is_valid(subtitle_transcribed_path)) or ((args.disable_hardsubs or file_utils.file_is_valid(hardsub_video_path)) and (args.disable_srt or file_utils.file_is_valid(subtitle_transcribed_path))) and not args.overwrite:
                        print("Transcription is unnecessary. Skipping.")
                    else:
                        # extract audio
                        audio_extracted = file_utils.TempFile(None, file_ext=".wav")
                        ffmpeg_utils.extract_audio_wav(
                            origin_media_path, audio_extracted.getpath())
                        # transcribe saving subtitles to temp .srt file
                        if args.transcription_engine == 'whisperx':
                            print(f"{wblue}Transcribing{default} with {gray}WhisperX{default}")
                            whisperx_utils.transcribe_audio(
                                whisper_model, audio_extracted.getpath(), transcribed_srt_temp.getpath(), audio_language, device=torch_device, batch_size=args.transcription_batch)
                        if args.transcription_engine == 'whisper':
                            print(f"{wblue}Transcribing{default} with {gray}Whisper{default}")
                            whisper_utils.transcribe_audio(
                                model=whisper_model, audio_path=audio_extracted.getpath(), srt_path=transcribed_srt_temp.getpath(), lang=audio_language, disable_fp16=False if transcription_compute_type == "float16" or transcription_compute_type == "fp16" else True)

                        audio_extracted.destroy()
                        # if save .srt is enabled, save it to destination dir, also update path with language code
                        if not args.disable_srt:
                            transcribed_srt_temp.save()
                    subtitles_path.append(transcribed_srt_temp.getvalidpath())
                    # translate transcribed subtitle using Google Translate if transcribed language is not equals to target
                    # skip translation if translation has not requested, has equal source and output language, if file is existing (without overwrite neabled) or will not be used in LeGen process
                    if args.translate == "none":
                        pass # translation not requested
                    elif args.translate == audio_language:
                        print("Translation is unnecessary because input and output language are the same. Skipping.")
                    elif (args.disable_hardsubs or file_utils.file_is_valid(hardsub_video_path)) and (args.disable_srt or (file_utils.file_is_valid(subtitle_translated_path) and file_utils.file_is_valid(subtitle_transcribed_path) and file_utils.file_is_valid(subtitle_translated_path))) and not args.overwrite:
                        print("Translation is unnecessary. Skipping.")
                        subtitles_path.insert(0, subtitle_translated_path)
                    elif file_utils.file_is_valid(subtitle_translated_path):
                        print("Translated file found. Skipping translation.")
                        subtitles_path.insert(0, subtitle_translated_path)
                    elif transcribed_srt_temp.getvalidpath():
                        # create the temp .srt translated file
                        translated_srt_temp = file_utils.TempFile(
                            subtitle_translated_path, file_ext=".srt")

                        # translating with google translate public API
                        print(f"{wblue}Translating{default} with {gray}Google Translate{default}")
                        subs = translate_utils.translate_srt_file(
                            transcribed_srt_temp.getvalidpath(), translated_srt_temp.getpath(), args.translate)
                        if not args.disable_srt:
                            translated_srt_temp.save()

                        subtitles_path.insert(0, translated_srt_temp.getvalidpath())
                    if not args.disable_softsubs:
                        if file_utils.file_is_valid(softsub_video_path) and not args.overwrite:
                            print(f"Existing video file {gray}{softsub_video_path}{default}. Skipping subtitle insert")
                        else:
                            # create the temp .mp4 with srt in video container
                            video_softsubs_temp = file_utils.TempFile(
                                softsub_video_path, file_ext=".mp4")

                            # insert subtitle into container using ffmpeg
                            print(f"{wblue}Inserting subtitle{default} in mp4 container using {gray}FFmpeg{default}")
                            ffmpeg_utils.insert_subtitle(input_media_path=origin_media_path, subtitles_path=subtitles_path,
                                                        burn_subtitles=False, output_video_path=video_softsubs_temp.getpath(),
                                                        codec_video=args.codec_video, codec_audio=args.codec_audio)
                            video_softsubs_temp.save()
                    if not args.disable_hardsubs:
                        if file_utils.file_is_valid(hardsub_video_path) and not args.overwrite:
                            print(f"Existing video file {gray}{hardsub_video_path}{default}. Skipping subtitle burn")
                        else:
                            # create the temp .mp4 with srt in video container
                            video_hardsubs_temp = file_utils.TempFile(
                                hardsub_video_path, file_ext=".mp4")
                            # insert subtitle into container and burn using ffmpeg
                            print(f"{wblue}Inserting subtitle{default} in mp4 container and {wblue}burning{default} using {gray}FFmpeg{default}")
                            ffmpeg_utils.insert_subtitle(input_media_path=origin_media_path, subtitles_path=subtitles_path,
                                                        burn_subtitles=True, output_video_path=video_hardsubs_temp.getpath(),
                                                        codec_video=args.codec_video, codec_audio=args.codec_audio)
                            video_hardsubs_temp.save()
                else:
                    print("not a video file")
                    if args.copy_files:
                        if not args.disable_srt:
                            # copia o arquivo extra para pasta que contém também os arquivos srt
                            file_utils.copy_file_if_different(path, Path(
                                args.output_softsubs, rel_path))
                        if not args.disable_hardsubs:
                            # copia o arquivo extra para pasta que contém os videos queimados
                            file_utils.copy_file_if_different(path, Path(
                                args.output_hardsubs, rel_path))
            except Exception as e:
                file = path.as_posix()
                print(f"{red}ERROR !!!{default} {file}")
                print(f"{yellow}check legen-errors.txt for details{default}")
                # extract the relevant information from the exception object
                current_time = time.strftime("%y/%m/%d %H:%M:%S", time.localtime())

                error_message = f"[{current_time}] {file}: {type(e).__name__}: {str(e)}"
                # write the error message to a file
                with open(Path(Path(getframeinfo(currentframe()).filename).resolve().parent, "legen-errors.txt"), "a") as f:
                    f.write(error_message + "\n")
                    f.close()

    print("Deleting temp folder")
    file_utils.delete_folder(
        Path(Path(getframeinfo(currentframe()).filename).resolve().parent, "temp"))

    print(f"{green}Tasks done!{default}")
