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

version = "v0.15.6"

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

# define parâmetros e configuraçṍes
parser = argparse.ArgumentParser(prog="LeGen", description="Normaliza arquivos de vídeo, transcreve legendas a partir do áudio de arquivos de vídeo e áudio, traduz as legendas geradas, salva as legendas em arquivos .srt, insere no container mp4 e queima diretamente em vídeo",
                                 argument_default=True, allow_abbrev=True, add_help=True)
parser.add_argument("-i", "--input_dir", type=str,
                    help="Caminho da pasta onde os vídeos e/ou audios originais estão localizados.", required=True)
parser.add_argument("--norm", default=False, action="store_true",
                    help="Update folder times and run vidqa in input folder before start LeGen processing.")
parser.add_argument("--whisperx", default=False, action="store_true",
                    help="Use m-bain/whisperX instead openai/whisper. Unstable!")
parser.add_argument("--model", type=str, default="medium",
                    help="Caminho ou nome do modelo de transcrição Whisper. (default: medium)")
parser.add_argument("--dev", type=str, default="auto",
                    help="Dispositivo para rodar a transcrição pelo Whisper. [cpu, cuda, auto]. (default: auto)")
parser.add_argument("--compute_type", type=str, default="default",
                    help="Quantization for the neural network. Ex: float32, float16, int8, ...")
parser.add_argument("--batch_size", type=int, default="4",
                    help="The higher the value, the faster the processing will be. If you have low RAM or have buggy subtitles, reduce this value. Works only using whisperX. (default: 4)")
parser.add_argument("--lang", type=str, default="pt",
                    help="Idioma para o qual as legendas devem ser traduzidas. Language equals to source video skip translation (default: pt)")
parser.add_argument("--input_lang", type=str, default="auto",
                    help="Indica (força) idioma da voz das midias de entrada (default: auto)")
parser.add_argument("-c:v", "--video_codec", type=str, default="h264",
                    help="Codec de vídeo destino. Pode ser usado para definir aceleração via GPU ou outra API de video [codec_api], se suportado (ffmpeg -encoders). Ex: h264, libx264, h264_vaapi, h264_nvenc, hevc, libx265 hevc_vaapi, hevc_nvenc, hevc_cuvid, hevc_qsv, hevc_amf (default: h264)")
parser.add_argument("-c:a", "--audio_codec", type=str, default="aac",
                    help="Codec de audio destino. (default: aac). Ex: aac, libopus, mp3, vorbis")
parser.add_argument("--srt_out_dir", type=str, default=None,
                    help="Caminho da pasta de saída para os arquivos de vídeo com legenda embutida no container mp4 e arquivos SRT. (default: legen_srt_$input_dir)")
parser.add_argument("--burned_out_dir", type=str, default=None,
                    help="Caminho da pasta de saída para os arquivos de vídeo com legendas queimadas no vídeo e embutidas no container mp4. (default: legen_burned_$lang_$input_dir)")
parser.add_argument("--overwrite", default=False, action="store_true",
                    help="Overwrite existing files in output dirs")
parser.add_argument("--disable_srt", default=False, action="store_true",
                    help="Disable .srt file generation and don't insert subtitles in mp4 container of $srt_out_dir")
parser.add_argument("--disable_embed", default=False, action="store_true",
                    help="Don't insert subtitles in mp4 container of $srt_out_dir. This option continue generating .srt files")
parser.add_argument("--disable_burn", default=False, action="store_true",
                    help="Disable subtitle burn in $burned_out_dir")
parser.add_argument("--only_video", default=False, action="store_true",
                    help="Don't copy other (no video) files present in input dir to output dirs. Only generate the subtitles and videos")
parser.add_argument("--only_srt_subtitles", default=False, action="store_true",
                    help="Just generates the subtitles. Do not encode the videos or copy other files")
parser.add_argument("--sub_style", type=str, default="'Futura,PrimaryColour=&H03fcff,Fontsize=18,BackColour=&H80000000,Bold=1,Spacing=0.09,Outline=1,Shadow=0,MarginL=10,MarginR=10'",
                    help="Style of subtitle text. (default: 'Fontname=Futura,PrimaryColour=&H03fcff,Fontsize=18,BackColour=&H80000000,Bold=1,Spacing=0.09,Outline=1,Shadow=0,MarginL=10,MarginR=10').")
parser.add_argument("--sub_align", type=str, default="2",
                    help="Set the subtitles position: {'Bottom left': 1, 'Bottom center': 2, 'Bottom right': 3, 'Top left': 5, 'Top center': 6, 'Top right': 7, 'Middle left': 9, 'Middle center': 10, 'Middle right': 11}")
args = parser.parse_args()

input_dir: Path = Path(args.input_dir)
print("After parsing : " + repr(args.sub_style))
print("After replace : " + repr(args.sub_style.replace("'", "")))
# input needs to be a directory (is good to suport single file in future)
#if not input_dir.is_dir():
#    print(f"{red}Invalid input{default} {args.input_dir}\nYou must insert a folder (directory) as input parater. Create a new one containing the files(s) if necessary")
#    raise SystemExit

if args.srt_out_dir is None:
    args.srt_out_dir = Path(input_dir.parent, "legen_srt_" + input_dir.name)
srt_out_dir = args.srt_out_dir
if args.burned_out_dir is None:
    args.burned_out_dir = Path(
        input_dir.parent, "legen_burned_" + input_dir.name)
burned_out_dir = args.burned_out_dir

if args.dev == "auto":
    import torch
    torch_device = ("cuda" if torch.cuda.is_available() else "cpu")
else:
    torch_device = str.lower(args.dev)

compute_type = args.compute_type if args.compute_type != "default" else "float16" if not torch_device == "cpu" else "float32"

if args.only_srt_subtitles:
    args.only_video = True

args.model = "large-v3" if args.model == "large" else args.model

# ----------------------------------------------------------------------------

if args.norm:
    # normalize video using vidqa
    with time_task(message_start=f"Running {wblue}vidqa{default} and updating folder modifiation times in {gray}{input_dir}{default}", end="\n"):
        subprocess.run(["vidqa", "-i", input_dir, "-m", "unique", "-fd",
                        Path(Path(getframeinfo(currentframe()).filename).resolve().parent, "vidqa_data")])
        # update folder time structure
        file_utils.update_folder_times(input_dir)

# load whisper model
with time_task(message_start=f"\nLoading " + ("WhisperX" if args.whisperx else "Whisper") + f" model: {wblue}{args.model}{default} on {wblue}{torch_device}{default}", end="\n"):
    if args.whisperx:
        import whisperx
        import whisperx_utils
    
        whisper_model = whisperx.load_model(
            whisper_arch=args.model, device=torch_device, compute_type=compute_type, asr_options={"repetition_penalty": 1, "prompt_reset_on_temperature": 0.5, "no_repeat_ngram_size": 2,})
    else:
        import whisper

        import whisper_utils
        whisper_model = whisper.load_model(
            name=args.model, device=torch_device, in_memory=True)

with time_task(message="⌛ Processing files for"):
    path: Path
    for path in (item for item in sorted(sorted(Path(input_dir).rglob('*'), key=lambda x: x.stat().st_mtime), key=lambda x: len(x.parts)) if item.is_file()):
        rel_path = path.relative_to(input_dir)
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

                    srt_video_dir = Path(srt_out_dir, rel_path.parent)
                    burned_video_dir = Path(burned_out_dir, rel_path.parent)
                    # output video extension will be changed to .mp4
                    srt_video_path = Path(srt_out_dir, rel_path.stem + posfix_extension + ".mp4")
                    burned_video_path = Path(burned_video_dir, rel_path.stem + posfix_extension + ".mp4")
                    subtitle_translated_path = Path(
                        srt_video_dir, rel_path.stem + posfix_extension + f"_{args.lang}.srt")
                    subtitles_path = []

                    if args.input_lang == "auto":
                        # extract audio
                        audio_short_extracted = file_utils.TempFile(
                            None, file_ext=".wav")
                        ffmpeg_utils.extract_short_wav(
                            origin_media_path, audio_short_extracted.getpath())
                        # detect language
                        print("Detecting audio language: ", end='', flush=True)
                        if args.whisperx:
                            audio_language = whisperx_utils.detect_language(
                                whisper_model, audio_short_extracted.getpath())
                        else:
                            audio_language = whisper_utils.detect_language(
                                whisper_model, audio_short_extracted.getpath())
                        print(f"{gray}{audio_language}{default}")

                        audio_short_extracted.destroy()
                    else:
                        audio_language = args.input_lang
                        print(f"Forced input audio language: {gray}{audio_language}{default}")
                    # set path after get transcribed language
                    subtitle_transcribed_path = Path(
                        srt_video_dir, rel_path.stem + posfix_extension + f"_{audio_language}.srt")
                    # create temp file for .srt
                    transcribed_srt_temp = file_utils.TempFile(
                        subtitle_transcribed_path, file_ext=".srt")
                    # skip transcription if transcribed srt for this language is existing (without overwrite neabled) or will not be used in LeGen process
                    if (file_utils.file_is_valid(subtitle_transcribed_path)) or ((args.disable_burn or file_utils.file_is_valid(burned_video_path)) and (args.disable_srt or file_utils.file_is_valid(subtitle_transcribed_path))) and not args.overwrite:
                        print("Transcription is unnecessary. Skipping.")
                    else:
                        # extract audio
                        audio_extracted = file_utils.TempFile(None, file_ext=".wav")
                        ffmpeg_utils.extract_audio_wav(
                            origin_media_path, audio_extracted.getpath())
                        # transcribe saving subtitles to temp .srt file
                        if args.whisperx:
                            print(f"{wblue}Transcribing{default} with {gray}WhisperX{default}")
                            whisperx_utils.transcribe_audio(
                                whisper_model, audio_extracted.getpath(), transcribed_srt_temp.getpath(), audio_language, device=torch_device, batch_size=args.batch_size)
                        else:
                            print(f"{wblue}Transcribing{default} with {gray}Whisper{default}")
                            whisper_utils.transcribe_audio(
                                model=whisper_model, audio_path=audio_extracted.getpath(), srt_path=transcribed_srt_temp.getpath(), lang=audio_language, disable_fp16=False if compute_type == "float16" or compute_type == "fp16" else True)

                        audio_extracted.destroy()
                        # if save .srt is enabled, save it to destination dir, also update path with language code
                        if not args.disable_srt:
                            transcribed_srt_temp.save()
                    subtitles_path.append(transcribed_srt_temp.getvalidpath())
                    # translate transcribed subtitle using Google Translate if transcribed language is not equals to target
                    # skip translation if translation has equal source and output language, if file is existing (without overwrite neabled) or will not be used in LeGen process
                    if args.lang == audio_language:
                        print("Translation is unnecessary because input and output language are the same. Skipping.")
                    elif (args.disable_burn or file_utils.file_is_valid(burned_video_path)) and (args.disable_srt or file_utils.file_is_valid(subtitle_translated_path)) and not args.overwrite:
                        print("Translation is unnecessary. Skipping.")
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
                            transcribed_srt_temp.getvalidpath(), translated_srt_temp.getpath(), args.lang)
                        if not args.disable_srt:
                            translated_srt_temp.save()

                        subtitles_path.insert(0, translated_srt_temp.getvalidpath())
                    if not args.disable_srt and not args.only_srt_subtitles and not args.disable_embed:
                        if file_utils.file_is_valid(srt_video_path) and not args.overwrite:
                            print(f"Existing video file {gray}{srt_video_path}{default}. Skipping subtitle insert")
                        else:
                            # create the temp .mp4 with srt in video container
                            video_srt_temp = file_utils.TempFile(
                                srt_video_path, file_ext=".mp4")

                            # insert subtitle into container using ffmpeg
                            print(f"{wblue}Inserting subtitle{default} in mp4 container using {gray}FFmpeg{default}")
                            ffmpeg_utils.insert_subtitle(input_media_path=origin_media_path, subtitles_path=subtitles_path,
                                                        burn_subtitles=False, output_video_path=video_srt_temp.getpath(),
                                                        video_codec=args.video_codec, audio_codec=args.audio_codec, sub_style=args.sub_style.replace("'", ""), sub_align=args.sub_align)
                            video_srt_temp.save()
                    if not args.disable_burn and not args.only_srt_subtitles:
                        if file_utils.file_is_valid(burned_video_path) and not args.overwrite:
                            print(f"Existing video file {gray}{burned_video_path}{default}. Skipping subtitle burn")
                        else:
                            # create the temp .mp4 with srt in video container
                            video_burned_temp = file_utils.TempFile(
                                burned_video_path, file_ext=".mp4")
                            # insert subtitle into container and burn using ffmpeg
                            print(f"{wblue}Inserting subtitle{default} in mp4 container and {wblue}burning{default} using {gray}FFmpeg{default}")
                            ffmpeg_utils.insert_subtitle(input_media_path=origin_media_path, subtitles_path=subtitles_path,
                                                        burn_subtitles=True, output_video_path=video_burned_temp.getpath(),
                                                        video_codec=args.video_codec, audio_codec=args.audio_codec, sub_style=args.sub_style.replace("'", ""), sub_align=args.sub_align)
                            video_burned_temp.save()
                else:
                    print("not a video file")
                    if not args.only_video:
                        if not args.disable_srt:
                            # copia o arquivo extra para pasta que contém também os arquivos srt
                            file_utils.copy_file_if_different(path, Path(
                                srt_out_dir, rel_path))
                        if not args.disable_burn:
                            # copia o arquivo extra para pasta que contém os videos queimados
                            file_utils.copy_file_if_different(path, Path(
                                burned_out_dir, rel_path))
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
