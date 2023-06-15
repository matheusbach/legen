import argparse
import os
import subprocess
import time

import ffmpeg_utils
import file_utils
import translate_utils

version = "v0.7.1"

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

{version} - github.com/matheusbach/legen{default}
""")
time.sleep(1.5)

# define parâmetros e configuraçṍes
parser = argparse.ArgumentParser(prog="LeGen", description="Normaliza arquivos de vídeo, transcreve legendas a partir do áudio de arquivos de vídeo e áudio, traduz as legendas geradas, salva as legendas em arquivos .srt, insere no container mp4 e queima diretamente em vídeo",
                                 argument_default=True, allow_abbrev=True, add_help=True)
parser.add_argument("-i", "--input_dir", type=str,
                    help="Caminho da pasta onde os vídeos e/ou audios originais estão localizados.", required=True)
parser.add_argument("--use_vidqa", default=False, action="store_true",
                    help="Run vidqa in input folder before start LeGen processing.")
parser.add_argument("--whisperx", default=False, action="store_true",
                    help="Use m-bain/whisperX instead openai/whisper")
parser.add_argument("--model", type=str, default="medium",
                    help="Caminho ou nome do modelo de transcrição Whisper. (default: medium)")
parser.add_argument("--dev", type=str, default="auto",
                    help="Dispositivo para rodar a transcrição pelo Whisper. [cpu, cuda, auto]. (default: auto)")
parser.add_argument("--lang", type=str, default="pt",
                    help="Idioma para o qual as legendas devem ser traduzidas. Language equals to source video skip translation (default: pt)")
parser.add_argument("--input_lang", type=str, default="auto",
                    help="Indica (força) idioma da voz das midias de entrada (default: auto)")
parser.add_argument("--crf", type=int, default=20,
                    help="Valor CRF a ser usado no vídeo. (default: 20)")
parser.add_argument("--maxrate", type=str, default="2M",
                    help="Maxrate a ser usado no vídeo. (default: 2M)")
parser.add_argument("-c:v", "--video_codec", type=str, default="h264",
                    help="Codec de vídeo destino. Pode ser usado para definir aceleração via GPU ou outra API de video [codec_api], se suportado (ffmpeg -encoders). Ex: h264, libx264, h264_vaapi, h264_nvenc, hevc, libx265 hevc_vaapi, hevc_nvenc, hevc_cuvid, hevc_qsv, hevc_amf (default: h264)")
parser.add_argument("-c:a", "--audio_codec", type=str, default="aac",
                    help="Codec de audio destino. (default: aac). Ex: aac, libopus, mp3, vorbis")
parser.add_argument("--preset", type=str, default=None,
                    help="ffmpeg codec preset. (default: auto / default of current codec). Ex: ultrafast, veryfast, fast, medium, slow, slower, veryslow")
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
args = parser.parse_args()

input_dir = args.input_dir
if args.srt_out_dir is None:
    args.srt_out_dir = os.path.join(
        *(os.path.split(input_dir)[:-1] + (f'legen_srt_{os.path.split(input_dir)[-1]}',)))
srt_out_dir = args.srt_out_dir
if args.burned_out_dir is None:
    args.burned_out_dir = os.path.join(
        *(os.path.split(input_dir)[:-1] + (f'legen_burned_{args.lang}_{os.path.split(input_dir)[-1]}',)))
burned_out_dir = args.burned_out_dir

if args.dev == "auto":
    import torch
    torch_device = ("cuda" if torch.cuda.is_available() else "cpu")
else:
    torch_device = str.lower(args.dev)
    
disable_fp16 = True if torch_device == "cpu" else False

if args.only_srt_subtitles:
    args.only_video = True

args.model = "large-v2" if args.model == "large" else args.model

# ----------------------------------------------------------------------------

# normalize video using vidqa
if args.use_vidqa:
    print(f"Running {wblue}vidqa{default} in {gray}{input_dir}{default}")
    subprocess.run(["vidqa", "-i", input_dir, "-m", "unique", "-fd",
                os.path.join(os.path.realpath(os.path.dirname(__file__)), "vidqa_data")])

# load whisper model
print(f"\nLoading Whisper model: {wblue}{args.model}{default} on {wblue}{torch_device}{default}")
if args.whisperx:
    import whisperx_utils
    import whisperx
    whisper_model = whisperx.load_model(whisper_arch=args.model, device=torch_device, compute_type="float16" if not disable_fp16 else "float32")
else:
    import whisper_utils
    import whisper
    whisper_model = whisper.load_model(name=args.model, device=torch_device, in_memory=True)

for dirpath, dirnames, filenames in os.walk(input_dir):
    for filename in sorted(filenames):
        try:
            rel_path = os.path.relpath(dirpath, input_dir)
            rel_path = rel_path if rel_path != '.' else ''

            print(f"\nProcessing {yellow}{os.path.join(rel_path, filename)}{default}")
            
            # define file type by extensions
            if filename.lower().endswith((".mp4", ".webm", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".vob", ".mts", ".m2ts", ".ts", ".yuv", ".mpg", ".mp2", ".mpeg", ".mpe", ".mpv", ".m2v", ".m4v", ".3gp", ".3g2", ".nsv")):
                file_type = "video" 
            elif filename.lower().endswith((".aa", ".aac", ".aax", ".act", ".aiff", ".alac", ".amr", ".ape", ".au", ".awb", ".dss", ".dvf", ".flac", ".gsm", ".iklax", ".ivs", ".m4a", ".m4b", ".m4p", ".mmf", ".mp3", ".mpc", ".msv", ".nmf", ".ogg", ".oga", ".mogg", ".opus", ".ra", ".rm", ".raw", ".rf64", ".sln", ".tta", ".voc", ".vox", ".wav", ".wma", ".wv", ".webm", ".8svx")):
                file_type = "audio"
            else:
                file_type = "other"
                
            if file_type == "video" or file_type == "audio":
                # define paths
                origin_media_path = os.path.join(dirpath, filename)
                srt_video_dir = os.path.join(srt_out_dir, rel_path)
                burned_video_dir = os.path.join(burned_out_dir, rel_path)
                # output video extension will be changed to .mp4
                srt_video_path = os.path.join(srt_video_dir, os.path.splitext(filename)[0] + ".mp4")
                burned_video_path = os.path.join(burned_video_dir, os.path.splitext(filename)[0] + ".mp4")
                subtitle_translated_path = os.path.join(
                    srt_video_dir, f"{os.path.splitext(filename)[0]}_{args.lang}.srt")
                subtitles_path = []
                
                if args.input_lang == "auto":
                    # extract audio
                    audio_short_extracted = file_utils.TempFile(None, file_ext=".mp3")
                    ffmpeg_utils.extract_short_mp3(
                        origin_media_path, audio_short_extracted.getname())

                    # detect language
                    print("Detecting audio language: ", end='', flush=True)
                    if args.whisperx:
                        audio_language = whisperx_utils.detect_language(
                            whisper_model, audio_short_extracted.getname())
                    else:
                        audio_language = whisper_utils.detect_language(
                            whisper_model, audio_short_extracted.getname())
                    print(f"{gray}{audio_language}{default}")
                    
                    audio_short_extracted.destroy()
                else:
                    audio_language = args.input_lang
                    print(f"Forced input audio language: {gray}{audio_language}{default}")

                # set path after get transcribed language
                subtitle_transcribed_path = os.path.join(
                    srt_video_dir, f"{os.path.splitext(filename)[0]}_{audio_language}.srt")

                # create temp file for .srt
                transcribed_srt_temp = file_utils.TempFile(
                    subtitle_transcribed_path, file_ext=".srt")

                # skip transcription if transcribed srt for this language is existing (without overwrite neabled) or will not be used in LeGen process
                if (file_utils.file_is_valid(subtitle_transcribed_path)) or ((args.disable_burn or file_utils.file_is_valid(burned_video_path)) and (args.disable_srt or file_utils.file_is_valid(subtitle_transcribed_path))) and not args.overwrite:
                    print("Transcription is unecessary. Skipping.")
                else:
                    # extract audio
                    audio_extracted = file_utils.TempFile(None, file_ext=".mp3")
                    ffmpeg_utils.extract_audio_mp3(
                        origin_media_path, audio_extracted.getname())

                    # transcribe saving subtitles to temp .srt file
                    if args.whisperx:
                        print(f"{wblue}Transcribing{default} with {gray}WhisperX{default}")
                        whisperx_utils.transcribe_audio(
                            whisper_model, audio_extracted.getname(), transcribed_srt_temp.getname(), audio_language, disable_fp16, device = torch_device)
                    else:
                        print(f"{wblue}Transcribing{default} with {gray}Whisper{default}")
                        whisper_utils.transcribe_audio(
                            whisper_model, audio_extracted.getname(), transcribed_srt_temp.getname(), audio_language, disable_fp16)
                        
                    audio_extracted.destroy()

                    # if save .srt is enabled, save it to destination dir, also update path with language code
                    if not args.disable_srt:
                        transcribed_srt_temp.save()

                subtitles_path.append(transcribed_srt_temp.getvalidname())

                # translate transcribed subtitle using Google Translate if transcribed language is not equals to target
                # skip translation if translation has equal source and output language, if file is existing (without overwrite neabled) or will not be used in LeGen process
                if args.lang == audio_language:
                    print("Translation is unecessary because input and output language are the same. Skipping.")
                elif (args.disable_burn or file_utils.file_is_valid(burned_video_path)) and (args.disable_srt or file_utils.file_is_valid(subtitle_translated_path)) and not args.overwrite:
                    print("Translation is unecessary. Skipping.")
                elif file_utils.file_is_valid(subtitle_translated_path):
                    print("Translated file found. Skipping translation.")
                    subtitles_path.insert(0, subtitle_translated_path)
                else:
                    # create the temp .srt translated file
                    translated_srt_temp = file_utils.TempFile(
                        subtitle_translated_path, file_ext=".srt")
                    
                    # translating with google translate public API
                    print(f"{wblue}Translating{default} with {gray}Google Translate{default}")
                    subs = translate_utils.translate_srt_file(
                        transcribed_srt_temp.getvalidname(), translated_srt_temp.getname(), args.lang)
                    if not args.disable_srt:
                        translated_srt_temp.save()
                            
                    subtitles_path.insert(0, translated_srt_temp.getvalidname())

                if not args.disable_srt and not args.only_srt_subtitles and not args.disable_embed:
                    if file_utils.file_is_valid(srt_video_path) and not args.overwrite:
                        print(f"Existing video file {gray}{srt_video_path}{default}. Skipping subtitle insert")
                    else:
                        # create the temp .mp4 with srt in video container
                        video_srt_temp = file_utils.TempFile(
                            srt_video_path, file_ext=".mp4")
                        
                        # insert subtitle into container using ffmpeg
                        print(f"{wblue}Inserting subtitle{default} in mp4 container using {gray}FFmpeg{default}")
                        ffmpeg_utils.insert_subtitle(origin_media_path, subtitles_path,
                                                     False, video_srt_temp.getname(), 
                                                     args.crf, args.maxrate, args.video_codec, args.audio_codec, args.preset)
                        video_srt_temp.save()

                if not args.disable_burn and not args.only_srt_subtitles:
                    if file_utils.file_is_valid(burned_video_path) and not args.overwrite:
                        print(
                            f"Existing video file {gray}{burned_video_path}{default}. Skipping subtitle burn")
                    else:
                        # create the temp .mp4 with srt in video container
                        video_burned_temp = file_utils.TempFile(
                            burned_video_path, file_ext=".mp4")
                        # insert subtitle into container and burn using ffmpeg
                        print(f"{wblue}Inserting subtitle{default} in mp4 container and {wblue}burning{default} using {gray}FFmpeg{default}")
                        ffmpeg_utils.insert_subtitle(origin_media_path, subtitles_path,
                                                     True, video_burned_temp.getname(),
                                                     args.crf, args.maxrate, args.video_codec, args.audio_codec, args.preset)
                        video_burned_temp.save()
            else:
                print("not a video file")
                if not args.only_video:
                    if not args.disable_srt:
                        # copia o arquivo extra para pasta que contém também os arquivos srt
                        file_utils.copy_file_if_different(os.path.join(dirpath, filename), os.path.join(
                            srt_out_dir, rel_path, filename))
                    if not args.disable_burn:
                        # copia o arquivo extra para pasta que contém os videos queimados
                        file_utils.copy_file_if_different(os.path.join(dirpath, filename), os.path.join(
                            burned_out_dir, rel_path, filename))
        except Exception as e:
            file = os.path.join(dirpath, filename)

            print(f"{red}ERROR !!!{default} {file}")
            print(f"{yellow}check legen-errors.txt for details{default}")
            # extract the relevant information from the exception object
            current_time = time.strftime("%y/%m/%d %H:%M:%S", time.localtime())
            
            error_message = f"[{current_time}] {file}: {type(e).__name__}: {str(e)}"

            # write the error message to a file
            with open(os.path.join(os.path.realpath(os.path.dirname(__file__)), "legen-errors.txt"), "a") as f:
                f.write(error_message + "\n")
                f.close()

print("Deleting temp folder")
file_utils.delete_folder(os.path.join(
    os.path.realpath(os.path.dirname(__file__)), "temp"))

print(f"{green}Processamento concluído!{default}")
