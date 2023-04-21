import re

from ffmpeg_progress_yield import FfmpegProgress
from tqdm import tqdm

import file_utils


def insert_subtitle(input_video_path: str, subtitles_path: [str], burn_subtitles: bool, output_video_path: str, crf: int = 20, maxrate: str = "2M"):
    # set bufsize as double of maxrate
    bufsize = str(float(re.match(r"([0-9.]+)([a-zA-Z]*)", maxrate).group(1))
                  * 3) + re.match(r"([0-9.]+)([a-zA-Z]*)", maxrate).group(2)
    
    # use only valid srt files
    subtitles_path = file_utils.validate_files(subtitles_path)

    # insert in comand the basics of ffmpeg and input video path
    cmd_ffmpeg = ["ffmpeg", "-y", "-i", "file:" + input_video_path]
    # map input video to video and audio channels
    cmd_ffmpeg_input_map = ["-map", "0:v", "-map", "0:a"]

    # map each subtitle
    for i, subtitle in enumerate(subtitles_path):
        cmd_ffmpeg.extend(["-i", "file:" + subtitle])
        cmd_ffmpeg_input_map.extend(["-map", f"{i+1}:s"])

    # add comand to burn subtitles if its demanded and has at least one valid subtitle in the array. Burn the first one
    if burn_subtitles and len(subtitles_path) > 0:
        # create temp file for .srt
        srt_temp = file_utils.TempFile(
            "", file_ext=".srt")
        
        file_utils.copy_file_if_different(
            subtitles_path[0], srt_temp.temp_file.name, True)
        
        # insert subtitles filter
        cmd_ffmpeg.extend(
            ["-vf", f"subtitles={srt_temp.temp_file.name}:force_style='Fontname=Verdana,PrimaryColour=&H03fcff,Fontsize=18,BackColour=&H80000000,Spacing=0.12,Outline=1,Shadow=1.2'"])

    cmd_ffmpeg.extend(cmd_ffmpeg_input_map)

    # add the remaining parameters and output path
    cmd_ffmpeg.extend(["-c:v", "h264", "-c:a", "aac", "-c:s", "mov_text",
                       "-crf", str(crf), "-maxrate", maxrate, "-bufsize", bufsize,
                       "file:" + output_video_path])

    # run FFmpeg command with a fancy progress bar
    ff = FfmpegProgress(cmd_ffmpeg)
    with tqdm(total=100, position=0, ascii="░▒█", desc="Inserting subtitles" if not burn_subtitles else "Burning subtitles", unit="%", unit_scale=True, leave=True, bar_format="{desc} [{bar}] {percentage:3.0f}% | ETA: {remaining} | {rate_fmt}{postfix}") as pbar:
        for progress in ff.run_command_with_progress():
            pbar.update(progress - pbar.n)
            
    # destroy unecessary file  
    if 'srt_temp' in locals():
        srt_temp.destroy()


def extract_audio_mp3(input_media_path: str, output_path: str):
    # set the FFMpeg command
    cmd_ffmpeg = ["ffmpeg", "-y", "-i", "file:" + input_media_path, "-vn", "-c:a",
                  "mp3", "-ar", "44100", "file:" + output_path]

    # run FFmpeg command with a fancy progress bar
    ff = FfmpegProgress(cmd_ffmpeg)
    with tqdm(total=100, position=0, ascii="░▒█", desc="Extracting audio", unit="%", unit_scale=True, leave=True, bar_format="{desc} {percentage:3.0f}% | ETA: {remaining}") as pbar:
        for progress in ff.run_command_with_progress():
            pbar.update(progress - pbar.n)