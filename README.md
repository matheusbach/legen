# LeGen

LeGen is a Python script that normalizes media files using vidqa, transcribes subtitles from audio or video-audio with Whisper, translates generated subtitles using Google Translator, saves subtitles in .srt files, inserts subtitles into .mp4 container and burns subtitles directly onto videos using FFmpeg.

This is very useful for making it available in another language, or even just subtitling any video that belongs to you or that you have the proper authorization to do so, be it a film, lecture, course, presentation, interview, etc.

## Installation:

Install FFMpeg from [FFMPeg Oficial Site](https://ffmpeg.org/download.html) or from your linux package manager. _If using windows, prefer gyan_dev release full_

Install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

Install [Python](https://www.python.org/downloads/) 3.8 or up. _If using windows, select "Add to PATH" option when installing_

Clone LeGen from git
```sh
git clone https://github.com/matheusbach/legen.git
cd legen
pip install -r requirements.txt --upgrade
```
And done. Now you can use LeGen

## Update

Update from git:
in LeGen folder:
```sh
git pull
pip install -r requirements.txt --upgrade
```

### Or [run on Google Colab](https://colab.research.google.com/github/matheusbach/legen/blob/main/legen.ipynb)
 <a href='https://colab.research.google.com/github/matheusbach/legen/blob/main/legen.ipynb' style='padding-left: 0.5rem;'><img src='https://colab.research.google.com/assets/colab-badge.svg' alt='Google Colab'></a>

## Usage:

To use LeGen, run the following command:

```sh
python legen.py -i [input_dir] {other optional args here}
```

The available arguments are:

-    **-i/--input_dir**: Path to the folder containing the original videos and/or audios (required).
-    **--norm**: Run vidqa and update folder times in input folder before start LeGen processing.
-    **--whisperx**: Use m-bain/whisperX implementation instead of openai/whisper. Unstable!
-    **--model**: Path or name of the Whisper transcription model to use (default: "medium").
-    **--dev**: Device to use for the Whisper transcription (options: "cpu", "cuda", "auto"; default: "auto").
-    **--compute_type**: Quantization for the neural network. Ex: float32, float16, int16, int8, ... (default will use float16 for GPU and float32 for CPU).
-    **--batch_size**: The higher the value, the faster the processing will be. If you have low RAM or have buggy subtitles, reduce this value. Works only using whisperX. (default: 4).
-    **--lang**: Language code to use for the subtitles translation (default: "pt").
-    **-c:v/--video_codec**: Output video codec. Can also be used to define hardware aceleration API. Check supported using [ffmpeg --encoders]. Ex: h264, libx264, h264_vaapi, h264_nvenc, hevc, libx265 hevc_vaapi, hevc_nvenc, hevc_cuvid, hevc_qsv, hevc_amf (default: h264).
-    **-c:a/--audio_codec**: Output audio codec. Check supported using [ffmpeg --encoders] (default: aac).
-    **--srt_out_dir**: The output folder path for the video files with subtitles embedded in the MP4 container and SRT files. (default: legen_srt_[input_dir]).
-    **--burned_out_dir**: The output folder path for the video files with burned subtitles and embedded in the MP4 container. (default: legen_burned_[lang]_[input_dir]).
-    **--overwrite**: Overwrite existing files in output directories.
-    **--disable_srt**: Disable .srt file generation and do not insert subtitles in the MP4 container of $srt_out_dir.
-    **--disable_embed**: Don't insert subtitles in mp4 container of $srt_out_dir. This option continue generating .srt files.
-    **--disable_burn**: Disable subtitle burn in $burned_out_dir.
-    **--only_video**: Do not copy other files present in the input directory to the output directories. Only generate the subtitles and videos.
-    **--only_srt_subtitles**: Only generates the subtitles. Do not encode the videos or copy other files.

## Dependencies

LeGen requires the following **pip** dependencies to be installed:
- deep_translator
- ffmpeg_progress_yield
- openai_whisper
- pysrt
- torch
- tqdm
- whisper
- vidqa
- m-bain/whisperx

This dependencies can be installed and updated with ```pip install -r requirements.txt --upgrade```

You also need to [install FFmpeg](https://ffmpeg.org/download.html)

## Contributing

Every contribution is welcome. Submit your pull request ❤️

## Issues, Doubts

Not being able to use the software, or encountering an error? open an [issue](https://github.com/matheusbach/legen/issues/new)

## Donation
Monero (XMR): ```86HjTCsiaELEoNhH96rTf3ezGMXgKmHjqFrNmca2tesCESdCTZvRvQ9QWQXPGDtmaZhKz4ryHCdZXFzdbmtGahVa5VMLJnx```

## License

This project is licensed under the terms of the [GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/).
