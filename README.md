# LeGen

LeGen is a Python script that normalizes video files using vidqa, transcribes subtitles from audio with Whisper, translates generated subtitles using Google Translator, saves subtitles in .srt files, inserts subtitles into .mp4 container and burns subtitles directly onto videos using FFmpeg.

This is very useful for making it available in another language, or even just subtitling any video that belongs to you or that you have the proper authorization to do so, be it a film, lecture, course, presentation, interview, etc.

## Usage:

To use LeGen, run the following command:

```sh
python legen.py -i [input_dir] --model [model_name] --dev [device] --lang [language_code] --crf [crf_value] --maxrate [maxrate_value] --srt_out_dir [output_dir_for_srt_files] --burned_out_dir [output_dir_for_burned_files] --overwrite --disable_srt --disable_burn --only_video
```

The available arguments are:

-    **-i/--input_dir**: Path to the folder containing the original videos (required).
-    **--model**: Path or name of the Whisper transcription model to use (default: "medium").
-    **--dev**: Device to use for the Whisper transcription (options: "cpu", "cuda", "auto"; default: "auto").
-    **--lang**: Language code to use for the subtitles translation (default: "pt").
-    **--crf**: CRF value to use for the output video (default: 20).
-    **--maxrate**: Maxrate value to use for the output video (default: "2M").
-    **-c:v/--video_codec**: Output video codec. Can also be used to define hardware API. Check supported using [ffmpeg -codecs]. Ex: h264_vaapi, h264_nvenc, hevc, hevc_vaapi (default: h264)
-    **-c:a/--audio_codec**: Output audio codec. Check supported using [ffmpeg -codecs] (default: aac)
-    **--preset**: ffmpeg codec preset. (default: auto / default of current codec). Ex: ultrafast, veryfast, fast, medium, slow, slower, veryslow
-    **--srt_out_dir**: The output folder path for the video files with subtitles embedded in the MP4 container and SRT files. (default: legen_srt_[input_dir]).
-    **--burned_out_dir**: The output folder path for the video files with burned subtitles and embedded in the MP4 container. (default: legen_burned_[lang]_[input_dir]).
-    **--overwrite**: Overwrite existing files in output directories.
-    **--disable_srt**: Disable .srt file generation and do not insert subtitles in the MP4 container of $srt_out_dir.
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
- tqdm
- whisper
- vidqa

This dependencies can be installed with ```pip install -re requirements.txt```

You also need to [install FFmpeg](https://ffmpeg.org/download.html)

## Contributing

Every contribution is welcome. Submit your pull request ❤️

## Issues, Doubts

Not being able to use the software, or encountering an error? open an [issue](https://github.com/matheusbach/legen/issues/new)

## Donation
Monero (XMR): ```86HjTCsiaELEoNhH96rTf3ezGMXgKmHjqFrNmca2tesCESdCTZvRvQ9QWQXPGDtmaZhKz4ryHCdZXFzdbmtGahVa5VMLJnx```

## License

This project is licensed under the terms of the [GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/).
