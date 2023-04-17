# LeGen

LeGen is a Python script that normalizes video files using vidqa, transcribes subtitles from audio with Whisper, translates generated subtitles using Google Translator, saves subtitles in .srt files, inserts subtitles into .mp4 container and burns subtitles directly onto videos using FFmpeg.

This is very useful for making it available in another language, or even just subtitling any video that belongs to you or that you have the proper authorization to do so, be it a film, lecture, course, presentation, interview, etc.

## Usage:

To use LeGen, run the following command:
```python LeGen.py -i [input_dir] --model [model_name] --dev [device] --lang [language_code] --crf [crf_value] --maxrate [maxrate_value] --srt_out_dir [output_dir_for_srt_files] --burned_out_dir [output_dir_for_burned_files] --overwrite --disable_srt --disable_burn --only_video```

The available arguments are:

-    **-i/--input_dir**: path to the folder containing the original videos (required).
-    **--model**: path or name of the Whisper transcription model to use (default: "base").
-    **--dev**: device to use for the Whisper transcription (options: "cpu", "cuda", "auto"; default: "auto").
-    **--lang**: language code to use for the subtitles translation (default: "pt").
-    **--crf**: CRF value to use for the output video (default: 20).
-    **--maxrate**: maxrate value to use for the output video (default: "2M").
-    **--srt_out_dir**: path to the output folder for the videos with subtitles embedded in an mp4 container and SRT files (default: legen_srt_[input_dir]).
-    **--burned_out_dir**: path to the output folder for the videos with burned-in subtitles and subtitles embedded in an mp4 container (default: legen_burned_[lang]_[input_dir]).
-    **--overwrite**: overwrite existing files in the output directories.
-    **--disable_srt**: disable SRT file generation and don't insert subtitles in the mp4 container in the srt_out_dir.
-    **--disable_burn**: disable subtitle burn in the burned_out_dir.
-    **--only_video**: don't copy other files present in the input directory to the output directories.

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

## License

This project is licensed under the terms of the [GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/).
