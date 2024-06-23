# LeGen

![legen-wide](https://github.com/matheusbach/legen/assets/35426162/05a7acd2-52d5-43e0-8f31-7da7d6aa7c3c)


LeGen is a Python script that uses Whisper/WhisperX AI to locally transcribes speech from media files, generating subtitle files, can translates the generated subtitles, inserts them into the mp4 container, and burns them directly into video

This is very useful for making it available in another language, or even just subtitling any video that belongs to you or that you have the proper authorization to do so, be it a film, lecture, course, presentation, interview, etc.

## Run on Colab

LeGen works on Google Colab, using their computing power to do the work. Aceess the link to [run on Google Colab](https://colab.research.google.com/github/matheusbach/legen/blob/main/legen.ipynb)

 <a href='https://colab.research.google.com/github/matheusbach/legen/blob/main/legen.ipynb' style='padding-left: 0.5rem;'><img src='https://colab.research.google.com/assets/colab-badge.svg' alt='Google Colab'></a>

## Install locally:

Install FFMpeg from [FFMPeg Oficial Site](https://ffmpeg.org/download.html) or from your linux package manager. _If using windows, prefer gyan_dev release full `choco install ffmpeg-full`_

Install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

Install [Python](https://www.python.org/downloads/) 3.8 or up. _If using windows, select "Add to PATH" option when installing_

Clone LeGen using git
```sh
git clone https://github.com/matheusbach/legen.git
cd legen
```

Install requirements using pip. Is recommended to create a virtual environment (venv) as a good practice
```sh
pip3 install -r requirements.txt --upgrade
```

### GPU compatibility

If having troubles with GPU compatibility, get [PyTorch](https://pytorch.org/get-started/locally/) for your GPU.

_And done. Now you can use LeGen_

### Update

For dry-run update, use in legen folder:
```sh
git fetch && git reset --hard origin/main && git pull
pip3 install -r requirements.txt --upgrade --force-reinstall
```

## Run locally:

To use LeGen, run the following command:

The minimum comand line is:

```sh
python3 legen.py -i [input_path]
```

Users could for example also translate generated subtitles for other language like portuguese (pt) adding `--translate pt` to the command line


Full options list are described bellow:

- `-i`, `--input_path`: Specifies the path to the media files. This can be a folder containing files or an individual file. Example: `LeGen -i /path/to/media/files`.

- `--norm`: Normalizes folder times and runs vidqa on the input path before starting to process files. Useful for synchronizing timestamps across multiple media files.

- `-ts:e`, `--transcription_engine`: Specifies the transcription engine to use. Possible values are "whisperx" and "whisper". Default is "whisperx".

- `-ts:m`, `--transcription_model`: Specifies the path or name of the Whisper transcription model. A larger model will consume more resources and be slower, but with better transcription quality. Possible values: tiny, base, small, medium (default), large, ...

- `-ts:d`, `--transcription_device`: Specifies the device to run the transcription through Whisper. Possible values: auto (default), cpu, cuda.

- `-ts:c`, `--transcription_compute_type`: Specifies the quantization for the neural network. Possible values: auto (default), int8, int8_float32, int8_float16, int8_bfloat16, int16, float16, bfloat16, float32.

- `-ts:b`, `--transcription_batch`: Specifies the number of simultaneous segments being transcribed. Higher values will speed up processing. If you have low RAM/VRAM, long duration media files or have buggy subtitles, reduce this value to avoid issues. Only works using transcription_engine whisperx. Default is 4.

- `--translate`: Translates subtitles to a language code if they are not the same as the original. The language code should be specified after the equals sign. For example, `LeGen --translate=fr` would translate the subtitles to French.

- `--input_lang`: Indicates (forces) the language of the voice in the input media. Default is "auto".

- `-c:v`, `--codec_video`: Specifies the target video codec. Can be used to set acceleration via GPU or another video API [codec_api], if supported (ffmpeg -encoders). Examples include h264, libx264, h264_vaapi, h264_nvenc, hevc, libx265 hevc_vaapi, hevc_nvenc, hevc_cuvid, hevc_qsv, hevc_amf. Default is h264.

- `-c:a`, `--codec_audio`: Specifies the target audio codec. Default is aac. Examples include aac, libopus, mp3, vorbis.

- `-o:s`, `--output_softsubs`: Specifies the path to the folder or output file for the video files with embedded softsub (embedded in the mp4 container and .srt files). Default is "softsubs_" followed by the input path.

- `-o:h`, `--output_hardsubs`: Specifies the output folder path for video files with burned-in captions and embedded in the mp4 container. Default is "hardsubs_" followed by the input path.

- `--overwrite`: Overwrites existing files in output directories. By default, this option is false.

- `--disable_srt`: Disables .srt file generation and doesn't insert subtitles in the mp4 container of output_softsubs. By default, this option is false.

- `--disable_softsubs`: Doesn't insert subtitles in the mp4 container of output_softsubs. This option continues generating .srt files. By default, this option is false.

- `--disable_hardsubs`: Disables subtitle burn in output_hardsubs. By default, this option is false.

- `--copy_files`: Copies other (non-video) files present in the input directory to output directories. Only generates the subtitles and videos. By default, this option is false.

Each of these options provides control over various aspects of the video processing workflow. Make sure to refer to the documentation or help message (`LeGen --help`) for more details on each option[Source 0](https://docs.python.org/3/library/argparse.html)[Source 2](https://realpython.com/command-line-interfaces-python-argparse/).

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
- matheusbach/whisperx (fork from m-bain/whisperx)

This dependencies can be installed and updated with ```pip install -r requirements.txt --upgrade```

You also need to [install FFmpeg](https://ffmpeg.org/download.html)

## Contributing

Contributions are welcome. Submit your pull request ❤️

## Issues, Doubts

Not being able to use the software, or encountering an error? open an [issue](https://github.com/matheusbach/legen/issues/new)

## Telegram Group

Welcome and don't be a sick. We are brazilian, but you can write in other language if you want. https://t.me/+c0VRonlcd9Q2YTAx

## Video Tutorials

[PT-BR] [SEMI-OUTDATED] [**Tutorial - LeGen no Google Colab**](https://odysee.com/@legen_software:d/legen_no_colab:0)

## Donations

You can donate to project using:
Monero (XMR): ```86HjTCsiaELEoNhH96rTf3ezGMXgKmHjqFrNmca2tesCESdCTZvRvQ9QWQXPGDtmaZhKz4ryHCdZXFzdbmtGahVa5VMLJnx```
LivePix: https://livepix.gg/legendonate

### Donators
- Picasso Neves
- Erasmo de Souza Mora
- viniciuspro
- Igor
- NiNi
- PopularC


## License

This project is licensed under the terms of the [GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/).
