# LeGen

![GitHub stars](https://img.shields.io/github/stars/matheusbach/legen?style=flat-square)
![GitHub forks](https://img.shields.io/github/forks/matheusbach/legen?style=flat-square)
![GitHub watchers](https://img.shields.io/github/watchers/matheusbach/legen?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/matheusbach/legen?style=flat-square)
![GitHub license](https://img.shields.io/github/license/matheusbach/legen?style=flat-square)
![GitHub last commit](https://img.shields.io/github/last-commit/matheusbach/legen?style=flat-square)
![Release](https://img.shields.io/github/v/release/matheusbach/legen?style=flat-square)
![PyPI downloads total](https://pepy.tech/badge/legen)
![PyPI downloads last month](https://pepy.tech/badge/legen/month)


![legen-wide](https://github.com/matheusbach/legen/assets/35426162/05a7acd2-52d5-43e0-8f31-7da7d6aa7c3c)


LeGen is a fast, AI-powered subtitle studio that runs right on your machine. It taps into Whisper and WhisperX to transcribe speech, translates the results into the language you need, then exports polished `.srt`/`.txt` files, muxes them into MP4 containers, or even burns them straight into the video. LeGen also speaks fluent `yt-dlp`, pulling remote videos or playlists and embedding every subtitle track it can find before the pipeline kicks in.

This is very useful for making it available in another language, or even just subtitling any video that belongs to you or that you have the proper authorization to do so, be it a film, lecture, course, presentation, interview, etc.

## Run on Colab

LeGen works on Google Colab, using their computing power to do the work. Aceess the link to [run on Google Colab](https://colab.research.google.com/github/matheusbach/legen/blob/main/legen.ipynb)

 <a href='https://colab.research.google.com/github/matheusbach/legen/blob/main/legen.ipynb' style='padding-left: 0.5rem;'><img src='https://colab.research.google.com/assets/colab-badge.svg' alt='Google Colab'></a>

## Install

### Using uv (recommended)

Install uv by following the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/). Once uv is available, install the latest LeGen release from PyPI with:

```sh
uv tool install legen
```

This command downloads the published wheel via uv's pip-compatible resolver and creates an isolated environment that exposes a `legen` launcher on your PATH. Keep FFmpeg installed on the host so the CLI can access it (see "From source" below for platform-specific tips).

To update an existing installation to the newest version, run:

```sh
uv tool upgrade legen
```

Run the CLI just like any other command-line tool:

```sh
legen -i /path/to/video.mp4
```

If your shell cannot find the command, ensure uv's tool shims directory (usually `~/.local/bin`) is on your PATH, or invoke the tool through uv directly with `uv tool run legen -i /path/to/video.mp4`.

### From PyPI

Install the published package directly from [PyPI](https://pypi.org/project/legen/):

```sh
pip install legen
```

The `legen` console script will be added to your PATH and mirrors all CLI options documented below.

### From source (pip)

Install FFMpeg from [FFMPeg Oficial Site](https://ffmpeg.org/download.html) or from your linux package manager. _If using windows, prefer gyan_dev release full `choco install ffmpeg-full`_

Install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

Install [Python](https://www.python.org/downloads/) Recomended version: 3.12.x (LeGen currently supports CPython 3.9 up to 3.12). _If using windows, select "Add to PATH" option when installing_

Clone LeGen using git
```sh
git clone https://github.com/matheusbach/legen.git
cd legen
```

Install requirements using pip. Is recommended to create a virtual environment (venv) as a good practice
```sh
pip3 install -r requirements.txt --upgrade
```

Ensure the `yt-dlp` command is available in your shell so LeGen can fetch remote videos. The provided requirements install `yt-dlp` for convenience, and LeGen will use it to embed all subtitle tracks it can find for each item into the MP4 container.

### GPU compatibility

If having troubles with GPU compatibility, get [PyTorch](https://pytorch.org/get-started/locally/) for your GPU.

_And done. Now you can use LeGen_

### Update

If you installed the packaged CLI with uv, use `uv tool upgrade legen` as shown above.

For pip-based environments:
```sh
git fetch && git reset --hard origin/main && git pull
pip3 install -r requirements.txt --upgrade --force-reinstall
```

## Run locally:

To use LeGen, run the following command:

The minimum comand line is:

```sh
legen -i [input_path]
```

If you installed from source without uv, replace the command with:

```sh
python3 legen.py -i [input_path]
```

Users could for example also translate generated subtitles for other language like portuguese (pt) adding `--translate pt` to the command line


Full options list are described bellow:

- `-i`, `--input_path`: Specifies the path to the media files or a direct video/playlist URL. The CLI will download URLs with `yt-dlp` before processing. Example: `LeGen -i /path/to/media/files` or `LeGen -i https://www.youtube.com/watch?v=…`.

- `--process_input_subs` (alias: `--process_srt_inputs`): Also process existing `.srt` subtitle files found in the input path (translate/TLTW). If a subtitle filename matches a media filename in the same folder (e.g. `video.mp4` + `video.srt` or `video_en.srt`), LeGen will use that `.srt` instead of transcribing the audio. Subtitles without a matching media file are processed as standalone inputs (no MP4 output).

- `--norm`: Normalizes folder times and runs vidqa on the input path before starting to process files. Useful for synchronizing timestamps across multiple media files.

- `-ts:e`, `--transcription_engine`: Specifies the transcription engine to use. Possible values are "whisperx" and "whisper". Default is "whisperx".

- `-ts:m`, `--transcription_model`: Specifies the path or name of the Whisper transcription model. A larger model will consume more resources and be slower, but with better transcription quality. Possible values: tiny, base, small, medium, large, large-v3, turbo, large-v3-turbo (default)...

- `-ts:d`, `--transcription_device`: Specifies the device to run the transcription through Whisper. Possible values: auto (default), cpu, cuda.

- `-ts:c`, `--transcription_compute_type`: Specifies the quantization for the neural network. Possible values: auto (default), int8, int8_float32, int8_float16, int8_bfloat16, int16, float16, bfloat16, float32.

- `-ts:v`, `--transcription_vad`: Selects the voice-activity detector used by WhisperX. Options: silero (default), pyannote.

- `-ts:b`, `--transcription_batch`: Specifies the number of simultaneous segments being transcribed. Higher values will speed up processing. If you have low RAM/VRAM, long duration media files or have buggy subtitles, reduce this value to avoid issues. Only works using transcription_engine whisperx. Default is 4.

- `--translate`: Translates subtitles to a language code if they are not the same as the original. The language code should be specified after the equals sign. For example, `LeGen --translate=fr` would translate the subtitles to French.

- `--input_lang`: Indicates (forces) the language of the voice in the input media. Default is "auto".

	When `--process_input_subs` is enabled, a non-`auto` `--input_lang` also forces the assumed source language for input `.srt` files.

- `-c:v`, `--codec_video`: Specifies the target video codec. Can be used to set acceleration via GPU or another video API [codec_api], if supported (ffmpeg -encoders). Examples include h264, libx264, h264_vaapi, h264_nvenc, hevc, libx265 hevc_vaapi, hevc_nvenc, hevc_cuvid, hevc_qsv, hevc_amf. Default is h264.

- `-c:a`, `--codec_audio`: Specifies the target audio codec. Default is aac. Examples include aac, libopus, mp3, vorbis.

- `-o:s`, `--output_softsubs`: Specifies the path to the folder or output file for the video files with embedded softsub (embedded in the mp4 container and .srt files). Default is "softsubs_" followed by the input path.

- `-o:h`, `--output_hardsubs`: Specifies the output folder path for video files with burned-in captions and embedded in the mp4 container. Default is "hardsubs_" followed by the input path.

- `-o:d`, `--output_downloads`: Overrides the folder used to store media downloaded from URL inputs. Default is `./downloads` when `-i` receives a URL.

- `--overwrite`: Overwrites existing files in output directories. By default, this option is false.

- `-dl:rs`, `--download_remote_subs`: When supplied alongside a URL input, instructs `yt-dlp` to download and embed every subtitle track it can find into the downloaded MP4. By default, remote subtitles are not fetched.

- `--subtitle_formats`: Specifies which subtitle formats should be exported. Separate multiple values with comma or space. Supported formats: `srt`, `txt`. Example: `--subtitle_formats srt,txt`.

- `--disable_srt`: Disables .srt file generation and doesn't insert subtitles in the mp4 container of output_softsubs. Equivalent to removing `srt` from `--subtitle_formats`. By default, this option is false.

- `--disable_softsubs`: Doesn't insert subtitles in the mp4 container of output_softsubs. This option continues generating .srt files. By default, this option is false.

- `--disable_hardsubs`: Disables subtitle burn in output_hardsubs. By default, this option is false.

- `--copy_files`: Copies other (non-video) files present in the input directory to output directories. Only generates the subtitles and videos. By default, this option is false.

- `--translate_engine`: Selects the translation engine. Possible values: `google` (default), `gemini`. If you provide `--gemini_api_key` and do not explicitly set `--translate_engine`, LeGen will prefer `gemini` when translation is enabled.
- `--gemini_api_key`: Gemini API key used for translation when `--translate_engine gemini`. Repeat the flag or separate keys with commas/line breaks to supply multiple keys (useful for rotating free-tier quotas). Get your keys at https://aistudio.google.com/apikey
- `--tltw`: Generates a Gemini-powered "Too Long To Watch" summary from the subtitles. Uses translated subtitles when a target language is provided, otherwise the original transcript. Requires `--gemini_api_key`.
- `--output_tltw`: Destination directory for TLTW summaries. Defaults to the softsubs output folder and mirrors the input directory structure.

TLTW output is a Markdown document with `# Title`, `*Tags:*`, `## Key Points`, and a timestamped `## Summary` section (chapter-title style lines like `HH:MM:SS description`).

Each of these options provides control over various aspects of the video processing workflow. Use the help message (`LeGen --help`) for more details.

### Downloading from URLs

When you pass a HTTP(S) URL to `-i`, LeGen will:

- Invoke `yt-dlp` to download the target video, playlist, or batch feed.
- Embed every subtitle track the platform exposes directly into the downloaded media **only when `--download_remote_subs` is provided**.
- Force `mp4` output with the best available video and audio combination.
- Store the media under `./downloads` or the path provided through `--output_downloads`.
- Continue the normal transcription/translation pipeline on the freshly downloaded files with no additional steps from you.

If the value supplied to `-i` is neither a reachable URL nor a valid local file/folder, LeGen will abort with a clear error message so you can correct the input.

## Run with Docker

You can run LeGen inside a container, keeping the host Python environment clean while still persisting downloads and outputs on disk.

1. Build the image with `docker compose build` (or `docker compose pull` once a registry image is available).
2. Place the media you want to process inside `./data` or mount a different host folder to `/data` when invoking Docker.
3. Run LeGen through Compose: `docker compose run --rm legen -i /data/my-video.mp4 --translate pt`. All generated downloads and subtitles stay under the mapped `downloads`, `softsubs_m`, and `hardsubs_m` directories.

To use a GPU-enabled Docker setup, add `--gpus all` to the compose command (Docker Engine 19.03+ with the NVIDIA Container Toolkit). LeGen will detect the GPU automatically, but you can still override it with `--transcription_device` if needed.

### Passing CLI arguments inside Docker
- Compose  command arguments override the default `--help`. Example: `docker compose run --rm legen -i /data/my-video.mp4 --translate pt --download_remote_subs`.
- Provide Gemini keys when translating with Gemini: `docker compose run --rm legen --gemini_api_key YOUR_KEY -i /data/file.mp4 --translate_engine gemini --translate en`.
- Forward environment variables if you prefer: `docker compose run --rm --env GEMINI_API_KEY=YOUR_KEY legen --gemini_api_key "$GEMINI_API_KEY" -i /data/file.mp4`.
- Run the raw image without Compose: `docker run --rm -it -v "$PWD/data:/data" -v "$PWD/downloads:/app/downloads" legen:local -i /data/input.mp4 --disable_hardsubs`.
- Keep the container running interactively for multiple executions by starting a shell: `docker compose run --rm --entrypoint /bin/bash legen`.
- List all CLI options from inside the container: `docker compose run --rm legen --help`.

## GPU acceleration

LeGen automatically selects the best accelerator at runtime (`cuda` > `mps` > `cpu`). When a compatible GPU is available, transcription and alignment transparently run on it; otherwise the pipeline falls back to the CPU. You can still force a specific backend with `--transcription_device`.

With Docker, the image installs the CUDA-enabled PyTorch wheels by default. Expose GPUs to the container using the NVIDIA Container Toolkit (`docker compose run --rm --gpus all legen ...`). If you need a CPU-only image, build with `docker compose build --build-arg PYTORCH_INSTALL_CUDA=false`.

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
- whisperx-legen-fork (legen flavored fork from m-bain/whisperx)
- gemini-srt-translator
- yt-dlp

This dependencies can be installed and updated with ```pip install -r requirements.txt --upgrade```

LeGen requires the `yt-dlp` CLI on your system to download remote content automatically.

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
- brauliobo
- luizc2026
- Waldomiro
- fabiodfmelo
- Soya198
- Ob1iiz
- rdoolfo
- The MDK Trader
- Hastur
- Fábio Delicato
- Lucas9925677
- Rodolfo Pereira


## License

This project is licensed under the terms of the [GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/).
