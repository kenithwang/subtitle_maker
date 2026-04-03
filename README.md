<div align="center">

# AI Subtitle Maker (CLI)

English | [中文](README_ZH.md)

Gemini-powered subtitle maker for local media files and online videos, with `.srt` and `.ass` export.

</div>

## ✨ Features

- 🎞️ Generate `.srt` or `.ass` subtitles from a **local video/audio file** or an online video URL.
- 🌐 Download online audio with `yt-dlp` and then subtitle it with Gemini.
- 🗣️ Use Gemini to create subtitle-timed output, not note-oriented transcripts.
- 🎤 Default preprocessing can build a same-length dialogue-focused `m4a` track via vocal separation.
- 🌏 Support `bilingual-zh` mode: original dialogue + Simplified Chinese per cue.
- ✂️ Chunk long audio automatically (up to 20-minute subtitle chunks) and merge back into one subtitle file.
- 🎯 Support `--timing align`: use local `faster-whisper` word timestamps to post-align cues and reduce drift.
- 🧰 Works in Linux, macOS, or WSL; can also point to Windows `ffmpeg.exe` via env vars.

## 🚀 Quick Start

### Requirements

- Python 3.10+
- FFmpeg / FFprobe
- `GEMINI_API_KEY`

### Installation

```bash
git clone https://github.com/yourname/AI-Video-Transcriber.git
cd AI-Video-Transcriber
uv sync
```

Create your local config outside the repo root:

```bash
mkdir -p ~/.config/ai-video-transcriber
cp .env.example ~/.config/ai-video-transcriber/.env
```

### Local media to subtitles

```bash
uv run python cli.py \
  --input /path/to/movie.mkv \
  --format ass \
  --timing align \
  --mode bilingual-zh
```

- If you omit `--outdir`, the subtitle file is written next to the local media file by default.

### URL to subtitles

```bash
uv run python cli.py \
  --url https://www.youtube.com/watch?v=xxxx \
  --format srt \
  --timing align \
  --mode bilingual-zh
```

### Multiple URLs

```bash
uv run python cli.py \
  --urls https://youtu.be/a https://youtu.be/b \
  --continue-on-error
```

### Useful options

- `--mode bilingual-zh|monolingual`
- `--format srt|ass`
- `--timing gemini|align|placeholder`
- `--align-model medium.en|small.en|medium|small`
- `--title "Custom Title"` for local-file jobs
- `--outdir /custom/output` to override the default output folder
- `--keep-audio` to keep the extracted/downloaded audio
- `--segment-seconds 1200` to control chunk size
- `--parallelism 2` to limit concurrent Gemini requests
- `--model gemini-3-flash-preview` to override `GEMINI_MODEL`

### Optional environment variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | **Required.** Your Gemini API key. |
| `GEMINI_MODEL` | Model to use (default: `gemini-3-flash-preview`). |
| `SUBTITLE_MODE` | Default subtitle mode (`bilingual-zh` or `monolingual`). |
| `SUBTITLE_FORMAT` | Default subtitle format (`srt` or `ass`). |
| `SUBTITLE_TIMING` | Default timing strategy (`align` or `gemini`). |
| `ALIGN_MODEL` | Local alignment model. For English, prefer `medium.en`; for multilingual audio, prefer `medium`. |
| `SEGMENT_SECONDS` | Max subtitle chunk duration in seconds (default: `1200`). |
| `TRANSCRIBE_CONCURRENCY` | Parallel workers for chunked subtitle generation. |
| `TRANSCRIBE_AUDIO_MODE` | Audio preprocessing mode: `vocals` (default) or `raw`. |
| `DEMUCS_PYTHON` | Optional Python interpreter path for a dedicated Demucs environment. |
| `DEMUCS_MODEL` | Optional Demucs model name (default: `htdemucs`). |
| `DEMUCS_DEVICE` | Optional Demucs device override, e.g. `cpu` or `cuda`. |
| `FFMPEG_BIN` | Optional path to `ffmpeg` / `ffmpeg.exe`. |
| `FFPROBE_BIN` | Optional path to `ffprobe` / `ffprobe.exe`. |
| `YDL_COOKIEFILE` | Netscape-format cookie file for YouTube (auto-selected for youtube.com/youtu.be URLs). |
| `BILIBILI_COOKIE_FILE` | Netscape-format cookie file for Bilibili (auto-selected for bilibili.com URLs). |
| `YDL_FORMAT` | yt-dlp format selector (default `bestaudio[ext=m4a]/bestaudio/best`). |
| `YDL_FORMAT_MAX_CANDIDATES` | Max fallback format attempts (default 20, recommend 5-10). |
| `YDL_USER_AGENT` | Custom User-Agent if the default is blocked. |

> If you're running inside WSL and your FFmpeg lives on Windows, set `FFMPEG_BIN` and `FFPROBE_BIN` to the Windows executable paths.
>
> If Demucs is installed in a separate Python environment, set `DEMUCS_PYTHON` to that interpreter.
>
> For a public repo, keep real secrets in `~/.config/ai-video-transcriber/.env` instead of the repo root.
>
> `--timing align` depends on `faster-whisper` and `rapidfuzz`. The first aligned run will also download a Whisper model.

## 📦 Outputs

- `subtitle_*.srt` / `subtitle_*.ass`: Final subtitle file.
- `subtitle_*.auto.words.json` or `subtitle_*.<model>.words.json`: local alignment cache files when `align` is enabled.
- Intermediate audio is retained only if `--keep-audio` is specified.

## 🛠️ Development

- Core logic lives in `backend/`:
  - `subtitle_pipeline.py`: source-to-subtitle pipeline
  - `srt_maker.py`: Gemini-driven subtitle generation, chunk merge, and export formatting
  - `video_processor.py`: online media download and audio extraction
- CLI entry point is `cli.py`; logs are printed directly to the terminal.

## 📄 License

MIT License. See `LICENSE` for details.
