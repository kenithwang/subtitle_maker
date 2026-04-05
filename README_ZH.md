<div align="center">

# AI 字幕制作器（CLI 版）

[English](README.md) | 中文

专注命令行的 AI 字幕工具，支持 Gemini 生成、本地强制对齐，以及 `.srt` / `.ass` 输出。

</div>

## ✨ 功能亮点

- 🎞️ 支持 **本地视频/音频文件** 直接生成 `.srt` 或 `.ass`。
- 🌐 支持 `yt-dlp` 下载在线视频音频后再生成字幕（YouTube、B 站等）。
- 🗣️ 用 Gemini 直接生成带时间轴字幕，并可导出为 `SRT` 或 `ASS`。
- 🎤 默认可先做人声分离，生成与原始时长一致、偏对白的转录用 `m4a`。
- 🌏 支持 `bilingual-zh` 模式：每条字幕输出“原文 + 简体中文”双语。
- ✂️ 长音频自动切成 20 分钟左右分片，再合并回一个完整字幕文件。
- 🎯 支持 `--timing align`：使用本地 `faster-whisper` 词级时间戳做二次强制对齐，减少字幕漂移。
- 🧰 支持 Linux / macOS / WSL，也支持通过环境变量显式指定 Windows 版 `ffmpeg.exe`。

## 🚀 快速上手

### 前置条件

- Python 3.10+
- FFmpeg / FFprobe
- `GEMINI_API_KEY`

### 安装

```bash
git clone https://github.com/yourname/AI-Video-Transcriber.git
cd AI-Video-Transcriber
uv sync
```

建议把本地配置放在仓库目录之外：

```bash
mkdir -p ~/.config/ai-video-transcriber
cp .env.example ~/.config/ai-video-transcriber/.env
```

### 本地文件生成字幕

```bash
uv run python cli.py \
  --input /path/to/movie.mkv \
  --format ass \
  --timing align \
  --mode bilingual-zh
```

- 如果不传 `--outdir`，本地文件模式下生成的字幕默认就写在原视频同目录。

## 英语电影的推荐做法

如果你要的是最快的一条龙流程，常规 CLI 仍然是：

```bash
uv run python cli.py \
  --input /path/to/movie.mkv \
  --format ass \
  --timing align \
  --mode bilingual-zh
```

但如果素材是对白很密的英语电影，而且你更在意台词覆盖率而不是一步到位的便利性，就不要把“Gemini 分片直出字幕”当成最高覆盖率方案，它更适合作为快捷方案。

更稳的最佳实践是：

1. 先用 `faster-whisper` 跑整片英文母稿。
2. 再用 GPT-5.4 单独翻译这份英文母稿。
3. 最后把中文回填到英文时间轴上，不改英文 cue 的时间。

一个关键限制：

- `--timing align` 能修时间。
- 但它修不回 Gemini 分片阶段根本没产出来的台词。
- 如果你看到的是整段漏句、压句、合句，根因通常是转录覆盖率，不是单纯时间漂移。

### 在线视频生成字幕

```bash
uv run python cli.py \
  --url https://www.youtube.com/watch?v=xxxx \
  --format srt \
  --timing align \
  --mode bilingual-zh
```

### 多链接批量处理

```bash
uv run python cli.py \
  --urls https://youtu.be/a https://youtu.be/b \
  --continue-on-error
```

### 常用参数

- `--mode bilingual-zh|monolingual`
- `--format srt|ass`
- `--timing gemini|align|placeholder`
- `--align-model medium.en|small.en|medium|small`
- `--title "自定义标题"`：本地文件模式下覆盖输出标题
- `--outdir /custom/output`：覆盖默认输出目录
- `--keep-audio`：保留中间音频文件
- `--segment-seconds 1200`：控制每段字幕分片的最长时长
- `--parallelism 2`：限制并发的 Gemini 请求数
- `--model gemini-3-flash-preview`：覆盖 `GEMINI_MODEL`

### 可选环境变量

| 变量 | 说明 |
|------|------|
| `GEMINI_API_KEY` | **必需。** Gemini API 密钥。 |
| `GEMINI_MODEL` | 使用的模型（默认：`gemini-3-flash-preview`）。 |
| `SUBTITLE_MODE` | 默认字幕模式（`bilingual-zh` 或 `monolingual`）。 |
| `SUBTITLE_FORMAT` | 默认字幕格式（`srt` 或 `ass`）。 |
| `SUBTITLE_TIMING` | 默认时间轴策略（`align` 或 `gemini`）。 |
| `ALIGN_MODEL` | 本地强制对齐模型，英语优先 `medium.en`，多语言优先 `medium`。 |
| `SEGMENT_SECONDS` | 字幕分片最大时长，单位秒（默认：`1200`）。 |
| `TRANSCRIBE_CONCURRENCY` | 分片并行处理数。 |
| `TRANSCRIBE_AUDIO_MODE` | 音频预处理模式：`vocals`（默认）或 `raw`。 |
| `DEMUCS_PYTHON` | 可选，单独 Demucs 环境的 Python 解释器路径。 |
| `DEMUCS_MODEL` | 可选，Demucs 模型名（默认：`htdemucs`）。 |
| `DEMUCS_DEVICE` | 可选，Demucs 设备覆盖，例如 `cpu` 或 `cuda`。 |
| `FFMPEG_BIN` | 可选，自定义 `ffmpeg` / `ffmpeg.exe` 路径。 |
| `FFPROBE_BIN` | 可选，自定义 `ffprobe` / `ffprobe.exe` 路径。 |
| `YDL_COOKIEFILE` | Netscape 格式 Cookie，用于 YouTube（自动识别 youtube.com/youtu.be）。 |
| `BILIBILI_COOKIE_FILE` | Netscape 格式 Cookie，用于 B 站（自动识别 bilibili.com）。 |
| `YDL_FORMAT` | yt-dlp 格式选择（默认 `bestaudio[ext=m4a]/bestaudio/best`）。 |
| `YDL_FORMAT_MAX_CANDIDATES` | 回退格式尝试上限（默认 20，建议 5-10）。 |
| `YDL_USER_AGENT` | 自定义 UA，避免被屏蔽。 |

> 如果你在 WSL 里运行，而 FFmpeg 装在 Windows 侧，可以通过 `FFMPEG_BIN` / `FFPROBE_BIN` 指向 `ffmpeg.exe` / `ffprobe.exe`。
>
> 如果 Demucs 装在单独的 Python 环境里，可以通过 `DEMUCS_PYTHON` 指向那个解释器。
>
> 如果要推公开仓库，真实密钥建议放在 `~/.config/ai-video-transcriber/.env`，不要放在仓库根目录。
>
> `--timing align` 依赖 `faster-whisper` 和 `rapidfuzz`。首次运行还会下载 Whisper 模型。

## 📦 输出文件

- `subtitle_*.srt` / `subtitle_*.ass`：最终字幕文件。
- `subtitle_*.auto.words.json` 或 `subtitle_*.<model>.words.json`：本地强制对齐缓存（开启 `align` 时生成）。
- 如开启 `--keep-audio`，还会在 `temp/` 中保留中间音频文件。

## 🛠️ 开发说明

- 核心逻辑位于 `backend/`：
  - `subtitle_pipeline.py`：从输入源到字幕文件的主流程
  - `srt_maker.py`：Gemini 字幕生成、分片合并与格式导出
  - `video_processor.py`：在线视频下载与音频提取
- CLI 入口为 `cli.py`，日志信息直接输出到终端。
- 仓库内还附带了一套可复用的 Codex skill，位于 `skills/movie-audio-subtitle-fix/`，用于本地电影音轨修复和更高覆盖率的字幕工作流。

## 📄 许可协议

MIT License，详见 `LICENSE`。
