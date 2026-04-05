---
name: movie-audio-subtitle-fix
description: Use when working with local movie files that need an ACG Player-compatible audio track or need subtitles generated from scratch. Default to the best-practice pipeline: fix playback audio when needed, generate a whole-movie English ASS with faster-whisper, then translate to Chinese with GPT-5.4 and merge the bilingual text back without changing timings. The older Gemini chunk-transcript path is legacy-only.
---

# Movie Audio Subtitle Fix

## When To Use

- Local movie opens but ACG Player cannot decode the audio track.
- A movie has no usable subtitles and needs a generated `.ass`.
- A generated subtitle drifts and should be re-aligned without trusting its original timestamps.

## Paths And Prerequisites

- This skill can be used in two layouts:
  - vendored inside this repo under `skills/movie-audio-subtitle-fix`
  - standalone next to a sibling checkout named `AI-Video-Transcriber`
- If a script cannot auto-detect the transcriber root, pass an explicit `--transcriber-dir`.
- `ffmpeg` and `ffprobe` must be discoverable, or provided through `FFMPEG_BIN` and `FFPROBE_BIN`.
- For subtitle work, prefer the transcriber repo venv.
- When this skill is vendored inside the repo, that is usually:
  `"../../.venv/bin/python"`
- For GPU faster-whisper, set `ALIGN_DEVICE=cuda` before running the English source step.

## Audio Workflow

Use `scripts/fix_movie_audio.py` when the user wants a single player-safe audio track.

- First inspect tracks with `--list-audio-tracks`.
- Then transcode the chosen track to AAC-LC stereo and keep only that audio stream.
- Video, subtitles, chapters, and attachments are copied when the output container supports them.
- Default output is `*_aac_single_audio.mkv`.

Example:

```bash
python3 scripts/fix_movie_audio.py \
  "/path/to/movie.mkv" \
  --list-audio-tracks
```

```bash
python3 scripts/fix_movie_audio.py \
  "/path/to/movie.mkv" \
  --audio-track 0
```

## Best Practice Subtitle Workflow

Default to this workflow for English-language movies.

1. If playback audio is problematic, repair it first with `scripts/fix_movie_audio.py`.
2. Generate a whole-movie English subtitle source with `scripts/prepare_english_subtitle_source.py`.
3. Translate the exported English lines to Simplified Chinese with GPT-5.4 directly.
4. Merge the translated Chinese back into the English ASS with `scripts/translate_ass_preserve_timing.py --translations-file`.

Why this is the default:

- `faster-whisper` on the full movie is much better at dialogue coverage than Gemini chunk subtitles.
- Gemini chunk transcripts can reset timestamps per chunk, merge adjacent lines, or drop entire stretches of dialogue.
- The best current practice is to trust `faster-whisper` for English transcription and timing, then use GPT-5.4 only for translation.

### Step 1: Build The English Source

Use `scripts/prepare_english_subtitle_source.py`.

What it does:

- If the input is a video, extracts a 16 kHz mono transcription audio.
- Runs full-movie `faster-whisper` English transcription.
- Writes an English `.ass`.
- Writes a word cache JSON.
- Writes a `source.tsv` file in the form `id<TAB>english`, which is the source material to translate with GPT-5.4.

Example:

```bash
ALIGN_DEVICE=cuda \
./.venv/bin/python \
  skills/movie-audio-subtitle-fix/scripts/prepare_english_subtitle_source.py \
  "/path/to/movie.mkv"
```

Typical outputs:

- English ASS: `*_fw_en.ass`
- Word cache: `*_fw_en.{model}.words.json`
- Source lines: `*_fw_en.source.tsv`

Defaults:

- Model: `medium.en`
- Language: `en`
- Beam size: `5`
- VAD: off
- Condition on previous text: off

### Step 2: Translate With GPT-5.4

Do not use Gemini by default.

- Use GPT-5.4 directly to translate `*_fw_en.source.tsv`.
- Produce a new TSV in the form `id<TAB>zh`.
- Keep one Chinese line per English cue. Do not merge or split cue ids.

### Step 3: Merge The Bilingual Subtitle

Use `scripts/translate_ass_preserve_timing.py --translations-file`.

Example:

```bash
python3 scripts/translate_ass_preserve_timing.py \
  "/path/to/movie_fw_en.ass" \
  --output "/path/to/movie_fw_en_bilingual.ass" \
  --translations-file "/path/to/movie_fw_en_zh.tsv"
```

This preserves the original English cue timings and only replaces subtitle text with:

- line 1: English
- line 2: Chinese

## Retiming An Existing ASS

If an existing ASS already has the right text order but bad timestamps, use `scripts/order_only_retime_ass.py` directly with an existing words JSON.

```bash
./.venv/bin/python \
  skills/movie-audio-subtitle-fix/scripts/order_only_retime_ass.py \
  "/path/to/input.ass" \
  "/path/to/input.medium.en.words.json" \
  "/path/to/output_order_only.ass"
```

## Legacy Gemini Workflow

The old Gemini chunk-transcript path is legacy-only.

- Script: `scripts/generate_and_retime_subtitles.py`
- It now requires explicit opt-in: `--allow-gemini-transcript`
- Only use it when the user explicitly asks for the old Gemini transcript flow, or when they want to salvage an existing Gemini transcript / Markdown transcript source.

Do not choose this path by default for English movies.

## Scripts

- `scripts/fix_movie_audio.py`: inspect and repair movie audio tracks for playback compatibility.
- `scripts/prepare_english_subtitle_source.py`: best-practice entrypoint for English source generation.
- `scripts/translate_ass_preserve_timing.py`: merge offline GPT-5.4 Chinese translations back into the timed English ASS.
- `scripts/order_only_retime_ass.py`: deterministic retimer that ignores original subtitle timestamps.
- `scripts/generate_and_retime_subtitles.py`: legacy Gemini transcript workflow, explicit opt-in only.
