# movie-audio-subtitle-fix

Utilities and workflow notes for fixing incompatible movie audio tracks and generating bilingual subtitles.

This skill can be vendored inside the main repo under `skills/movie-audio-subtitle-fix`, or used standalone next to an `AI-Video-Transcriber` checkout.

## Current Best Practice

For English-language movies, the default workflow is:

1. Repair playback audio if needed with `scripts/fix_movie_audio.py`.
2. Build a whole-movie English subtitle source with `scripts/prepare_english_subtitle_source.py`.
3. Translate the exported `id<TAB>english` lines with GPT-5.4 into `id<TAB>zh`.
4. Merge the translated Chinese back into the English ASS with `scripts/translate_ass_preserve_timing.py --translations-file`.

This is the preferred path because full-movie `faster-whisper` is much more stable than chunked Gemini transcript generation for dialogue coverage and timing consistency.

## Scripts

- `scripts/fix_movie_audio.py`
  Inspect audio tracks and transcode one selected track to a single AAC track for player compatibility.
- `scripts/prepare_english_subtitle_source.py`
  Extract transcription audio when needed, run full-movie `faster-whisper`, write an English ASS, write a word cache, and export `*.source.tsv`.
- `scripts/translate_ass_preserve_timing.py`
  Merge offline Chinese translations back into the English ASS without changing cue timings.
- `scripts/order_only_retime_ass.py`
  Retime an ASS by trusting cue order plus word timestamps, not the subtitle's original timestamps.
- `scripts/generate_and_retime_subtitles.py`
  Legacy Gemini chunk-transcript path. This is opt-in only and should not be the default for English movies.
- `scripts/md_transcript_to_ass.py`
  Convert a raw Markdown transcript into ASS for salvage workflows.

## Example

Generate the English source:

```bash
ALIGN_DEVICE=cuda \
./.venv/bin/python \
  skills/movie-audio-subtitle-fix/scripts/prepare_english_subtitle_source.py \
  "/path/to/movie.mkv"
```

Merge bilingual subtitles after GPT-5.4 translation:

```bash
python3 skills/movie-audio-subtitle-fix/scripts/translate_ass_preserve_timing.py \
  "/path/to/movie_fw_en.ass" \
  --output "/path/to/movie_fw_en_bilingual.ass" \
  --translations-file "/path/to/movie_fw_en_zh.tsv"
```

## Notes

- Keep one Chinese line per English cue when producing the translation TSV.
- For English movies, do not default to Gemini transcript generation.
- If the input subtitles already have the correct text order but bad timestamps, use `scripts/order_only_retime_ass.py` directly.
