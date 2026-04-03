import json
from pathlib import Path
from typing import Optional

import pysubs2

from .timing_aligner import AlignmentEntry, align_entries


def _guess_language(text: str) -> Optional[str]:
    total = len(text) or 1
    hiragana = sum(1 for ch in text if "\u3040" <= ch <= "\u309f")
    katakana = sum(1 for ch in text if "\u30a0" <= ch <= "\u30ff")
    jp_kana = hiragana + katakana
    kanji = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    en = sum(1 for ch in text if ch.isascii() and ch.isalpha())

    if jp_kana / total > 0.05:
        return "ja"
    if kanji / total > 0.2:
        return "zh"
    if en / total > 0.2:
        return "en"
    return None


def _sample_primary_text(subs: pysubs2.SSAFile) -> str:
    sample: list[str] = []
    for line in subs.events:
        text = (line.plaintext or "").strip()
        if not text:
            continue
        first = text.splitlines()[0].strip()
        if not first:
            continue
        sample.append(first)
        if len("\n".join(sample)) >= 800:
            break
    return "\n".join(sample)


def retime_subtitle(
    subtitle_path: Path,
    audio_path: Path,
    *,
    output_path: Optional[Path] = None,
    align_model: str = "medium.en",
    cache_path: Optional[Path] = None,
    language: Optional[str] = None,
) -> dict:
    subtitle_path = Path(subtitle_path)
    audio_path = Path(audio_path)
    if not subtitle_path.exists():
        raise FileNotFoundError(f"字幕文件不存在: {subtitle_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    subs = pysubs2.load(str(subtitle_path))
    target_indexes: list[int] = []
    entries: list[AlignmentEntry] = []

    for idx, line in enumerate(subs.events):
        text = (line.plaintext or "").strip()
        if not text:
            continue
        target_indexes.append(idx)
        entries.append(
            AlignmentEntry(
                start_ms=int(line.start),
                end_ms=int(line.end),
                text=text,
            )
        )

    if not entries:
        raise RuntimeError("字幕里没有可用于重定时的对白行。")

    inferred_language = language or _guess_language(_sample_primary_text(subs))
    target_output = output_path or subtitle_path.with_name(f"{subtitle_path.stem}_retimed{subtitle_path.suffix}")
    target_cache = cache_path or target_output.with_suffix(f".{align_model}.words.json")

    timings, stats = align_entries(
        entries,
        audio_path,
        model_name=align_model,
        cache_path=target_cache,
        language=inferred_language,
    )

    for event_idx, (start_ms, end_ms) in zip(target_indexes, timings):
        subs.events[event_idx].start = start_ms
        subs.events[event_idx].end = end_ms

    subs.save(str(target_output))
    return {
        "subtitle": str(target_output),
        "cache": str(target_cache),
        "align_model": align_model,
        "language": inferred_language,
        "stats": stats,
    }


def retime_subtitle_json(*args, **kwargs) -> str:
    return json.dumps(retime_subtitle(*args, **kwargs), ensure_ascii=False, indent=2)
