import bisect
import json
import math
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


TOKEN_RE = re.compile(r"[a-z0-9']+|[\u3040-\u30ff\u3400-\u9fff]")


@dataclass
class Word:
    text: str
    norm: str
    start: float
    end: float


@dataclass
class AlignmentEntry:
    start_ms: int
    end_ms: int
    text: str


def _require_alignment_deps():
    try:
        from faster_whisper import WhisperModel  # type: ignore
        from rapidfuzz import fuzz  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "本地强制对齐依赖未安装。请在项目目录运行 `uv sync`，"
            "确保 `faster-whisper` 和 `rapidfuzz` 已安装。"
        ) from exc
    return WhisperModel, fuzz


def _iter_model_cache_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()

    def add(path: Optional[Path]) -> None:
        if not path:
            return
        resolved = path.expanduser()
        if resolved in seen:
            return
        seen.add(resolved)
        roots.append(resolved)

    hub_cache = os.getenv("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        add(Path(hub_cache))

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        add(Path(hf_home) / "hub")

    xdg_cache = os.getenv("XDG_CACHE_HOME")
    if xdg_cache:
        add(Path(xdg_cache) / "huggingface" / "hub")
        add(Path(xdg_cache) / "faster-whisper")

    add(Path.home() / ".cache" / "huggingface" / "hub")
    add(Path.home() / ".cache" / "faster-whisper")
    add(Path.cwd() / ".cache" / "huggingface" / "hub")
    add(Path.cwd() / ".cache" / "faster-whisper")

    mnt_users = Path("/mnt/c/Users")
    if mnt_users.exists():
        for user_dir in mnt_users.iterdir():
            if not user_dir.is_dir():
                continue
            add(user_dir / ".cache" / "huggingface" / "hub")
            add(user_dir / ".cache" / "faster-whisper")

    return roots


def _resolve_local_model_path(model_name: str) -> Optional[Path]:
    repo_dir_name = f"models--Systran--faster-whisper-{model_name}"
    for root in _iter_model_cache_roots():
        repo_dir = root / repo_dir_name
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.exists():
            continue
        snapshots = sorted(
            (path for path in snapshots_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )
        for snapshot in snapshots:
            if (snapshot / "model.bin").exists() and (snapshot / "config.json").exists():
                return snapshot
    return None


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("’", "'").replace("`", "'")
    tokens = TOKEN_RE.findall(text)
    return " ".join(tokens)


def extract_alignment_text(text: str) -> str:
    lines = (text or "").replace(r"\N", "\n").splitlines()
    for line in lines:
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return ""


def build_model(model_name: str):
    WhisperModel, _ = _require_alignment_deps()
    local_model_path = _resolve_local_model_path(model_name)
    model_ref = str(local_model_path) if local_model_path else model_name
    device = (os.getenv("ALIGN_DEVICE") or "cpu").strip().lower()
    if device == "cuda":
        try:
            return WhisperModel(model_ref, device="cuda", compute_type="float16")
        except Exception:
            pass
    return WhisperModel(model_ref, device="cpu", compute_type="int8")


def transcribe_words(
    audio_path: Path,
    model_name: str,
    cache_path: Path,
    language: Optional[str] = None,
) -> list[Word]:
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        return [Word(**item) for item in raw]

    model = build_model(model_name)
    transcribe_kwargs: dict[str, Any] = {
        "beam_size": 5,
        "word_timestamps": True,
        "vad_filter": True,
        "condition_on_previous_text": False,
    }
    if language and not model_name.endswith(".en"):
        transcribe_kwargs["language"] = language

    segments, _ = model.transcribe(str(audio_path), **transcribe_kwargs)

    words: list[Word] = []
    for segment in segments:
        for word in segment.words or []:
            if word.start is None or word.end is None:
                continue
            norm = normalize_text(word.word)
            if not norm:
                continue
            words.append(
                Word(
                    text=word.word,
                    norm=norm,
                    start=float(word.start),
                    end=float(word.end),
                )
            )

    cache_path.write_text(
        json.dumps([word.__dict__ for word in words], ensure_ascii=False),
        encoding="utf-8",
    )
    return words


def score_candidate(
    target: str,
    candidate: str,
    start_penalty: int,
    len_penalty: int,
) -> float:
    _, fuzz = _require_alignment_deps()
    ratio = fuzz.ratio(target, candidate)
    partial = fuzz.partial_ratio(target, candidate) - 4.0
    score = max(ratio, partial)
    score -= start_penalty * 0.35
    score -= len_penalty * 0.80
    return score


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def estimate_anchor_model(anchors: list[tuple[float, float]]) -> tuple[float, float]:
    if not anchors:
        return 1.0, 0.0
    if len(anchors) == 1:
        entry_time, asr_time = anchors[0]
        return 1.0, asr_time - entry_time

    sample = anchors[-12:]
    xs = [entry_time for entry_time, _ in sample]
    ys = [asr_time for _, asr_time in sample]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    variance = sum((x - mean_x) ** 2 for x in xs)
    if variance <= 1e-9:
        slope = 1.0
    else:
        slope = sum((x - mean_x) * (y - mean_y) for x, y in sample) / variance

    slope = clamp(slope, 0.94, 1.06)
    intercept = mean_y - slope * mean_x
    return slope, intercept


def predict_expected_time(
    entry_start_s: float,
    anchors: list[tuple[float, float]],
    subtitle_total_s: float,
    asr_total_s: float,
) -> float:
    global_prediction = (
        (entry_start_s / subtitle_total_s) * asr_total_s if subtitle_total_s > 0 else entry_start_s
    )
    if not anchors:
        return clamp(global_prediction, 0.0, asr_total_s)

    slope, intercept = estimate_anchor_model(anchors)
    local_prediction = entry_start_s * slope + intercept
    if len(anchors) == 1:
        predicted = local_prediction
    else:
        predicted = local_prediction * 0.9 + global_prediction * 0.1
    return clamp(predicted, 0.0, asr_total_s)


def compute_search_window(token_count: int, anchor_count: int) -> float:
    base = max(35.0, min(90.0, token_count * 5.0))
    if anchor_count <= 0:
        return min(150.0, base + 50.0)
    if anchor_count == 1:
        return min(130.0, base + 25.0)
    if anchor_count < 4:
        return min(110.0, base + 12.0)
    return base


def timing_weight(anchor_count: int) -> float:
    if anchor_count <= 1:
        return 0.0
    if anchor_count < 4:
        return 0.65
    if anchor_count < 8:
        return 1.20
    return 1.80


def max_anchor_deviation(anchor_count: int, token_count: int) -> float:
    if anchor_count <= 0:
        return 30.0
    if anchor_count == 1:
        return 20.0
    if anchor_count < 4:
        return 18.0
    if anchor_count < 10:
        return 15.0
    if anchor_count < 20:
        return 12.0
    return 8.0 if token_count <= 3 else 6.0


def gap_penalty(words: list[Word], start: int, end: int) -> float:
    penalty = 0.0
    for idx in range(start + 1, end):
        gap = words[idx].start - words[idx - 1].end
        if gap > 0.45:
            penalty += gap - 0.45
    return penalty


def build_word_index(words: list[Word]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for idx, word in enumerate(words):
        index.setdefault(word.norm, []).append(idx)
    return index


def match_line(
    target: str,
    words: list[Word],
    word_starts: list[float],
    cursor: int,
    expected_time: float,
    window_seconds: float,
    anchor_count: int,
) -> tuple[Optional[tuple[int, int, float]], int]:
    tokens = target.split()
    token_count = len(tokens)
    if token_count <= 1:
        return None, cursor

    best: Optional[tuple[int, int, float]] = None
    best_cursor = cursor

    search_start = bisect.bisect_left(word_starts, max(0.0, expected_time - window_seconds))
    max_start = bisect.bisect_right(word_starts, expected_time + window_seconds) - 1
    search_start = max(0, min(search_start, len(words) - 1))
    max_start = max(search_start, min(len(words) - 1, max_start))
    search_start = max(search_start, cursor - 20)
    max_start = max(search_start, max_start)

    for start in range(search_start, max_start + 1):
        min_len = max(1, token_count - 3)
        max_len = min(len(words) - start, max(token_count + 8, int(math.ceil(token_count * 1.8))))
        for span_len in range(min_len, max_len + 1):
            end = start + span_len
            candidate = " ".join(word.norm for word in words[start:end])
            timing_error = abs(words[start].start - expected_time)
            score = score_candidate(
                target,
                candidate,
                abs(start - cursor),
                abs(span_len - token_count),
            )
            score -= gap_penalty(words, start, end) * 3.5
            score -= timing_error * timing_weight(anchor_count)
            if best is None or score > best[2]:
                best = (start, end, score)
                best_cursor = end

    threshold = 58.0
    if token_count <= 3:
        threshold = 78.0
    elif token_count <= 5:
        threshold = 70.0
    elif token_count <= 8:
        threshold = 64.0

    if best is None or best[2] < threshold:
        return None, cursor
    if abs(words[best[0]].start - expected_time) > max_anchor_deviation(anchor_count, token_count):
        return None, cursor
    return best, best_cursor


def fallback_match_line(
    target: str,
    words: list[Word],
    cursor: int,
    word_index: dict[str, list[int]],
    expected_time: float,
    min_start_idx: Optional[int] = None,
    max_start_idx: Optional[int] = None,
) -> tuple[Optional[tuple[int, int, float]], int]:
    tokens = target.split()
    token_count = len(tokens)
    if token_count < 8:
        return None, cursor

    ranked_tokens: list[tuple[int, int, int, str]] = []
    for tok_idx, token in enumerate(tokens):
        if len(token) <= 2:
            continue
        positions = word_index.get(token)
        if not positions:
            continue
        ranked_tokens.append((len(positions), -len(token), tok_idx, token))

    if not ranked_tokens:
        return None, cursor

    lower_bound = max(0, cursor - 3, min_start_idx or 0)
    upper_bound = max_start_idx if max_start_idx is not None else len(words)
    if upper_bound - lower_bound <= 0:
        return None, cursor

    candidate_starts: set[int] = set()
    for _freq, _neg_len, tok_idx, token in sorted(ranked_tokens)[:4]:
        for pos in word_index[token][:160]:
            start = pos - tok_idx
            for delta in (-2, -1, 0, 1, 2):
                probe = start + delta
                if probe < lower_bound:
                    continue
                if probe >= upper_bound:
                    continue
                candidate_starts.add(probe)

    best: Optional[tuple[int, int, float]] = None
    best_cursor = cursor
    min_len = max(1, token_count - 3)

    for start in sorted(candidate_starts):
        max_len = min(upper_bound - start, max(token_count + 8, int(math.ceil(token_count * 1.8))))
        for span_len in range(min_len, max_len + 1):
            end = start + span_len
            candidate = " ".join(word.norm for word in words[start:end])
            score = score_candidate(target, candidate, 0, abs(span_len - token_count))
            score -= gap_penalty(words, start, end) * 3.5
            score -= abs(words[start].start - expected_time) * 0.08
            if best is None or score > best[2]:
                best = (start, end, score)
                best_cursor = end

    if best is None:
        return None, cursor

    max_deviation = 90.0
    if token_count >= 16:
        max_deviation = 180.0
    elif token_count >= 12:
        max_deviation = 120.0

    if abs(words[best[0]].start - expected_time) > max_deviation:
        return None, cursor

    threshold = 92.0
    if token_count >= 16:
        threshold = 84.0
    elif token_count >= 12:
        threshold = 88.0
    elif token_count >= 10:
        threshold = 90.0

    if best[2] < threshold:
        return None, cursor
    return best, best_cursor


def interpolate_expected_from_neighbors(
    entries: list[AlignmentEntry],
    idx: int,
    prev_idx: int,
    next_idx: int,
    words: list[Word],
    prev_span: tuple[int, int],
    next_span: tuple[int, int],
) -> float:
    prev_entry_t = entries[prev_idx].start_ms / 1000.0
    next_entry_t = entries[next_idx].start_ms / 1000.0
    prev_word_t = words[prev_span[0]].start
    next_word_t = words[next_span[0]].start
    if next_entry_t <= prev_entry_t:
        return prev_word_t
    ratio = (entries[idx].start_ms / 1000.0 - prev_entry_t) / (next_entry_t - prev_entry_t)
    return prev_word_t + ratio * (next_word_t - prev_word_t)


def build_reference_entries(
    entries: list[AlignmentEntry],
    assigned: list[Optional[tuple[int, int]]],
) -> tuple[list[AlignmentEntry], int]:
    offsets = [
        entries[idx].start_ms - assigned_time[0]
        for idx, assigned_time in enumerate(assigned)
        if assigned_time is not None
    ]
    if not offsets:
        return entries, 0

    global_shift_ms = int(round(statistics.median(offsets)))
    if abs(global_shift_ms) < 1000:
        return entries, global_shift_ms

    normalized = [
        AlignmentEntry(
            start_ms=entry.start_ms - global_shift_ms,
            end_ms=entry.end_ms - global_shift_ms,
            text=entry.text,
        )
        for entry in entries
    ]
    return normalized, global_shift_ms


def interpolate_block(
    entries: list[AlignmentEntry],
    assigned: list[Optional[tuple[int, int]]],
    start_idx: int,
    end_idx: int,
    timeline_end_ms: Optional[int] = None,
) -> None:
    prev_idx = start_idx - 1
    next_idx = end_idx + 1
    prev_time = assigned[prev_idx] if prev_idx >= 0 else None
    next_time = assigned[next_idx] if next_idx < len(assigned) else None

    block_old_start = entries[start_idx].start_ms
    block_old_end = entries[end_idx].end_ms

    if prev_time and next_time and block_old_end > block_old_start:
        new_start = prev_time[1]
        new_end = next_time[0]
        if new_end <= new_start:
            new_end = new_start + max(400, block_old_end - block_old_start)
        scale = (new_end - new_start) / float(block_old_end - block_old_start)
        for idx in range(start_idx, end_idx + 1):
            rel_start = entries[idx].start_ms - block_old_start
            rel_end = entries[idx].end_ms - block_old_start
            assigned[idx] = (
                int(round(new_start + rel_start * scale)),
                int(round(new_start + rel_end * scale)),
            )
        return

    if prev_time:
        if timeline_end_ms and prev_idx >= 0 and entries[end_idx].end_ms > entries[prev_idx].end_ms:
            old_start = entries[prev_idx].end_ms
            old_end = entries[end_idx].end_ms
            new_start = prev_time[1]
            new_end = max(new_start + 400, timeline_end_ms)
            scale = (new_end - new_start) / float(old_end - old_start)
            for idx in range(start_idx, end_idx + 1):
                rel_start = entries[idx].start_ms - old_start
                rel_end = entries[idx].end_ms - old_start
                assigned[idx] = (
                    int(round(new_start + rel_start * scale)),
                    int(round(new_start + rel_end * scale)),
                )
        else:
            delta = prev_time[1] - entries[prev_idx].end_ms
            for idx in range(start_idx, end_idx + 1):
                assigned[idx] = (entries[idx].start_ms + delta, entries[idx].end_ms + delta)
        return

    if next_time:
        delta = next_time[0] - entries[next_idx].start_ms
        for idx in range(start_idx, end_idx + 1):
            assigned[idx] = (entries[idx].start_ms + delta, entries[idx].end_ms + delta)
        return

    for idx in range(start_idx, end_idx + 1):
        assigned[idx] = (entries[idx].start_ms, entries[idx].end_ms)


def smooth_times(
    entries: list[AlignmentEntry],
    assigned: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    prev_end = 0
    smoothed: list[tuple[int, int]] = []
    for idx, _entry in enumerate(entries):
        start, end = assigned[idx]
        if end <= start:
            end = start + 500
        if idx > 0 and start < prev_end - 150:
            start = prev_end - 150
        if end <= start:
            end = start + 500
        start = max(0, int(start))
        end = max(start + 200, int(end))
        smoothed.append((start, end))
        prev_end = end
    return smoothed


def _run_alignment_pass(
    entries: list[AlignmentEntry],
    words: list[Word],
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    word_starts = [word.start for word in words]
    word_index = build_word_index(words)
    asr_total = words[-1].end
    subtitle_total = max(entry.end_ms for entry in entries) / 1000.0 if entries else 0.0
    assigned: list[Optional[tuple[int, int]]] = [None] * len(entries)
    matched_spans: list[Optional[tuple[int, int]]] = [None] * len(entries)
    cursor = 0
    matched = 0
    fallback_matched = 0
    block_refined = 0
    anchors: list[tuple[float, float]] = []

    for idx, entry in enumerate(entries):
        target = normalize_text(extract_alignment_text(entry.text))
        if not target:
            continue

        entry_start_s = entry.start_ms / 1000.0
        expected_time = predict_expected_time(entry_start_s, anchors, subtitle_total, asr_total)
        window_seconds = compute_search_window(len(target.split()), len(anchors))
        match, next_cursor = match_line(
            target,
            words,
            word_starts,
            cursor,
            expected_time,
            window_seconds,
            len(anchors),
        )
        if match is None:
            match, next_cursor = fallback_match_line(target, words, cursor, word_index, expected_time)
            if match is None:
                continue
            fallback_matched += 1

        start_idx, end_idx, _score = match
        if start_idx + 3 < cursor:
            continue

        start_ms = int(round(words[start_idx].start * 1000))
        end_ms = int(round(words[end_idx - 1].end * 1000))
        assigned[idx] = (start_ms, end_ms)
        matched_spans[idx] = (start_idx, end_idx)
        matched += 1
        cursor = max(cursor, next_cursor)
        anchors.append((entry_start_s, words[start_idx].start))

    for _ in range(8):
        changed = False
        pos = 0
        while pos < len(matched_spans):
            if matched_spans[pos] is not None:
                pos += 1
                continue

            block_start = pos
            while pos < len(matched_spans) and matched_spans[pos] is None:
                pos += 1
            block_end = pos - 1

            prev_idx = block_start - 1
            next_idx = pos
            if prev_idx < 0 or next_idx >= len(matched_spans):
                continue
            prev_span = matched_spans[prev_idx]
            next_span = matched_spans[next_idx]
            if prev_span is None or next_span is None:
                continue

            search_lo = max(0, prev_span[1] - 2)
            search_hi = min(len(words), next_span[0] + 2)
            if search_hi - search_lo < 8:
                continue

            candidates: list[tuple[int, int, str]] = []
            for idx in range(block_start, block_end + 1):
                target = normalize_text(extract_alignment_text(entries[idx].text))
                token_count = len(target.split())
                if token_count < 8:
                    continue
                candidates.append((token_count, idx, target))
            candidates.sort(reverse=True)

            for _token_count, idx, target in candidates[:6]:
                expected_time = interpolate_expected_from_neighbors(
                    entries,
                    idx,
                    prev_idx,
                    next_idx,
                    words,
                    prev_span,
                    next_span,
                )
                match, _next_cursor = fallback_match_line(
                    target,
                    words,
                    search_lo,
                    word_index,
                    expected_time,
                    min_start_idx=search_lo,
                    max_start_idx=search_hi,
                )
                if match is None:
                    continue

                start_idx, end_idx, _score = match
                assigned[idx] = (
                    int(round(words[start_idx].start * 1000)),
                    int(round(words[end_idx - 1].end * 1000)),
                )
                matched_spans[idx] = (start_idx, end_idx)
                matched += 1
                fallback_matched += 1
                block_refined += 1
                changed = True
                break

        if not changed:
            break

    reference_entries, global_shift_ms = build_reference_entries(entries, assigned)

    timeline_end_ms = int(round(asr_total * 1000))
    pos = 0
    while pos < len(assigned):
        if assigned[pos] is not None:
            pos += 1
            continue
        block_start = pos
        while pos < len(assigned) and assigned[pos] is None:
            pos += 1
        interpolate_block(reference_entries, assigned, block_start, pos - 1, timeline_end_ms=timeline_end_ms)

    finalized = [slot for slot in assigned if slot is not None]
    if len(finalized) != len(entries):
        raise RuntimeError("强制对齐未能为所有字幕行生成时间轴。")

    smoothed = smooth_times(entries, finalized)
    stats = {
        "matched_lines": matched,
        "fallback_matched_lines": fallback_matched,
        "block_refined_lines": block_refined,
        "global_shift_ms": global_shift_ms,
        "total_lines": len(entries),
        "word_count": len(words),
    }
    return smoothed, stats


def align_entries(
    entries: list[AlignmentEntry],
    audio_path: Path,
    *,
    model_name: str,
    cache_path: Path,
    language: Optional[str] = None,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    words = transcribe_words(audio_path, model_name, cache_path, language=language)
    if not words:
        raise RuntimeError("本地 ASR 没有返回任何词级时间戳。")

    first_timings, first_stats = _run_alignment_pass(entries, words)

    estimated_input_shift_ms = int(first_stats.get("global_shift_ms", 0))
    final_timings = first_timings
    final_stats = dict(first_stats)

    if abs(estimated_input_shift_ms) >= 1000:
        normalized_entries = [
            AlignmentEntry(
                start_ms=entry.start_ms - estimated_input_shift_ms,
                end_ms=entry.end_ms - estimated_input_shift_ms,
                text=entry.text,
            )
            for entry in entries
        ]
        second_timings, second_stats = _run_alignment_pass(normalized_entries, words)
        final_timings = second_timings
        final_stats = second_stats

    final_stats.update(
        {
            "estimated_input_shift_ms": estimated_input_shift_ms,
            "cache": str(cache_path),
            "model": model_name,
            "language": language,
        }
    )
    return final_timings, final_stats
