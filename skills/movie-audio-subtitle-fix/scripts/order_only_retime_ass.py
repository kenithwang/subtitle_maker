#!/usr/bin/env python3
import argparse
import bisect
import collections
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz


TAG_RE = re.compile(r"\{[^}]*\}")
TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass
class Word:
    norm: str
    start: float
    end: float


@dataclass
class DialogueLine:
    file_line_idx: int
    fields: list[str]
    text: str
    english: str
    tokens: list[str]


@dataclass
class AnchorCandidate:
    line_idx: int
    estimated_start: int
    estimated_end: int
    confidence: float


@dataclass
class MatchSpan:
    start_idx: int
    end_idx: int
    score: float
    kind: str


def strip_ass_tags(text: str) -> str:
    return TAG_RE.sub("", text)


def extract_english_text(text: str) -> str:
    plain = strip_ass_tags(text)
    first_line = plain.split(r"\N", 1)[0]
    return first_line.strip()


def normalize_tokens(text: str) -> list[str]:
    clean = (text or "").lower().replace("’", "'").replace("`", "'")
    return TOKEN_RE.findall(clean)


def parse_ass(path: Path) -> tuple[list[str], list[DialogueLine]]:
    source_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    dialogues: list[DialogueLine] = []
    for file_line_idx, raw in enumerate(source_lines):
        if not raw.startswith("Dialogue:"):
            continue
        fields = raw.split(",", 9)
        if len(fields) != 10:
            continue
        text = fields[9]
        english = extract_english_text(text)
        dialogues.append(
            DialogueLine(
                file_line_idx=file_line_idx,
                fields=fields,
                text=text,
                english=english,
                tokens=normalize_tokens(english),
            )
        )
    return source_lines, dialogues


def load_words(path: Path) -> list[Word]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    words: list[Word] = []
    for item in raw:
        norm = item.get("norm")
        start = item.get("start")
        end = item.get("end")
        if not norm or start is None or end is None:
            continue
        words.append(Word(norm=str(norm), start=float(start), end=float(end)))
    if not words:
        raise RuntimeError("词级时间戳 JSON 里没有可用数据。")
    return words


def build_ngram_index(words: list[Word], min_n: int = 2, max_n: int = 5) -> dict[int, dict[tuple[str, ...], list[int]]]:
    index: dict[int, dict[tuple[str, ...], list[int]]] = {
        n: collections.defaultdict(list) for n in range(min_n, max_n + 1)
    }
    norms = [word.norm for word in words]
    for n in range(min_n, max_n + 1):
        for start_idx in range(0, len(norms) - n + 1):
            index[n][tuple(norms[start_idx : start_idx + n])].append(start_idx)
    return index


def cluster_offsets(hits: list[tuple[int, int, int, int]], tolerance: int = 2) -> tuple[int, float]:
    offsets = sorted({hit[0] for hit in hits})
    best_offset = offsets[0]
    best_score = -1.0
    for probe in offsets:
        members = [hit for hit in hits if abs(hit[0] - probe) <= tolerance]
        weighted = sum(hit[3] for hit in members)
        count_bonus = len(members) * 0.5
        score = weighted + count_bonus
        if score > best_score:
            best_offset = int(round(statistics.median(hit[0] for hit in members)))
            best_score = score
    return best_offset, best_score


def find_anchor_candidate(
    line_idx: int,
    tokens: list[str],
    ngram_index: dict[int, dict[tuple[str, ...], list[int]]],
) -> Optional[AnchorCandidate]:
    if len(tokens) < 2:
        return None

    max_n = min(5, len(tokens))
    min_n = 3 if len(tokens) >= 3 else 2
    for n in range(max_n, min_n - 1, -1):
        hits: list[tuple[int, int, int, int]] = []
        for tok_pos in range(0, len(tokens) - n + 1):
            gram = tuple(tokens[tok_pos : tok_pos + n])
            occurrences = ngram_index[n].get(gram)
            if not occurrences or len(occurrences) != 1:
                continue
            asr_pos = occurrences[0]
            hits.append((asr_pos - tok_pos, tok_pos, asr_pos, n))
        if not hits:
            continue
        best_offset, confidence = cluster_offsets(hits)
        return AnchorCandidate(
            line_idx=line_idx,
            estimated_start=best_offset,
            estimated_end=best_offset + max(1, len(tokens)),
            confidence=confidence,
        )
    return None


def select_monotonic_anchors(candidates: list[AnchorCandidate]) -> list[AnchorCandidate]:
    if not candidates:
        return []
    count = len(candidates)
    dp = [cand.confidence for cand in candidates]
    prev = [-1] * count
    for idx, cand in enumerate(candidates):
        for earlier in range(idx):
            prev_cand = candidates[earlier]
            if prev_cand.estimated_start >= cand.estimated_start:
                continue
            score = dp[earlier] + cand.confidence
            if score > dp[idx]:
                dp[idx] = score
                prev[idx] = earlier
    best_idx = max(range(count), key=lambda idx: dp[idx])
    chain: list[AnchorCandidate] = []
    while best_idx != -1:
        chain.append(candidates[best_idx])
        best_idx = prev[best_idx]
    chain.reverse()
    return chain


def build_word_index(words: list[Word]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for idx, word in enumerate(words):
        out.setdefault(word.norm, []).append(idx)
    return out


def score_candidate(target_tokens: list[str], candidate_tokens: list[str]) -> float:
    target = " ".join(target_tokens)
    candidate = " ".join(candidate_tokens)
    score = max(fuzz.ratio(target, candidate), fuzz.partial_ratio(target, candidate) - 4.0)
    score -= abs(len(candidate_tokens) - len(target_tokens)) * 0.90
    if candidate_tokens:
        if target_tokens[0] != candidate_tokens[0]:
            score -= 1.25
        if target_tokens[-1] != candidate_tokens[-1]:
            score -= 1.25
    return score


def gap_penalty(words: list[Word], start_idx: int, end_idx: int) -> float:
    penalty = 0.0
    for idx in range(start_idx + 1, end_idx):
        gap = words[idx].start - words[idx - 1].end
        if gap > 0.45:
            penalty += gap - 0.45
    return penalty


def line_threshold(token_count: int) -> float:
    if token_count <= 2:
        return 86.0
    if token_count <= 3:
        return 79.0
    if token_count <= 5:
        return 70.0
    if token_count <= 8:
        return 63.0
    return 57.0


def iter_window_positions(positions: list[int], low: int, high: int) -> list[int]:
    left = bisect.bisect_left(positions, low)
    right = bisect.bisect_right(positions, high - 1)
    return positions[left:right]


def candidate_starts(
    tokens: list[str],
    word_index: dict[str, list[int]],
    low: int,
    high: int,
    approx_start: int,
) -> list[int]:
    starts: list[int] = []
    seen: set[int] = set()

    token_choices: list[tuple[int, int, int, str]] = []
    for tok_pos, token in enumerate(tokens):
        if len(token) <= 2:
            continue
        positions = word_index.get(token)
        if not positions:
            continue
        window_positions = iter_window_positions(positions, low, high)
        if not window_positions:
            continue
        token_choices.append((len(window_positions), -len(token), tok_pos, token))
    token_choices.sort()

    for _freq, _neg_len, tok_pos, token in token_choices[:5]:
        for pos in iter_window_positions(word_index[token], low, high)[:180]:
            start_idx = pos - tok_pos
            if start_idx < low or start_idx >= high:
                continue
            if start_idx not in seen:
                starts.append(start_idx)
                seen.add(start_idx)

    step = 1 if len(tokens) <= 3 else 2 if len(tokens) <= 8 else 3
    radius = max(24, len(tokens) * 4)
    for delta in range(-radius, radius + 1, step):
        start_idx = approx_start + delta
        if start_idx < low or start_idx >= high:
            continue
        if start_idx not in seen:
            starts.append(start_idx)
            seen.add(start_idx)

    if low not in seen:
        starts.append(low)
    return starts


def refine_span(
    tokens: list[str],
    words: list[Word],
    word_index: dict[str, list[int]],
    low: int,
    high: int,
    approx_start: int,
    kind: str,
) -> Optional[MatchSpan]:
    if not tokens:
        return None
    if high <= low:
        return None

    best: Optional[MatchSpan] = None
    token_count = len(tokens)
    min_len = max(1, token_count - 3)

    for start_idx in candidate_starts(tokens, word_index, low, high, approx_start)[:220]:
        max_len = min(high - start_idx, max(token_count + 8, int(math.ceil(token_count * 1.8))))
        if max_len <= 0:
            continue
        for span_len in range(min_len, max_len + 1):
            end_idx = start_idx + span_len
            candidate_tokens = [word.norm for word in words[start_idx:end_idx]]
            score = score_candidate(tokens, candidate_tokens)
            score -= gap_penalty(words, start_idx, end_idx) * 3.2
            score -= abs(start_idx - approx_start) * 0.05
            if best is None or score > best.score:
                best = MatchSpan(start_idx=start_idx, end_idx=end_idx, score=score, kind=kind)

    if best is None or best.score < line_threshold(token_count):
        return None
    return best


def local_alignment_blocks(tokens: list[str], candidate_tokens: list[str]) -> tuple[int, int]:
    import difflib

    matcher = difflib.SequenceMatcher(None, tokens, candidate_tokens, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size > 0]
    if not blocks:
        return 0, 0
    first = blocks[0]
    last = blocks[-1]
    return first.a, len(tokens) - (last.a + last.size)


def span_to_ms(span: MatchSpan, tokens: list[str], words: list[Word]) -> tuple[int, int]:
    start_ms = int(round(words[span.start_idx].start * 1000))
    end_ms = int(round(words[span.end_idx - 1].end * 1000))

    gap_before_ms = 0
    if span.start_idx > 0:
        gap_before_ms = int(round((words[span.start_idx].start - words[span.start_idx - 1].end) * 1000))
    gap_after_ms = 0
    if span.end_idx < len(words):
        gap_after_ms = int(round((words[span.end_idx].start - words[span.end_idx - 1].end) * 1000))

    candidate_tokens = [word.norm for word in words[span.start_idx : span.end_idx]]
    missing_prefix, missing_suffix = local_alignment_blocks(tokens, candidate_tokens)

    start_pad = min(140, max(0, gap_before_ms // 4))
    start_pad += min(260, missing_prefix * 95)
    end_pad = min(180, max(0, gap_after_ms // 4))
    end_pad += min(300, missing_suffix * 110)

    start_ms = max(0, start_ms - start_pad)
    end_ms = max(start_ms + 220, end_ms + end_pad)
    return start_ms, end_ms


def line_weight(tokens: list[str]) -> int:
    return max(1, len(tokens))


def max_interpolated_duration_ms(weight: int) -> int:
    return max(900, min(9000, 800 + weight * 260))


def boundary_start_ms(words: list[Word], start_idx: int) -> int:
    if start_idx <= 0:
        return 0
    return int(round(words[start_idx - 1].end * 1000))


def boundary_end_ms(words: list[Word], end_idx: int) -> int:
    if end_idx >= len(words):
        return int(round(words[-1].end * 1000))
    return int(round(words[end_idx].start * 1000))


def assign_weighted_times(
    line_indices: list[int],
    dialogues: list[DialogueLine],
    output_times: list[Optional[tuple[int, int]]],
    start_ms: int,
    end_ms: int,
) -> None:
    if not line_indices:
        return
    weights = [line_weight(dialogues[idx].tokens) for idx in line_indices]
    total_weight = sum(weights)
    if end_ms <= start_ms:
        cursor = start_ms
        for idx, weight in zip(line_indices, weights):
            duration = max(320, min(2400, 260 + weight * 85))
            output_times[idx] = (cursor, cursor + duration)
            cursor += max(240, duration - 120)
        return

    window_ms = end_ms - start_ms
    reasonable_total = sum(max_interpolated_duration_ms(weight) for weight in weights)

    if window_ms <= int(reasonable_total * 1.15):
        prefix = 0
        for idx, weight in zip(line_indices, weights):
            seg_start = start_ms + int(round(window_ms * (prefix / total_weight)))
            prefix += weight
            seg_end = start_ms + int(round(window_ms * (prefix / total_weight)))
            output_times[idx] = (seg_start, max(seg_start + 220, seg_end))
        return

    tentative: list[tuple[int, int, int]] = []
    prefix = 0
    for idx, weight in zip(line_indices, weights):
        center = start_ms + int(round(window_ms * ((prefix + weight / 2) / total_weight)))
        share = int(round(window_ms * (weight / total_weight)))
        duration = min(max_interpolated_duration_ms(weight), max(320, min(share, share // 2 + 650)))
        tentative.append((idx, center, duration))
        prefix += weight

    cursor = start_ms
    packed: list[tuple[int, int, int]] = []
    for idx, center, duration in tentative:
        seg_start = center - duration // 2
        seg_start = max(start_ms, seg_start, cursor - 120)
        seg_end = seg_start + duration
        if seg_end > end_ms:
            seg_end = end_ms
            seg_start = max(start_ms, seg_end - duration)
        if seg_end <= seg_start:
            seg_end = seg_start + 220
        packed.append((idx, seg_start, seg_end))
        cursor = seg_end

    next_start = end_ms
    for idx, seg_start, seg_end in reversed(packed):
        if seg_end > next_start + 120:
            seg_end = next_start + 120
            seg_start = min(seg_start, seg_end - 220)
        output_times[idx] = (max(start_ms, seg_start), max(seg_start + 220, seg_end))
        next_start = output_times[idx][0]


def fill_gap_block(
    start_line: int,
    end_line: int,
    left_line: Optional[int],
    right_line: Optional[int],
    dialogues: list[DialogueLine],
    words: list[Word],
    word_index: dict[str, list[int]],
    spans: list[Optional[MatchSpan]],
    output_times: list[Optional[tuple[int, int]]],
) -> None:
    if start_line > end_line:
        return

    low_word = spans[left_line].end_idx if left_line is not None and spans[left_line] else 0
    high_word = spans[right_line].start_idx if right_line is not None and spans[right_line] else len(words)
    low_word = max(0, low_word)
    high_word = min(len(words), max(low_word, high_word))

    block_lines = list(range(start_line, end_line + 1))
    block_weight_total = sum(line_weight(dialogues[idx].tokens) for idx in block_lines) or 1
    matched_lines: list[int] = []
    cursor = low_word

    for line_idx in block_lines:
        tokens = dialogues[line_idx].tokens
        if len(tokens) < 3:
            continue
        before_weight = sum(line_weight(dialogues[idx].tokens) for idx in range(start_line, line_idx))
        approx_word = low_word + int(round((high_word - low_word) * (before_weight / block_weight_total)))
        local_low = max(cursor, low_word)
        match = refine_span(tokens, words, word_index, local_low, high_word, approx_word, kind="fuzzy")
        if match is None:
            continue
        spans[line_idx] = match
        output_times[line_idx] = span_to_ms(match, tokens, words)
        cursor = max(cursor, match.end_idx)
        matched_lines.append(line_idx)

    segment_boundaries: list[tuple[Optional[int], Optional[int]]] = []
    current_left = left_line
    for line_idx in matched_lines:
        segment_boundaries.append((current_left, line_idx))
        current_left = line_idx
    segment_boundaries.append((current_left, right_line))

    for segment_left, segment_right in segment_boundaries:
        missing: list[int] = []
        missing_start = (segment_left + 1) if segment_left is not None else start_line
        missing_end = (segment_right - 1) if segment_right is not None else end_line
        if missing_start <= missing_end:
            for line_idx in range(missing_start, missing_end + 1):
                if output_times[line_idx] is None:
                    missing.append(line_idx)
        if not missing:
            continue

        left_ms = output_times[segment_left][1] if segment_left is not None and output_times[segment_left] else boundary_start_ms(words, low_word)
        right_ms = output_times[segment_right][0] if segment_right is not None and output_times[segment_right] else boundary_end_ms(words, high_word)
        assign_weighted_times(missing, dialogues, output_times, left_ms, right_ms)


def smooth_times(dialogues: list[DialogueLine], output_times: list[Optional[tuple[int, int]]]) -> list[tuple[int, int]]:
    finalized: list[tuple[int, int]] = []
    prev_end = 0
    for line_idx, slot in enumerate(output_times):
        if slot is None:
            start_ms = prev_end
            end_ms = start_ms + 700
        else:
            start_ms, end_ms = slot
        min_duration = max(260, min(3200, 280 + line_weight(dialogues[line_idx].tokens) * 80))
        if start_ms < prev_end - 20:
            start_ms = prev_end - 20
        if end_ms < start_ms + min_duration:
            end_ms = start_ms + min_duration
        if line_idx + 1 < len(output_times) and output_times[line_idx + 1] is not None:
            next_start = output_times[line_idx + 1][0]
            if next_start > start_ms + 180:
                end_ms = min(end_ms, next_start + 120)
                end_ms = max(end_ms, start_ms + 220)
        finalized.append((max(0, int(start_ms)), max(int(start_ms) + 220, int(end_ms))))
        prev_end = finalized[-1][1]
    return finalized


def ass_time(ms: int) -> str:
    centiseconds = max(0, int(round(ms / 10.0)))
    seconds, cs = divmod(centiseconds, 100)
    minutes, sec = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours}:{minute:02d}:{sec:02d}.{cs:02d}"


def rewrite_ass(
    source_lines: list[str],
    dialogues: list[DialogueLine],
    times: list[tuple[int, int]],
    output_path: Path,
) -> None:
    rewritten = list(source_lines)
    for dialogue, (start_ms, end_ms) in zip(dialogues, times):
        fields = list(dialogue.fields)
        fields[1] = ass_time(start_ms)
        fields[2] = ass_time(end_ms)
        rewritten[dialogue.file_line_idx] = ",".join(fields)
    output_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def retime_ass(input_ass: Path, words_json: Path, output_ass: Path) -> dict[str, object]:
    source_lines, dialogues = parse_ass(input_ass)
    if not dialogues:
        raise RuntimeError("ASS 里没有可处理的 Dialogue 行。")
    words = load_words(words_json)
    ngram_index = build_ngram_index(words)
    word_index = build_word_index(words)

    raw_candidates = [
        cand
        for idx, dialogue in enumerate(dialogues)
        if (cand := find_anchor_candidate(idx, dialogue.tokens, ngram_index)) is not None
    ]
    anchor_chain = select_monotonic_anchors(raw_candidates)

    spans: list[Optional[MatchSpan]] = [None] * len(dialogues)
    output_times: list[Optional[tuple[int, int]]] = [None] * len(dialogues)

    kept_anchors = 0
    skipped_conflicts = 0
    prev_anchor_end = 0
    prev_anchor_line = -1
    for anchor in anchor_chain:
        tokens = dialogues[anchor.line_idx].tokens
        if not tokens:
            continue
        low = max(prev_anchor_end, anchor.estimated_start - 32)
        high = min(len(words), max(low + 1, anchor.estimated_end + 42))
        match = refine_span(tokens, words, word_index, low, high, anchor.estimated_start, kind="anchor")
        if match is None:
            continue
        if anchor.line_idx <= prev_anchor_line or match.start_idx < prev_anchor_end:
            skipped_conflicts += 1
            continue
        spans[anchor.line_idx] = match
        output_times[anchor.line_idx] = span_to_ms(match, tokens, words)
        prev_anchor_end = match.end_idx
        prev_anchor_line = anchor.line_idx
        kept_anchors += 1

    known_lines = [idx for idx, span in enumerate(spans) if span is not None]
    last_known = None
    for line_idx in known_lines:
        if last_known is not None and last_known + 1 <= line_idx - 1:
            fill_gap_block(
                last_known + 1,
                line_idx - 1,
                last_known,
                line_idx,
                dialogues,
                words,
                word_index,
                spans,
                output_times,
            )
        last_known = line_idx

    if known_lines:
        first_known = known_lines[0]
        if first_known > 0:
            fill_gap_block(
                0,
                first_known - 1,
                None,
                first_known,
                dialogues,
                words,
                word_index,
                spans,
                output_times,
            )
        last_known = known_lines[-1]
        if last_known < len(dialogues) - 1:
            fill_gap_block(
                last_known + 1,
                len(dialogues) - 1,
                last_known,
                None,
                dialogues,
                words,
                word_index,
                spans,
                output_times,
            )
    else:
        fill_gap_block(
            0,
            len(dialogues) - 1,
            None,
            None,
            dialogues,
            words,
            word_index,
            spans,
            output_times,
        )

    final_times = smooth_times(dialogues, output_times)
    rewrite_ass(source_lines, dialogues, final_times, output_ass)

    matched_lines = sum(1 for span in spans if span is not None)
    fuzzy_lines = sum(1 for span in spans if span is not None and span.kind == "fuzzy")
    anchor_lines = sum(1 for span in spans if span is not None and span.kind == "anchor")
    return {
        "input": str(input_ass),
        "words_json": str(words_json),
        "output": str(output_ass),
        "total_lines": len(dialogues),
        "raw_anchor_candidates": len(raw_candidates),
        "kept_anchors": kept_anchors,
        "skipped_anchor_conflicts": skipped_conflicts,
        "matched_lines": matched_lines,
        "anchor_lines": anchor_lines,
        "fuzzy_lines": fuzzy_lines,
        "interpolated_lines": len(dialogues) - matched_lines,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retime ASS subtitles using word timestamps without trusting original subtitle times.")
    parser.add_argument("input_ass", type=Path)
    parser.add_argument("words_json", type=Path)
    parser.add_argument("output_ass", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = retime_ass(args.input_ass, args.words_json, args.output_ass)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
