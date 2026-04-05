#!/usr/bin/env python3
import argparse
import re
from dataclasses import dataclass
from pathlib import Path


RANGE_LINE_RE = re.compile(
    r"^\*{0,2}\s*"
    r"(?P<start>\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(?P<end>\d{1,2}:\d{2}(?::\d{2})?)"
    r"\s*\*{0,2}$"
)
INLINE_RANGE_RE = re.compile(
    r"^\*{0,2}\s*"
    r"(?P<start>\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(?P<end>\d{1,2}:\d{2}(?::\d{2})?)"
    r"\s*:\s*(?P<text>.+?)\s*\*{0,2}$"
)
BRACKET_LINE_RE = re.compile(r"^\[(?P<time>\d{2}:\d{2}:\d{2})\]\s*(?P<text>.+?)\s*$")
NOTE_RE = re.compile(r"^\*?\(注：", re.IGNORECASE)


@dataclass
class Cue:
    english: str
    chinese: str


def format_ass_timestamp(total_ms: int) -> str:
    total_ms = max(0, total_ms)
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    seconds = total_ms // 1000
    centiseconds = (total_ms % 1000) // 10
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def escape_ass_text(text: str) -> str:
    escaped_lines: list[str] = []
    for line in text.replace("\r", "").splitlines() or [""]:
        escaped = line.replace("\\", r"\\")
        escaped = escaped.replace("{", r"\{").replace("}", r"\}")
        escaped_lines.append(escaped)
    return r"\N".join(escaped_lines)


def clean_text_line(line: str) -> str:
    cleaned = line.strip()
    if cleaned.startswith("EN:"):
        cleaned = cleaned[3:].strip()
    elif cleaned.startswith("ZH:"):
        cleaned = cleaned[3:].strip()
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    return cleaned


def looks_like_time_header(line: str) -> bool:
    stripped = line.strip()
    return (
        RANGE_LINE_RE.match(stripped) is not None
        or INLINE_RANGE_RE.match(stripped) is not None
        or BRACKET_LINE_RE.match(stripped) is not None
    )


def should_skip_payload(english: str, chinese: str) -> bool:
    joined = "\n".join(part for part in (english, chinese) if part).strip().lower()
    if not joined:
        return True
    if "no speech detected" in joined and "audio is silent" in joined:
        return True
    if NOTE_RE.match(joined):
        return True
    return False


def build_cue(english: str, chinese: str) -> Cue | None:
    if not english:
        return None
    if should_skip_payload(english, chinese):
        return None
    return Cue(english=english, chinese=chinese)


def next_nonempty_index(lines: list[str], start: int) -> int | None:
    for idx in range(start, len(lines)):
        if lines[idx].strip():
            return idx
    return None


def parse_markdown_transcript(path: Path) -> list[Cue]:
    lines = path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n").splitlines()
    cues: list[Cue] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue

        inline_match = INLINE_RANGE_RE.match(line)
        if inline_match:
            english = clean_text_line(inline_match.group("text"))
            zh_idx = next_nonempty_index(lines, idx + 1)
            chinese = ""
            if zh_idx is not None and not looks_like_time_header(lines[zh_idx]):
                chinese = clean_text_line(lines[zh_idx])
                idx = zh_idx
            cue = build_cue(english, chinese)
            if cue is not None:
                cues.append(cue)
            idx += 1
            continue

        range_match = RANGE_LINE_RE.match(line)
        if range_match:
            en_idx = next_nonempty_index(lines, idx + 1)
            if en_idx is None:
                break
            english = clean_text_line(lines[en_idx])
            zh_idx = next_nonempty_index(lines, en_idx + 1)
            chinese = ""
            if zh_idx is not None and not looks_like_time_header(lines[zh_idx]):
                chinese = clean_text_line(lines[zh_idx])
                idx = zh_idx
            else:
                idx = en_idx
            cue = build_cue(english, chinese)
            if cue is not None:
                cues.append(cue)
            idx += 1
            continue

        bracket_match = BRACKET_LINE_RE.match(line)
        if bracket_match:
            english = clean_text_line(bracket_match.group("text"))
            chinese = ""
            next_idx = next_nonempty_index(lines, idx + 1)
            if next_idx is not None:
                next_match = BRACKET_LINE_RE.match(lines[next_idx].strip())
                if next_match and next_match.group("time") == bracket_match.group("time"):
                    chinese = clean_text_line(next_match.group("text"))
                    idx = next_idx
            cue = build_cue(english, chinese)
            if cue is not None:
                cues.append(cue)
            idx += 1
            continue

        idx += 1

    return cues


def placeholder_duration_ms(text: str) -> int:
    visible = text.replace(r"\N", "\n").replace("\r", "")
    word_count = len(re.findall(r"\S+", visible))
    line_count = max(1, len([line for line in visible.splitlines() if line.strip()]))
    duration = 650 + word_count * 260 + (line_count - 1) * 220
    return min(4200, max(900, duration))


def render_ass(cues: list[Cue]) -> str:
    header = "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "YCbCr Matrix: TV.601",
            "PlayResX: 1920",
            "PlayResY: 1080",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding",
            "Style: Default,Microsoft YaHei,50,&H00FFFFFF,&H0000FFFF,&H00141414,&H64000000,"
            "0,0,0,0,100,100,0,0,1,2.2,0,2,80,80,42,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )
    events: list[str] = []
    cursor = 0
    for cue in cues:
        text = cue.english
        if cue.chinese:
            text += r"\N" + cue.chinese
        start_ms = cursor
        end_ms = start_ms + placeholder_duration_ms(text)
        events.append(
            "Dialogue: 0,{start},{end},Default,,0,0,0,,{text}".format(
                start=format_ass_timestamp(start_ms),
                end=format_ass_timestamp(end_ms),
                text=escape_ass_text(text),
            )
        )
        cursor = end_ms + 80
    return header + "\n" + "\n".join(events).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a raw Markdown transcript into a bilingual ASS subtitle."
    )
    parser.add_argument("input", type=Path, help="Input Markdown transcript")
    parser.add_argument("output", type=Path, help="Output ASS path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input.expanduser()
    output_path = args.output.expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input transcript not found: {input_path}")

    cues = parse_markdown_transcript(input_path)
    if not cues:
        raise RuntimeError("No usable cues were parsed from the Markdown transcript.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_ass(cues), encoding="utf-8")
    print(
        {
            "input": str(input_path),
            "output": str(output_path),
            "cue_count": len(cues),
        }
    )


if __name__ == "__main__":
    main()
