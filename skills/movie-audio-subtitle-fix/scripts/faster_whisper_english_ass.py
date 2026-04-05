#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path

from _skill_paths import find_transcriber_dir


THIS_FILE = Path(__file__).resolve()
SKILL_DIR = THIS_FILE.parents[1]
TRANSCRIBER_DIR = find_transcriber_dir(SKILL_DIR)


def maybe_reexec_for_cuda() -> None:
    device = (os.getenv("ALIGN_DEVICE") or "cpu").strip().lower()
    if device != "cuda":
        return
    if os.getenv("FW_CUDA_BOOTSTRAPPED") == "1":
        return

    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(sys.prefix) / "lib" / pyver / "site-packages"
    extra_dirs = [
        site_packages / "nvidia" / "cublas" / "lib",
        site_packages / "nvidia" / "cudnn" / "lib",
        site_packages / "nvidia" / "cuda_nvrtc" / "lib",
        Path("/usr/lib/wsl/lib"),
    ]
    existing = [str(path) for path in extra_dirs if path.exists()]
    if not existing:
        return

    env = os.environ.copy()
    current = env.get("LD_LIBRARY_PATH", "")
    prefix = ":".join(existing)
    env["LD_LIBRARY_PATH"] = prefix if not current else prefix + ":" + current
    env["FW_CUDA_BOOTSTRAPPED"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


maybe_reexec_for_cuda()

sys.path.insert(0, str(TRANSCRIBER_DIR))
from backend.timing_aligner import build_model, normalize_text  # noqa: E402


TAG_RE = re.compile(r"\s+")


def clean_segment_text(text: str) -> str:
    cleaned = (text or "").replace("\r", " ").replace("\n", " ").strip()
    cleaned = TAG_RE.sub(" ", cleaned)
    return cleaned


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


def render_ass(cues: list[dict]) -> str:
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
    events = [
        "Dialogue: 0,{start},{end},Default,,0,0,0,,{text}".format(
            start=format_ass_timestamp(cue["start_ms"]),
            end=format_ass_timestamp(cue["end_ms"]),
            text=escape_ass_text(cue["text"]),
        )
        for cue in cues
    ]
    return header + "\n" + "\n".join(events).strip() + "\n"


def default_output_path(input_audio: Path) -> Path:
    return input_audio.with_name(f"{input_audio.stem}_fw_en.ass")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an English ASS subtitle directly from audio using faster-whisper."
    )
    parser.add_argument("input", type=Path, help="Input audio/video path")
    parser.add_argument("--output", type=Path, default=None, help="Output ASS path")
    parser.add_argument("--words-json", type=Path, default=None, help="Output word timestamp cache path")
    parser.add_argument("--model", default="medium.en", help="Whisper model, default medium.en")
    parser.add_argument("--language", default="en", help="Language code, default en")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size, default 5")
    parser.add_argument("--vad-filter", action="store_true", help="Enable VAD filtering")
    parser.add_argument(
        "--condition-on-previous-text",
        action="store_true",
        help="Enable conditioning on previous text",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input.expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_path = (args.output or default_output_path(input_path)).expanduser()
    words_json = (
        args.words_json.expanduser()
        if args.words_json
        else output_path.with_suffix(f".{args.model}.words.json")
    )

    model = build_model(args.model)
    transcribe_kwargs = {
        "beam_size": args.beam_size,
        "word_timestamps": True,
        "vad_filter": args.vad_filter,
        "condition_on_previous_text": args.condition_on_previous_text,
        "language": args.language,
    }
    segments, info = model.transcribe(str(input_path), **transcribe_kwargs)

    cues: list[dict] = []
    words: list[dict] = []
    for segment in segments:
        text = clean_segment_text(segment.text)
        if not text:
            continue
        start = getattr(segment, "start", None)
        end = getattr(segment, "end", None)
        if start is None or end is None:
            continue
        start_ms = int(round(float(start) * 1000))
        end_ms = int(round(float(end) * 1000))
        if end_ms <= start_ms:
            continue
        cues.append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": text,
            }
        )
        for word in segment.words or []:
            if word.start is None or word.end is None:
                continue
            norm = normalize_text(word.word)
            if not norm:
                continue
            words.append(
                {
                    "text": word.word,
                    "norm": norm,
                    "start": float(word.start),
                    "end": float(word.end),
                }
            )

    if not cues:
        raise RuntimeError("No subtitle cues were generated from faster-whisper.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_ass(cues), encoding="utf-8")
    words_json.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "words_json": str(words_json),
                "cue_count": len(cues),
                "word_count": len(words),
                "language": getattr(info, "language", args.language),
                "duration": getattr(info, "duration", None),
                "model": args.model,
                "vad_filter": args.vad_filter,
                "condition_on_previous_text": args.condition_on_previous_text,
                "device": (os.getenv("ALIGN_DEVICE") or "cpu").strip().lower(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
