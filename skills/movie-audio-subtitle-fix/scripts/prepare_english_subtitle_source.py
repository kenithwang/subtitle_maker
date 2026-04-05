#!/usr/bin/env python3
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
SKILL_DIR = THIS_FILE.parents[1]
SCRIPTS_DIR = THIS_FILE.parent
FIX_AUDIO_PATH = SCRIPTS_DIR / "fix_movie_audio.py"
FW_SCRIPT_PATH = SCRIPTS_DIR / "faster_whisper_english_ass.py"

VIDEO_SUFFIXES = {
    ".mkv",
    ".mp4",
    ".mov",
    ".avi",
    ".m4v",
    ".ts",
    ".m2ts",
    ".webm",
}
DIRECT_AUDIO_SUFFIXES = {
    ".m4a",
    ".aac",
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
}


def import_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def input_is_direct_audio(path: Path) -> bool:
    return path.suffix.lower() in DIRECT_AUDIO_SUFFIXES


def default_audio_output(input_path: Path, outdir: Path) -> Path:
    return outdir / f"{input_path.stem}_subtitle_audio.m4a"


def default_ass_output(input_path: Path, outdir: Path) -> Path:
    return outdir / f"{input_path.stem}_fw_en.ass"


def default_source_lines_output(ass_path: Path) -> Path:
    return ass_path.with_suffix(".source.tsv")


def extract_transcription_audio(input_path: Path, output_path: Path) -> None:
    fix_audio = import_from_path("fix_movie_audio_skill", FIX_AUDIO_PATH)
    ffmpeg_bin = fix_audio.get_ffmpeg_bin()
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        fix_audio.normalize_arg(str(input_path), ffmpeg_bin),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        fix_audio.normalize_arg(str(output_path), ffmpeg_bin),
    ]
    subprocess.check_call(cmd)


def write_source_lines_tsv(ass_path: Path, output_path: Path) -> int:
    count = 0
    lines: list[str] = []
    for raw in ass_path.read_text(encoding="utf-8-sig").splitlines():
        if not raw.startswith("Dialogue:"):
            continue
        fields = raw.split(",", 9)
        if len(fields) != 10:
            continue
        count += 1
        english = fields[9].split(r"\N", 1)[0].strip()
        lines.append(f"{count}\t{english}")
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Best-practice subtitle source preparation: optional audio extraction + full faster-whisper English ASS."
    )
    parser.add_argument("input", type=Path, help="输入视频或音频文件")
    parser.add_argument("--outdir", type=Path, default=None, help="输出目录，默认和输入文件同目录")
    parser.add_argument("--audio-output", type=Path, default=None, help="转录音频输出路径，默认 *_subtitle_audio.m4a")
    parser.add_argument("--output", type=Path, default=None, help="英文 ASS 输出路径，默认 *_fw_en.ass")
    parser.add_argument("--words-json", type=Path, default=None, help="词级时间戳 JSON 输出路径")
    parser.add_argument("--source-lines", type=Path, default=None, help="导出的 id<TAB>英文原句 TSV，供 GPT-5.4 翻译")
    parser.add_argument("--model", default="medium.en", help="Whisper 模型，默认 medium.en")
    parser.add_argument("--language", default="en", help="语言代码，默认 en")
    parser.add_argument("--beam-size", type=int, default=5, help="beam size，默认 5")
    parser.add_argument("--vad-filter", action="store_true", help="启用 VAD")
    parser.add_argument("--condition-on-previous-text", action="store_true", help="启用跨段上下文")
    parser.add_argument("--keep-audio", action="store_true", help="输入是视频时保留提取出来的转录音频")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input.expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    outdir = (args.outdir or input_path.parent).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    ass_output = (args.output or default_ass_output(input_path, outdir)).expanduser()
    words_json = (
        args.words_json.expanduser()
        if args.words_json
        else ass_output.with_suffix(f".{args.model}.words.json")
    )
    source_lines = (args.source_lines or default_source_lines_output(ass_output)).expanduser()

    generated_audio = False
    if input_is_direct_audio(input_path):
        audio_input = input_path
    else:
        audio_input = (args.audio_output or default_audio_output(input_path, outdir)).expanduser()
        audio_input.parent.mkdir(parents=True, exist_ok=True)
        extract_transcription_audio(input_path, audio_input)
        generated_audio = True

    fw_cmd = [
        sys.executable,
        str(FW_SCRIPT_PATH),
        str(audio_input),
        "--output",
        str(ass_output),
        "--words-json",
        str(words_json),
        "--model",
        args.model,
        "--language",
        args.language,
        "--beam-size",
        str(args.beam_size),
    ]
    if args.vad_filter:
        fw_cmd.append("--vad-filter")
    if args.condition_on_previous_text:
        fw_cmd.append("--condition-on-previous-text")
    subprocess.check_call(fw_cmd)

    source_count = write_source_lines_tsv(ass_output, source_lines)

    deleted_audio = False
    if generated_audio and not args.keep_audio:
        audio_input.unlink(missing_ok=True)
        deleted_audio = True

    print(
        json.dumps(
            {
                "input": str(input_path),
                "english_ass": str(ass_output),
                "words_json": str(words_json),
                "source_lines_tsv": str(source_lines),
                "audio_input": None if deleted_audio else str(audio_input),
                "audio_deleted": deleted_audio,
                "generated_audio": generated_audio,
                "source_line_count": source_count,
                "model": args.model,
                "language": args.language,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
