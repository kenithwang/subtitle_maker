#!/usr/bin/env python3
import argparse
import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path

from _skill_paths import find_transcriber_dir


SKILL_DIR = Path(__file__).resolve().parents[1]
TRANSCRIBER_DIR = find_transcriber_dir(SKILL_DIR)
ORDER_ONLY_PATH = SKILL_DIR / "scripts" / "order_only_retime_ass.py"


def load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return

    candidate_paths = [
        Path.cwd() / ".env.local",
        Path.cwd() / ".env",
        Path.home() / ".config" / "ai-video-transcriber" / ".env",
        TRANSCRIBER_DIR / ".env.local",
        TRANSCRIBER_DIR / ".env",
        SKILL_DIR / ".env.local",
        SKILL_DIR / ".env",
    ]
    for path in candidate_paths:
        if path.exists():
            load_dotenv(path, override=False)
            break


def ensure_transcriber_dir(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"未找到 AI-Video-Transcriber: {path}")
    if not (path / "cli.py").exists():
        raise RuntimeError(f"目录不是 AI-Video-Transcriber 工程: {path}")


def import_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def guess_language_from_ass(path: Path) -> str | None:
    sample_lines: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw.startswith("Dialogue:"):
            continue
        fields = raw.split(",", 9)
        if len(fields) != 10:
            continue
        text = fields[9]
        plain = text.replace(r"\N", "\n")
        plain = plain.replace("{", " ").replace("}", " ")
        first = plain.splitlines()[0].strip()
        if not first:
            continue
        sample_lines.append(first)
        if len("\n".join(sample_lines)) >= 800:
            break
    text = "\n".join(sample_lines)
    total = len(text) or 1
    hiragana = sum(1 for ch in text if "\u3040" <= ch <= "\u309f")
    katakana = sum(1 for ch in text if "\u30a0" <= ch <= "\u30ff")
    kanji = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    en = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if (hiragana + katakana) / total > 0.05:
        return "ja"
    if kanji / total > 0.2:
        return "zh"
    if en / total > 0.2:
        return "en"
    return None


async def generate_raw_ass(
    *,
    input_path: Path,
    outdir: Path,
    title: str | None,
    subtitle_mode: str,
    model: str | None,
    segment_seconds: int | None,
    parallelism: int | None,
    transcriber_dir: Path,
) -> dict:
    sys.path.insert(0, str(transcriber_dir))
    try:
        if model:
            os.environ["GEMINI_MODEL"] = model

        from backend.media_tools import ensure_media_tools
        from backend.subtitle_pipeline import process_local_media_to_subtitles

        ensure_media_tools()
        return await process_local_media_to_subtitles(
            input_path=input_path,
            temp_dir=outdir,
            title=title,
            keep_audio=True,
            subtitle_mode=subtitle_mode,
            subtitle_format="ass",
            # The follow-up order-only retimer ignores original subtitle times and only
            # needs stable text ordering, so avoid asking Gemini for precise timestamps.
            timing_mode="placeholder",
            align_model=None,
            segment_seconds=segment_seconds,
            parallelism=parallelism,
            on_update=None,
        )
    finally:
        if sys.path and sys.path[0] == str(transcriber_dir):
            sys.path.pop(0)


def ensure_words_json(
    *,
    raw_ass_path: Path,
    audio_path: Path,
    align_model: str,
    language: str | None,
    words_json: Path,
    transcriber_dir: Path,
) -> dict:
    sys.path.insert(0, str(transcriber_dir))
    try:
        from backend.timing_aligner import transcribe_words

        words = transcribe_words(audio_path, align_model, words_json, language=language)
        return {
            "path": str(words_json),
            "word_count": len(words),
            "language": language,
            "align_model": align_model,
            "raw_ass": str(raw_ass_path),
        }
    finally:
        if sys.path and sys.path[0] == str(transcriber_dir):
            sys.path.pop(0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legacy Gemini transcript workflow: generate a raw ASS and then retime it with order-only alignment.")
    parser.add_argument("input", type=Path, help="输入视频文件")
    parser.add_argument("--allow-gemini-transcript", action="store_true", help="显式允许旧的 Gemini 分片转录流程")
    parser.add_argument("--outdir", type=Path, default=None, help="输出目录，默认和输入视频同目录")
    parser.add_argument("--title", default=None, help="覆盖字幕标题")
    parser.add_argument("--subtitle-mode", choices=("bilingual-zh", "monolingual"), default="bilingual-zh")
    parser.add_argument("--model", default=None, help="覆盖 GEMINI_MODEL")
    parser.add_argument("--segment-seconds", type=int, default=None, help="传给 AI-Video-Transcriber 的分片长度")
    parser.add_argument("--parallelism", type=int, default=None, help="传给 AI-Video-Transcriber 的并行度")
    parser.add_argument("--align-model", default=os.getenv("ALIGN_MODEL") or "medium.en", help="faster-whisper 对齐模型，默认 medium.en")
    parser.add_argument("--language", default=None, help="非英语素材可显式指定语言，如 en / zh / ja")
    parser.add_argument("--output", type=Path, default=None, help="重对时后的 ASS 输出路径，默认 *_order_only.ass")
    parser.add_argument("--words-json", type=Path, default=None, help="词级时间戳缓存路径，默认和原始 ASS 同名")
    parser.add_argument("--keep-audio", action="store_true", help="保留中间音频文件")
    parser.add_argument("--transcriber-dir", type=Path, default=TRANSCRIBER_DIR, help="AI-Video-Transcriber 路径")
    return parser


def main() -> None:
    load_dotenv_if_present()
    args = build_parser().parse_args()

    if not args.allow_gemini_transcript:
        raise RuntimeError(
            "旧的 Gemini 分片转录流程已降级为 legacy。"
            "默认改用 scripts/prepare_english_subtitle_source.py 先生成整片 faster-whisper 英文底稿，"
            "再用 GPT-5.4 产出离线 zh.tsv，最后用 scripts/translate_ass_preserve_timing.py 回填双语 ASS。"
            "只有在用户明确要求旧流程时，才传 --allow-gemini-transcript。"
        )

    input_path = args.input.expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("未设置 GEMINI_API_KEY，无法生成原始字幕。")

    transcriber_dir = args.transcriber_dir.expanduser()
    ensure_transcriber_dir(transcriber_dir)

    outdir = (args.outdir or input_path.parent).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    transcriber_result = asyncio.run(
        generate_raw_ass(
            input_path=input_path,
            outdir=outdir,
            title=args.title,
            subtitle_mode=args.subtitle_mode,
            model=args.model,
            segment_seconds=args.segment_seconds,
            parallelism=args.parallelism,
            transcriber_dir=transcriber_dir,
        )
    )

    subtitle_file = transcriber_result.get("subtitle_file")
    audio_file = transcriber_result.get("audio_file")
    if not subtitle_file or not audio_file:
        raise RuntimeError(f"AI-Video-Transcriber 没有返回必要输出: {transcriber_result}")

    raw_ass_path = outdir / subtitle_file
    audio_path = Path(audio_file)
    if not raw_ass_path.exists():
        raise RuntimeError(f"原始 ASS 不存在: {raw_ass_path}")
    if not audio_path.exists():
        raise RuntimeError(f"中间音频不存在: {audio_path}")

    inferred_language = args.language or guess_language_from_ass(raw_ass_path)
    words_json = (args.words_json or raw_ass_path.with_suffix(f".{args.align_model}.words.json")).expanduser()
    words_json.parent.mkdir(parents=True, exist_ok=True)
    words_info = ensure_words_json(
        raw_ass_path=raw_ass_path,
        audio_path=audio_path,
        align_model=args.align_model,
        language=inferred_language,
        words_json=words_json,
        transcriber_dir=transcriber_dir,
    )

    order_only = import_from_path("order_only_retime_ass_skill", ORDER_ONLY_PATH)
    output_ass = (args.output or raw_ass_path.with_name(f"{raw_ass_path.stem}_order_only{raw_ass_path.suffix}")).expanduser()
    output_ass.parent.mkdir(parents=True, exist_ok=True)
    retime_stats = order_only.retime_ass(raw_ass_path, words_json, output_ass)

    audio_kept = args.keep_audio
    if not args.keep_audio:
        audio_path.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "input": str(input_path),
                "raw_ass": str(raw_ass_path),
                "retimed_ass": str(output_ass),
                "words_json": str(words_json),
                "audio_file": str(audio_path) if audio_kept else None,
                "audio_deleted": not audio_kept,
                "detected_language": inferred_language,
                "transcriber": transcriber_result,
                "words": words_info,
                "retime_stats": retime_stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
