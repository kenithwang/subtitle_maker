#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from backend.env_loader import load_env_if_present


def _load_dotenv_if_present() -> None:
    path = load_env_if_present(Path(__file__).resolve().parent)
    if path:
        print(f"[i] 已加载环境文件: {path}")


def _require_api_key() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        print("[!] 未设置 GEMINI_API_KEY，无法生成字幕。", file=sys.stderr)
        sys.exit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI 字幕制作器（CLI）")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input", type=Path, help="本地视频或音频文件路径")
    source_group.add_argument("--url", help="单个在线视频链接（如 YouTube/Bilibili）")
    source_group.add_argument("--urls", nargs="+", help="多个在线视频链接，按顺序串行处理")

    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="输出目录（默认：本地输入时为同目录，URL 模式为 temp）",
    )
    parser.add_argument("--title", help="本地文件模式下覆盖输出标题")
    parser.add_argument(
        "--mode",
        choices=("bilingual-zh", "monolingual"),
        default=os.getenv("SUBTITLE_MODE", "bilingual-zh"),
        help="字幕模式：双语中英或仅原文",
    )
    parser.add_argument(
        "--format",
        choices=("srt", "ass"),
        default=os.getenv("SUBTITLE_FORMAT", "srt").lower(),
        help="字幕格式：SRT 或 ASS",
    )
    parser.add_argument(
        "--timing",
        choices=("gemini", "align", "placeholder"),
        default=os.getenv("SUBTITLE_TIMING", "align").lower(),
        help="时间轴策略：直接使用 Gemini 时间轴、使用占位时间轴，或使用本地 Whisper 二次强制对齐",
    )
    parser.add_argument(
        "--align-model",
        default=os.getenv("ALIGN_MODEL"),
        help="本地强制对齐模型，如 small.en、medium.en、small、medium",
    )
    parser.add_argument("--segment-seconds", type=int, default=None, help="每个字幕分片最长秒数（默认 1200）")
    parser.add_argument("--parallelism", type=int, default=None, help="并行处理的分片数")
    parser.add_argument("--keep-audio", action="store_true", help="保留中间音频文件")
    parser.add_argument("--model", help="覆盖 GEMINI_MODEL")
    parser.add_argument("--continue-on-error", action="store_true", help="批量 URL 模式下遇错继续")
    return parser


def _print_result(outdir: Path, result: dict) -> None:
    print("\n=== 字幕生成完成 ===")
    print(f"标题: {result.get('video_title')}")
    print(f"检测语言: {result.get('detected_language') or 'unknown'}")
    print("输出文件：")
    if result.get("subtitle_file"):
        print(f" - subtitle: {outdir / result['subtitle_file']}")
    if result.get("audio_file") and not result.get("audio_deleted"):
        print(f" - audio: {result['audio_file']}")

    warnings = result.get("warnings") or []
    if warnings:
        print("\n警告：")
        for item in warnings:
            print(f" - {item}")


async def _run_local(args: argparse.Namespace) -> None:
    from backend.subtitle_pipeline import process_local_media_to_subtitles

    async def on_update(evt: dict):
        print(f"[ {evt.get('progress', 0):>3}% ] {evt.get('message', '')}")

    result = await process_local_media_to_subtitles(
        input_path=args.input,
        temp_dir=args.outdir,
        title=args.title,
        keep_audio=args.keep_audio,
        subtitle_mode=args.mode,
        subtitle_format=args.format,
        timing_mode=args.timing,
        align_model=args.align_model,
        segment_seconds=args.segment_seconds,
        parallelism=args.parallelism,
        on_update=on_update,
    )
    _print_result(args.outdir, result)


async def _run_urls(args: argparse.Namespace) -> None:
    from backend.subtitle_pipeline import process_url_to_subtitles

    urls = args.urls or ([args.url] if args.url else [])
    total = len(urls)

    for index, url in enumerate(urls, start=1):
        print(f"\n=== 开始处理 {index}/{total} ===")

        async def on_update(evt: dict):
            print(f"[ {evt.get('progress', 0):>3}% ] {evt.get('message', '')}")

        try:
            result = await process_url_to_subtitles(
                url=url,
                temp_dir=args.outdir,
                keep_audio=args.keep_audio,
                subtitle_mode=args.mode,
                subtitle_format=args.format,
                timing_mode=args.timing,
                align_model=args.align_model,
                segment_seconds=args.segment_seconds,
                parallelism=args.parallelism,
                on_update=on_update,
            )
            _print_result(args.outdir, result)
        except Exception as exc:
            print(f"[!] 第 {index} 个链接处理失败: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                raise


def main() -> None:
    _load_dotenv_if_present()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    parser = build_parser()
    args = parser.parse_args()

    if args.model:
        os.environ["GEMINI_MODEL"] = args.model

    _require_api_key()

    from backend.media_tools import ensure_media_tools

    try:
        ffmpeg_bin, ffprobe_bin = ensure_media_tools()
    except Exception as exc:
        print(f"[!] {exc}", file=sys.stderr)
        sys.exit(2)

    logging.info("使用 ffmpeg: %s", ffmpeg_bin)
    logging.info("使用 ffprobe: %s", ffprobe_bin)

    if args.outdir is None:
        args.outdir = args.input.parent if args.input else Path("temp")

    args.outdir.mkdir(parents=True, exist_ok=True)

    try:
        if args.input:
            asyncio.run(_run_local(args))
        else:
            asyncio.run(_run_urls(args))
    except KeyboardInterrupt:
        print("\n已取消")
        sys.exit(130)
    except Exception as exc:
        print(f"[!] 处理失败: {exc}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
