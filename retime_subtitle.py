#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path

from backend.subtitle_retimer import retime_subtitle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对现有 ASS/SRT 字幕做本地强制重定时")
    parser.add_argument("--subtitle", type=Path, required=True, help="已有字幕文件路径（.ass / .srt）")
    parser.add_argument("--audio", type=Path, required=True, help="用于对齐的音频文件路径")
    parser.add_argument("--output", type=Path, default=None, help="输出字幕路径（默认在原文件名后追加 _retimed）")
    parser.add_argument("--align-model", default="medium.en", help="Whisper 对齐模型，默认 medium.en")
    parser.add_argument("--language", default=None, help="可选，显式指定语言，例如 en / zh / ja")
    parser.add_argument("--cache", type=Path, default=None, help="词级时间戳缓存路径")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = build_parser().parse_args()
    result = retime_subtitle(
        args.subtitle,
        args.audio,
        output_path=args.output,
        align_model=args.align_model,
        cache_path=args.cache,
        language=args.language,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
