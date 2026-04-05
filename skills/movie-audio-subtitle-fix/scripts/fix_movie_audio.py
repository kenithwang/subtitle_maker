#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


def resolve_binary(env_var: str, names: list[str], patterns: list[str]) -> str | None:
    custom = os.getenv(env_var)
    if custom:
        path = Path(custom).expanduser()
        if path.exists():
            return str(path)

    for name in names:
        resolved = shutil.which(name)
        if resolved:
            return resolved

    for pattern in patterns:
        matches = sorted(glob.glob(pattern), reverse=True)
        if matches:
            return matches[0]
    return None


def get_ffmpeg_bin() -> str:
    ffmpeg_bin = resolve_binary(
        "FFMPEG_BIN",
        ["ffmpeg", "ffmpeg.exe"],
        [
            "/mnt/c/ffmpeg/ffmpeg-*/bin/ffmpeg.exe",
            "/mnt/c/Users/*/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe",
        ],
    )
    if not ffmpeg_bin:
        raise RuntimeError("未找到 ffmpeg。请安装 FFmpeg，或通过 FFMPEG_BIN 指定路径。")
    return ffmpeg_bin


def get_ffprobe_bin() -> str:
    ffprobe_bin = resolve_binary(
        "FFPROBE_BIN",
        ["ffprobe", "ffprobe.exe"],
        [
            "/mnt/c/ffmpeg/ffmpeg-*/bin/ffprobe.exe",
            "/mnt/c/Users/*/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg*/ffmpeg-*/bin/ffprobe.exe",
        ],
    )
    if not ffprobe_bin:
        raise RuntimeError("未找到 ffprobe。请安装 FFmpeg，或通过 FFPROBE_BIN 指定路径。")
    return ffprobe_bin


def to_windows_path(value: str) -> str:
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", value)
    if not match:
        return value
    drive = match.group(1).upper()
    rest = match.group(2).replace("/", "\\")
    return f"{drive}:\\{rest}"


def normalize_arg(arg: str, binary: str) -> str:
    if binary.lower().endswith(".exe"):
        return to_windows_path(arg)
    return arg


def run_ffprobe(input_path: Path) -> dict:
    ffprobe_bin = get_ffprobe_bin()
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        normalize_arg(str(input_path), ffprobe_bin),
    ]
    raw = subprocess.check_output(cmd)
    return json.loads(raw.decode("utf-8", errors="ignore"))


def list_audio_tracks(input_path: Path) -> list[dict]:
    probe = run_ffprobe(input_path)
    tracks: list[dict] = []
    audio_idx = 0
    for stream in probe.get("streams", []):
        if stream.get("codec_type") != "audio":
            continue
        tags = stream.get("tags") or {}
        tracks.append(
            {
                "audio_track": audio_idx,
                "stream_index": stream.get("index"),
                "codec": stream.get("codec_name"),
                "channels": stream.get("channels"),
                "channel_layout": stream.get("channel_layout"),
                "sample_rate": stream.get("sample_rate"),
                "language": tags.get("language"),
                "title": tags.get("title"),
                "disposition": stream.get("disposition") or {},
            }
        )
        audio_idx += 1
    return tracks


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_aac_single_audio.mkv")


def convert_track(
    input_path: Path,
    output_path: Path,
    track_info: dict,
    *,
    audio_track: int,
    channels: int,
    sample_rate: int,
    bitrate: str,
) -> None:
    ffmpeg_bin = get_ffmpeg_bin()
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        normalize_arg(str(input_path), ffmpeg_bin),
        "-map",
        "0:v?",
        "-map",
        f"0:a:{audio_track}",
        "-map",
        "0:s?",
        "-map",
        "0:d?",
        "-map",
        "0:t?",
        "-map_metadata",
        "0",
        "-map_chapters",
        "0",
        "-c:v",
        "copy",
        "-c:s",
        "copy",
        "-c:d",
        "copy",
        "-c:t",
        "copy",
        "-c:a",
        "aac",
        "-profile:a",
        "aac_low",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-b:a",
        bitrate,
        "-disposition:a:0",
        "default",
    ]

    language = track_info.get("language")
    if language:
        cmd.extend(["-metadata:s:a:0", f"language={language}"])
    title = track_info.get("title")
    if title:
        cmd.extend(["-metadata:s:a:0", f"title={title}"])

    cmd.append(normalize_arg(str(output_path), ffmpeg_bin))
    subprocess.check_call(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcode one movie audio stream to a single AAC track for ACG Player compatibility.")
    parser.add_argument("input", type=Path, help="输入视频文件")
    parser.add_argument("--output", type=Path, default=None, help="输出文件路径，默认 *_aac_single_audio.mkv")
    parser.add_argument("--audio-track", type=int, default=0, help="要保留的音轨序号，从 0 开始")
    parser.add_argument("--channels", type=int, default=2, help="输出声道数，默认 2")
    parser.add_argument("--sample-rate", type=int, default=48000, help="输出采样率，默认 48000")
    parser.add_argument("--bitrate", default="192k", help="AAC 码率，默认 192k")
    parser.add_argument("--list-audio-tracks", action="store_true", help="只列出音轨信息，不转码")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input.expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    tracks = list_audio_tracks(input_path)
    if not tracks:
        raise RuntimeError("输入文件里没有音轨。")

    if args.list_audio_tracks:
        print(json.dumps({"input": str(input_path), "audio_tracks": tracks}, ensure_ascii=False, indent=2))
        return

    if args.audio_track < 0 or args.audio_track >= len(tracks):
        raise RuntimeError(f"音轨序号越界: {args.audio_track}，可选范围 0..{len(tracks) - 1}")

    output_path = (args.output or default_output_path(input_path)).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    track_info = tracks[args.audio_track]
    convert_track(
        input_path,
        output_path,
        track_info,
        audio_track=args.audio_track,
        channels=args.channels,
        sample_rate=args.sample_rate,
        bitrate=args.bitrate,
    )

    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "selected_audio_track": track_info,
                "codec": "aac",
                "channels": args.channels,
                "sample_rate": args.sample_rate,
                "bitrate": args.bitrate,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
