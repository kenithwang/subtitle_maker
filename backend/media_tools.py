import glob
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_binary(env_var: str, names: list[str], patterns: list[str]) -> str | None:
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


@lru_cache(maxsize=1)
def get_ffmpeg_bin() -> str | None:
    return _resolve_binary(
        env_var="FFMPEG_BIN",
        names=["ffmpeg", "ffmpeg.exe"],
        patterns=[
            "/mnt/c/ffmpeg/ffmpeg-*/bin/ffmpeg.exe",
            "/mnt/c/Users/*/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe",
        ],
    )


@lru_cache(maxsize=1)
def get_ffprobe_bin() -> str | None:
    return _resolve_binary(
        env_var="FFPROBE_BIN",
        names=["ffprobe", "ffprobe.exe"],
        patterns=[
            "/mnt/c/ffmpeg/ffmpeg-*/bin/ffprobe.exe",
            "/mnt/c/Users/*/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg*/ffmpeg-*/bin/ffprobe.exe",
        ],
    )


def ensure_media_tools() -> tuple[str, str]:
    ffmpeg_bin = get_ffmpeg_bin()
    ffprobe_bin = get_ffprobe_bin()

    missing: list[str] = []
    if not ffmpeg_bin:
        missing.append("ffmpeg")
    if not ffprobe_bin:
        missing.append("ffprobe")

    if missing:
        raise RuntimeError(
            "未找到媒体工具: "
            + ", ".join(missing)
            + "。请安装 FFmpeg，或通过 FFMPEG_BIN / FFPROBE_BIN 指定可执行文件路径。"
        )

    return ffmpeg_bin, ffprobe_bin


def ffmpeg_cmd(*args: str) -> list[str]:
    ffmpeg_bin, _ = ensure_media_tools()
    return [ffmpeg_bin, *[_normalize_arg_for_binary(arg, ffmpeg_bin) for arg in args]]


def ffprobe_cmd(*args: str) -> list[str]:
    _, ffprobe_bin = ensure_media_tools()
    return [ffprobe_bin, *[_normalize_arg_for_binary(arg, ffprobe_bin) for arg in args]]


def _normalize_arg_for_binary(arg: str, binary: str) -> str:
    if not binary.lower().endswith(".exe"):
        return arg
    return _to_windows_path(arg)


def _to_windows_path(value: str) -> str:
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", value)
    if not match:
        return value

    drive = match.group(1).upper()
    rest = match.group(2).replace("/", "\\")
    return f"{drive}:\\{rest}"


def check_media_tools() -> bool:
    try:
        subprocess.run(
            ffmpeg_cmd("-version"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        subprocess.run(
            ffprobe_cmd("-version"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        ffprobe_cmd(
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        )
    ).decode().strip()
    return float(out) if out else 0.0


def _get_transcription_audio_mode() -> str:
    mode = (os.getenv("TRANSCRIBE_AUDIO_MODE") or "vocals").strip().lower()
    if mode in {"vocals", "raw"}:
        return mode
    raise RuntimeError(
        f"不支持的 TRANSCRIBE_AUDIO_MODE: {mode}。可选值: vocals, raw"
    )


def _tail_text(text: str, max_lines: int = 20) -> str:
    lines = [line.rstrip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def _transcode_to_transcription_m4a(
    input_path: Path,
    output_path: Path,
    *,
    duration: float | None = None,
) -> None:
    ffmpeg_bin = get_ffmpeg_bin()
    if not ffmpeg_bin:
        raise RuntimeError("未找到 ffmpeg。")

    cmd = ffmpeg_cmd(
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        "-movflags",
        "+faststart",
    )
    if duration and duration > 0:
        cmd.extend(
            [
                "-af",
                f"apad=pad_dur={duration:.3f},atrim=0:{duration:.3f}",
            ]
        )
    cmd.append(_normalize_arg_for_binary(str(output_path), ffmpeg_bin))
    subprocess.check_call(cmd)


def _transcode_for_vocal_separation(input_path: Path, output_path: Path) -> None:
    subprocess.check_call(
        ffmpeg_cmd(
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        )
    )


def _run_demucs_vocals(input_path: Path, workdir: Path) -> Path:
    model_name = (os.getenv("DEMUCS_MODEL") or "htdemucs").strip() or "htdemucs"
    demucs_python = (os.getenv("DEMUCS_PYTHON") or sys.executable).strip() or sys.executable
    worker_script = Path(__file__).with_name("demucs_worker.py")
    output_path = workdir / "vocals.wav"
    cmd = [
        demucs_python,
        str(worker_script),
        str(input_path),
        str(output_path),
        "--model",
        model_name,
    ]
    device = (os.getenv("DEMUCS_DEVICE") or "").strip()
    if device:
        cmd.extend(["--device", device])
    segment = (os.getenv("DEMUCS_SEGMENT") or "").strip()
    if segment:
        cmd.extend(["--segment", segment])
    jobs = (os.getenv("DEMUCS_JOBS") or "").strip()
    if jobs:
        cmd.extend(["--jobs", jobs])

    env = os.environ.copy()
    demucs_bin_dir = str(Path(demucs_python).resolve().parent)
    env["PATH"] = demucs_bin_dir + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if result.returncode != 0:
        tail = _tail_text(result.stdout)
        raise RuntimeError(
            "Demucs 人声分离失败。请确认 DEMUCS_PYTHON 指向的解释器已安装 demucs / torch / torchaudio，"
            "或设置 TRANSCRIBE_AUDIO_MODE=raw 关闭人声分离。"
            + (f"\n{tail}" if tail else "")
        )
    if not output_path.exists():
        raise RuntimeError("Demucs 未生成 vocals.wav 输出。")
    return output_path


def prepare_transcription_audio(
    input_path: Path,
    output_dir: Path,
    *,
    prefix: str = "audio_local",
) -> Path:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{prefix}_{uuid.uuid4().hex[:8]}.m4a"
    mode = _get_transcription_audio_mode()
    if mode == "raw":
        _transcode_to_transcription_m4a(input_path, output_path)
        return output_path

    workdir = Path(tempfile.mkdtemp(prefix="audio_prep_", dir=str(output_dir)))
    try:
        demucs_input = workdir / "demucs_input.wav"
        _transcode_for_vocal_separation(input_path, demucs_input)
        source_duration = probe_duration(demucs_input)
        vocals_wav = _run_demucs_vocals(demucs_input, workdir / "separated")
        _transcode_to_transcription_m4a(
            vocals_wav,
            output_path,
            duration=source_duration if source_duration > 0 else None,
        )
        logger.info("已生成转录专用人声轨: %s", output_path)
        return output_path
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def extract_audio_from_media(input_path: Path, output_dir: Path) -> Path:
    return prepare_transcription_audio(input_path, output_dir, prefix="audio_local")
