import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Optional

from .media_tools import extract_audio_from_media, prepare_transcription_audio
from .srt_maker import GeminiSRTMaker
from .video_processor import VideoProcessor

logger = logging.getLogger(__name__)


def _create_emitter(on_update: Optional[Callable[[dict], Awaitable[None]]]):
    def emit(update: dict):
        if on_update:
            return on_update(update)
        return asyncio.sleep(0)

    return emit


async def _write_file(path: Path, content: str) -> None:
    await asyncio.to_thread(path.write_text, content, encoding="utf-8")


def _sanitize_title_for_filename(title: str, max_bytes: int = 200) -> str:
    import re

    if not title:
        return "untitled"
    safe = re.sub(r"[^\w\-\s]", "", title)
    safe = re.sub(r"\s+", "_", safe).strip("._-")
    if not safe:
        return "untitled"
    encoded = safe.encode("utf-8")
    if len(encoded) <= max_bytes:
        return safe
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return truncated.rstrip("._-") or "untitled"


async def _cleanup_audio(audio_path: str, keep_audio: bool) -> bool:
    if keep_audio:
        return False
    try:
        await asyncio.to_thread(lambda: Path(audio_path).unlink(missing_ok=True))
        return True
    except Exception as exc:
        logger.warning("删除中间音频失败: %s (%s)", audio_path, exc)
        return False


async def _prepare_audio_for_subtitles(audio_path: str | Path, temp_dir: Path) -> str:
    source_path = Path(audio_path)
    prepared_path = await asyncio.to_thread(
        prepare_transcription_audio,
        source_path,
        temp_dir,
        prefix=f"{source_path.stem}_dialogue",
    )
    if prepared_path != source_path:
        try:
            source_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.debug("删除原始音频失败: %s (%s)", source_path, exc)
    return str(prepared_path)


async def process_url_to_subtitles(
    url: str,
    temp_dir: Path,
    *,
    keep_audio: bool = False,
    subtitle_mode: str = "bilingual-zh",
    subtitle_format: str = "srt",
    timing_mode: str = "align",
    align_model: Optional[str] = None,
    segment_seconds: Optional[int] = None,
    parallelism: Optional[int] = None,
    on_update: Optional[Callable[[dict], Awaitable[None]]] = None,
) -> dict:
    temp_dir.mkdir(parents=True, exist_ok=True)
    emit = _create_emitter(on_update)

    short_id = uuid.uuid4().hex[:6]
    status = {
        "status": "processing",
        "progress": 0,
        "message": "starting",
        "url": url,
    }
    await emit(status)

    processor = VideoProcessor()
    maker = GeminiSRTMaker(
        segment_seconds=segment_seconds or int(os.getenv("SEGMENT_SECONDS", "1200")),
        parallelism=parallelism,
    )

    status.update({"progress": 5, "message": "checking model..."})
    await emit(status)
    await asyncio.to_thread(maker.validate_model)

    status.update({"progress": 10, "message": "downloading media..."})
    await emit(status)
    audio_path, video_title = await processor.download_and_convert(url, temp_dir)

    status.update({"progress": 25, "message": "preparing dialogue track..."})
    await emit(status)
    audio_path = await _prepare_audio_for_subtitles(audio_path, temp_dir)

    safe_title = _sanitize_title_for_filename(video_title)
    subtitle_ext = subtitle_format.lower()
    subtitle_filename = f"subtitle_{safe_title}_{short_id}.{subtitle_ext}"
    subtitle_path = temp_dir / subtitle_filename
    align_model_label = (align_model or os.getenv("ALIGN_MODEL") or "").strip() or "auto"
    align_cache_path = subtitle_path.with_suffix(f".{align_model_label}.words.json")

    status.update({"progress": 45, "message": "media ready; generating subtitles..."})
    await emit(status)
    subtitle_text, detected_language, warnings = await asyncio.to_thread(
        maker.make_subtitles,
        Path(audio_path),
        subtitle_mode,
        subtitle_format,
        timing_mode,
        align_model,
        align_cache_path,
    )

    await _write_file(subtitle_path, subtitle_text)

    audio_deleted = await _cleanup_audio(audio_path, keep_audio)

    status.update({"progress": 100, "message": "completed", "status": "completed"})
    await emit(status)

    return {
        "status": "completed",
        "video_title": video_title,
        "detected_language": detected_language,
        "subtitle_file": subtitle_filename,
        "subtitle_format": subtitle_ext,
        "timing_mode": timing_mode,
        "audio_file": None if audio_deleted else audio_path,
        "audio_deleted": audio_deleted,
        "warnings": warnings,
    }


async def process_local_media_to_subtitles(
    input_path: Path,
    temp_dir: Path,
    *,
    title: str | None = None,
    keep_audio: bool = False,
    subtitle_mode: str = "bilingual-zh",
    subtitle_format: str = "srt",
    timing_mode: str = "align",
    align_model: Optional[str] = None,
    segment_seconds: Optional[int] = None,
    parallelism: Optional[int] = None,
    on_update: Optional[Callable[[dict], Awaitable[None]]] = None,
) -> dict:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    temp_dir.mkdir(parents=True, exist_ok=True)
    emit = _create_emitter(on_update)

    short_id = uuid.uuid4().hex[:6]
    status = {
        "status": "processing",
        "progress": 0,
        "message": "starting",
        "input": str(input_path),
    }
    await emit(status)

    maker = GeminiSRTMaker(
        segment_seconds=segment_seconds or int(os.getenv("SEGMENT_SECONDS", "1200")),
        parallelism=parallelism,
    )

    status.update({"progress": 5, "message": "checking model..."})
    await emit(status)
    await asyncio.to_thread(maker.validate_model)

    status.update({"progress": 10, "message": "extracting dialogue audio..."})
    await emit(status)
    audio_path = await asyncio.to_thread(extract_audio_from_media, input_path, temp_dir)
    video_title = title or input_path.stem
    safe_title = _sanitize_title_for_filename(video_title)
    subtitle_ext = subtitle_format.lower()
    subtitle_filename = f"subtitle_{safe_title}_{short_id}.{subtitle_ext}"
    subtitle_path = temp_dir / subtitle_filename
    align_model_label = (align_model or os.getenv("ALIGN_MODEL") or "").strip() or "auto"
    align_cache_path = subtitle_path.with_suffix(f".{align_model_label}.words.json")

    status.update({"progress": 40, "message": "audio ready; generating subtitles..."})
    await emit(status)
    subtitle_text, detected_language, warnings = await asyncio.to_thread(
        maker.make_subtitles,
        Path(audio_path),
        subtitle_mode,
        subtitle_format,
        timing_mode,
        align_model,
        align_cache_path,
    )

    await _write_file(subtitle_path, subtitle_text)

    audio_deleted = await _cleanup_audio(str(audio_path), keep_audio)

    status.update({"progress": 100, "message": "completed", "status": "completed"})
    await emit(status)

    return {
        "status": "completed",
        "video_title": video_title,
        "detected_language": detected_language,
        "subtitle_file": subtitle_filename,
        "subtitle_format": subtitle_ext,
        "timing_mode": timing_mode,
        "audio_file": None if audio_deleted else str(audio_path),
        "audio_deleted": audio_deleted,
        "warnings": warnings,
    }
