import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple

from google import genai
from google.genai import types

from .media_tools import ffmpeg_cmd, probe_duration
from .timing_aligner import AlignmentEntry, align_entries

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    path: Path
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class SubtitleCue:
    start_ms: int
    end_ms: int
    text: str


def _guess_language(text: str) -> Optional[str]:
    total = len(text) or 1
    hiragana = sum(1 for ch in text if "\u3040" <= ch <= "\u309f")
    katakana = sum(1 for ch in text if "\u30a0" <= ch <= "\u30ff")
    jp_kana = hiragana + katakana
    kanji = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    en = sum(1 for ch in text if ch.isascii() and ch.isalpha())

    if jp_kana / total > 0.05:
        return "ja"
    if kanji / total > 0.2:
        return "zh"
    if en / total > 0.2:
        return "en"
    return None


class GeminiSRTMaker:
    RELIABLE_SEGMENT_DURATION = 20 * 60
    SUPPORTED_SUBTITLE_FORMATS = {"srt", "ass"}
    SUPPORTED_TIMING_MODES = {"gemini", "align", "placeholder"}

    def __init__(self, segment_seconds: int = 20 * 60, parallelism: Optional[int] = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("未设置 GEMINI_API_KEY")

        self.api_key = api_key
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
        if self.model_name.startswith("models/"):
            self.model_name = self.model_name.split("/", 1)[-1]

        safe_segment = max(60, int(segment_seconds))
        self.segment_seconds = min(safe_segment, self.RELIABLE_SEGMENT_DURATION)

        if parallelism is None:
            par_env = os.getenv("TRANSCRIBE_CONCURRENCY")
            try:
                self.parallelism = int(par_env) if par_env else 0
            except Exception:
                self.parallelism = 0
        else:
            self.parallelism = int(parallelism)

        if self.parallelism < 1:
            cpu_default = os.cpu_count() or 4
            self.parallelism = min(4, max(1, cpu_default // 2 or 1))

        try:
            self.file_api_retries = max(0, int(os.getenv("GEMINI_FILE_API_RETRIES", "3")))
        except Exception:
            self.file_api_retries = 3

        self._system_instruction = (
            "You are a professional subtitle editor.\n"
            "Return ONLY valid SRT subtitle content.\n"
            "Do not add explanations, Markdown fences, or commentary.\n"
            "Use precise, natural subtitle segmentation.\n"
            "Keep each subtitle concise and readable.\n"
            "Do not hallucinate dialogue that is not present in the audio."
        )
        self._generation_config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=65536,
            system_instruction=self._system_instruction,
        )

    def validate_model(self) -> None:
        logger.info("[srt] 预检模型: %s", self.model_name)
        client = genai.Client(api_key=self.api_key)
        try:
            resp = client.models.generate_content(
                model=self.model_name,
                contents="Reply with OK.",
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    # Some Gemini models may consume small token budgets on hidden reasoning
                    # and return an empty body with MAX_TOKENS. Use a generous cap here so
                    # validation proves the model can actually emit user-visible text.
                    max_output_tokens=65536,
                ),
            )
            text = self._extract_response_text(resp)
            if not text:
                raise RuntimeError(
                    f"模型返回空内容 (finish_reason={self._summarize_finish_reasons(resp)})"
                )
        except Exception as exc:
            raise RuntimeError(f"模型不可用: {self.model_name}: {exc}") from exc
        logger.info("[srt] 模型预检通过: %s", self.model_name)

    def _build_prompt(self, chunk: AudioChunk, subtitle_mode: str) -> str:
        bilingual_rule = (
            "For each subtitle cue, use two lines:\n"
            "1. Original spoken dialogue in the original language.\n"
            "2. Simplified Chinese translation.\n"
            "If the spoken language is already Chinese, keep a single Chinese line and do not translate into Chinese again.\n"
        )
        monolingual_rule = (
            "For each subtitle cue, keep only the original spoken dialogue with no translation.\n"
        )

        mode_rule = bilingual_rule if subtitle_mode == "bilingual-zh" else monolingual_rule

        return (
            "Create subtitles for this audio chunk.\n"
            f"The chunk duration is {chunk.duration:.3f} seconds.\n"
            "Rules:\n"
            "- Return ONLY valid SRT.\n"
            "- Use chunk-relative timestamps starting at 00:00:00,000.\n"
            "- Keep all timestamps within this chunk's duration.\n"
            "- Subtitle timing must be continuous and non-negative.\n"
            "- Use at most two short displayed lines per cue.\n"
            "- Keep names, numbers, and punctuation accurate.\n"
            "- Do not include speaker labels unless absolutely necessary.\n"
            f"{mode_rule}"
        )

    def _get_mime_type(self, path: Path) -> str:
        ext = path.suffix.lower()
        mime_map = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".aac": "audio/aac",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
            ".webm": "audio/webm",
        }
        return mime_map.get(ext, "audio/mp4")

    def _extract_response_text(self, resp) -> str:
        direct_text = getattr(resp, "text", None)
        if direct_text:
            return str(direct_text).strip()

        try:
            for cand in getattr(resp, "candidates", []) or []:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None)
                if not parts:
                    continue
                acc: list[str] = []
                for part in parts:
                    text = getattr(part, "text", None)
                    if text:
                        acc.append(text)
                joined = "\n".join(acc).strip()
                if joined:
                    return joined
            return ""
        except Exception:
            return ""

    def _summarize_finish_reasons(self, resp) -> str:
        reasons: list[str] = []
        for cand in getattr(resp, "candidates", []) or []:
            reason = getattr(cand, "finish_reason", None)
            if reason is not None:
                reasons.append(str(reason))
        return ", ".join(reasons) if reasons else "unknown"

    def _extract_srt_body(self, text: str) -> str:
        text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        text = re.sub(r"^```(?:srt)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        match = re.search(
            r"\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}",
            text,
        )
        if match:
            text = text[match.start():]
        return text.strip()

    def _parse_timestamp_ms(self, value: str) -> int:
        normalized = value.strip().replace(".", ",")
        hh, mm, ss_ms = normalized.split(":")
        ss, ms = ss_ms.split(",")
        ms = (ms + "000")[:3]
        total = (
            int(hh) * 3600 * 1000
            + int(mm) * 60 * 1000
            + int(ss) * 1000
            + int(ms)
        )
        return total

    def _format_timestamp(self, total_ms: int) -> str:
        total_ms = max(0, total_ms)
        hours = total_ms // 3_600_000
        total_ms %= 3_600_000
        minutes = total_ms // 60_000
        total_ms %= 60_000
        seconds = total_ms // 1000
        milliseconds = total_ms % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def _format_ass_timestamp(self, total_ms: int) -> str:
        total_ms = max(0, total_ms)
        hours = total_ms // 3_600_000
        total_ms %= 3_600_000
        minutes = total_ms // 60_000
        total_ms %= 60_000
        seconds = total_ms // 1000
        centiseconds = (total_ms % 1000) // 10
        return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"

    def _fmt_duration(self, seconds: float) -> str:
        minutes = int(seconds // 60)
        remainder = int(round(seconds - minutes * 60))
        if remainder == 60:
            minutes += 1
            remainder = 0
        return f"{minutes}m {remainder:02d}s"

    def _parse_srt(self, text: str) -> List[SubtitleCue]:
        text = self._extract_srt_body(text)
        if not text:
            return []

        cues: list[SubtitleCue] = []
        blocks = re.split(r"\n{2,}", text)
        timestamp_re = re.compile(
            r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
        )

        for block in blocks:
            lines = [line.rstrip() for line in block.split("\n") if line.strip()]
            if not lines:
                continue
            if re.fullmatch(r"\d+", lines[0]):
                lines = lines[1:]
            if len(lines) < 2:
                continue
            match = timestamp_re.match(lines[0])
            if not match:
                continue

            start_ms = self._parse_timestamp_ms(match.group("start"))
            end_ms = self._parse_timestamp_ms(match.group("end"))
            payload = "\n".join(lines[1:]).strip()
            if not payload:
                continue
            cues.append(SubtitleCue(start_ms=start_ms, end_ms=end_ms, text=payload))

        return cues

    def _sanitize_cues(self, cues: List[SubtitleCue], chunk: AudioChunk) -> List[SubtitleCue]:
        duration_ms = int(round(chunk.duration * 1000))
        cleaned: list[SubtitleCue] = []
        prev_end = 0

        for cue in cues:
            start_ms = max(0, cue.start_ms)
            end_ms = min(duration_ms, cue.end_ms)
            if start_ms < prev_end:
                start_ms = prev_end
            if end_ms <= start_ms:
                end_ms = min(duration_ms, start_ms + 1200)
            if end_ms <= start_ms:
                continue

            text = cue.text.replace("\r", "").strip()
            text = re.sub(r"[ \t]+\n", "\n", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            if not text:
                continue

            cleaned.append(SubtitleCue(start_ms=start_ms, end_ms=end_ms, text=text))
            prev_end = end_ms

        return cleaned

    def _offset_cues(self, cues: List[SubtitleCue], offset_ms: int) -> List[SubtitleCue]:
        return [
            SubtitleCue(
                start_ms=cue.start_ms + offset_ms,
                end_ms=cue.end_ms + offset_ms,
                text=cue.text,
            )
            for cue in cues
        ]

    def _format_srt(self, cues: List[SubtitleCue]) -> str:
        blocks: list[str] = []
        for idx, cue in enumerate(cues, start=1):
            blocks.append(
                "\n".join(
                    [
                        str(idx),
                        f"{self._format_timestamp(cue.start_ms)} --> {self._format_timestamp(cue.end_ms)}",
                        cue.text,
                    ]
                )
            )
        return "\n\n".join(blocks).strip() + "\n"

    def _escape_ass_text(self, text: str) -> str:
        escaped_lines: list[str] = []
        for line in text.replace("\r", "").splitlines() or [""]:
            escaped = line.replace("\\", r"\\")
            escaped = escaped.replace("{", r"\{").replace("}", r"\}")
            escaped_lines.append(escaped)
        return r"\N".join(escaped_lines)

    def _format_ass(self, cues: List[SubtitleCue]) -> str:
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
                start=self._format_ass_timestamp(cue.start_ms),
                end=self._format_ass_timestamp(cue.end_ms),
                text=self._escape_ass_text(cue.text),
            )
            for cue in cues
        ]
        return header + "\n" + "\n".join(events).strip() + "\n"

    def _placeholder_duration_ms(self, text: str) -> int:
        visible = (text or "").replace(r"\N", "\n").replace("\r", "")
        word_count = len(re.findall(r"\S+", visible))
        line_count = max(1, len([line for line in visible.splitlines() if line.strip()]))
        duration = 650 + word_count * 260 + (line_count - 1) * 220
        return min(4200, max(900, duration))

    def _placeholderize_cues(self, cues: List[SubtitleCue]) -> List[SubtitleCue]:
        cursor = 0
        rewritten: list[SubtitleCue] = []
        for cue in cues:
            duration = self._placeholder_duration_ms(cue.text)
            rewritten.append(replace(cue, start_ms=cursor, end_ms=cursor + duration))
            cursor += duration + 80
        return rewritten

    def _render_subtitle_text(self, cues: List[SubtitleCue], subtitle_format: str) -> str:
        subtitle_format = (subtitle_format or "srt").lower()
        if subtitle_format == "srt":
            return self._format_srt(cues)
        if subtitle_format == "ass":
            return self._format_ass(cues)
        raise ValueError(f"不支持的字幕格式: {subtitle_format}")

    def _resolve_align_model(self, detected_language: Optional[str]) -> str:
        configured = (os.getenv("ALIGN_MODEL") or "").strip()
        if configured:
            return configured
        return "medium.en" if detected_language == "en" else "medium"

    def _force_align_cues(
        self,
        cues: List[SubtitleCue],
        audio_path: Path,
        *,
        detected_language: Optional[str],
        cache_path: Optional[Path],
        align_model: Optional[str],
    ) -> tuple[List[SubtitleCue], dict]:
        model_name = align_model or self._resolve_align_model(detected_language)
        cache_target = cache_path or audio_path.with_suffix(audio_path.suffix + f".{model_name}.words.json")
        entries = [
            AlignmentEntry(start_ms=cue.start_ms, end_ms=cue.end_ms, text=cue.text)
            for cue in cues
        ]
        timings, stats = align_entries(
            entries,
            audio_path,
            model_name=model_name,
            cache_path=cache_target,
            language=detected_language,
        )
        aligned = [
            replace(cue, start_ms=start_ms, end_ms=end_ms)
            for cue, (start_ms, end_ms) in zip(cues, timings)
        ]
        return aligned, stats

    def _ffprobe_duration(self, path: Path) -> float:
        try:
            return probe_duration(path)
        except Exception:
            return 0.0

    def _split_audio(self, audio_path: Path) -> tuple[List[AudioChunk], Path]:
        workdir = Path(tempfile.mkdtemp(prefix="srt_work_", dir=str(audio_path.parent)))
        norm_wav = workdir / "normalized.wav"
        subprocess.check_call(
            ffmpeg_cmd(
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(norm_wav),
            )
        )

        duration = self._ffprobe_duration(norm_wav)
        if duration <= 0:
            raise RuntimeError("无法探测音频时长")

        cmd_sil = ffmpeg_cmd(
            "-hide_banner",
            "-nostats",
            "-i",
            str(norm_wav),
            "-af",
            "silencedetect=noise=-30dB:d=0.3",
            "-f",
            "null",
            "-",
        )
        try:
            out = subprocess.check_output(cmd_sil, stderr=subprocess.STDOUT).decode("utf-8", "ignore")
        except subprocess.CalledProcessError as exc:
            out = exc.output.decode("utf-8", "ignore") if exc.output else ""

        silence_points: list[float] = []
        for line in out.splitlines():
            match = re.search(r"silence_(start|end):\s*([0-9.]+)", line)
            if match:
                silence_points.append(float(match.group(2)))
        silence_points = sorted(set(silence_points))

        target = min(self.segment_seconds, self.RELIABLE_SEGMENT_DURATION)
        search_window = 5.0
        min_seg = 1.0
        cuts: list[tuple[float, float]] = []
        start = 0.0
        while start < duration:
            desired = start + target
            if desired >= duration:
                end = duration
            else:
                nearby = [t for t in silence_points if (desired - search_window) <= t <= (desired + search_window)]
                if nearby:
                    end = min(nearby, key=lambda item: abs(item - desired))
                    if end - start < min_seg:
                        end = min(desired, duration)
                else:
                    end = min(desired, duration)
            if end - start >= min_seg:
                cuts.append((start, end))
            start = end

        outdir = workdir / "chunks"
        outdir.mkdir(parents=True, exist_ok=True)

        if len(cuts) == 1 and abs(cuts[0][0]) < 1e-3 and abs(cuts[0][1] - duration) < 1e-3:
            target_path = outdir / "chunk_001.wav"
            shutil.copy2(norm_wav, target_path)
            return [AudioChunk(target_path, cuts[0][0], cuts[0][1])], workdir

        segment_points = ",".join(f"{end:.3f}" for _, end in cuts[:-1])
        subprocess.check_call(
            ffmpeg_cmd(
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(norm_wav),
                "-f",
                "segment",
                "-segment_times",
                segment_points,
                "-segment_start_number",
                "1",
                "-reset_timestamps",
                "1",
                "-c",
                "copy",
                str(outdir / "chunk_%03d.wav"),
            )
        )

        produced = sorted(outdir.glob("chunk_*.wav"))
        if len(produced) != len(cuts):
            raise RuntimeError(f"分片数量不匹配，期望 {len(cuts)} 实际 {len(produced)}")

        return [
            AudioChunk(path=path, start=cuts[idx][0], end=cuts[idx][1])
            for idx, path in enumerate(produced)
        ], workdir

    def _render_chunk(
        self,
        chunk: AudioChunk,
        subtitle_mode: str,
        chunk_index: int,
        chunk_total: int,
    ) -> List[SubtitleCue]:
        prompt = self._build_prompt(chunk, subtitle_mode)
        logger.info(
            "[srt] 分片 %s/%s 开始: %s-%s (~%s)",
            chunk_index,
            chunk_total,
            self._fmt_duration(chunk.start),
            self._fmt_duration(chunk.end),
            self._fmt_duration(chunk.duration),
        )
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.file_api_retries + 2):
            client = genai.Client(api_key=self.api_key)
            uploaded = None
            try:
                logger.info(
                    "[srt] 分片 %s/%s File API 尝试 %s/%s：上传文件…",
                    chunk_index,
                    chunk_total,
                    attempt,
                    self.file_api_retries + 1,
                )
                uploaded = client.files.upload(file=str(chunk.path))
                logger.info(
                    "[srt] 分片 %s/%s File API 尝试 %s/%s：请求 Gemini…",
                    chunk_index,
                    chunk_total,
                    attempt,
                    self.file_api_retries + 1,
                )
                resp = client.models.generate_content(
                    model=self.model_name,
                    contents=[uploaded, prompt],
                    config=self._generation_config,
                )
                raw_text = self._extract_response_text(resp)
                cues = self._sanitize_cues(self._parse_srt(raw_text), chunk)
                if cues:
                    logger.info("[srt] 分片 %s/%s 完成: %s 条字幕", chunk_index, chunk_total, len(cues))
                    return cues
                raise RuntimeError(
                    "Gemini 返回了空内容或不可解析的 SRT "
                    f"(finish_reason={self._summarize_finish_reasons(resp)})"
                )
            except Exception as exc:
                last_exc = exc
                if attempt > self.file_api_retries:
                    raise RuntimeError(
                        f"分片 {chunk_index}/{chunk_total} File API 在 {attempt} 次尝试后仍失败: {exc}"
                    ) from exc
                delay = min(20, 2 ** (attempt - 1))
                logger.warning(
                    "[srt] 分片 %s/%s File API 尝试 %s/%s 失败，%s 秒后重试: %s",
                    chunk_index,
                    chunk_total,
                    attempt,
                    self.file_api_retries + 1,
                    delay,
                    exc,
                )
                time.sleep(delay)
            finally:
                try:
                    if uploaded:
                        client.files.delete(name=uploaded.name)
                except Exception:
                    pass

        if last_exc is not None:
            raise RuntimeError(
                f"分片 {chunk_index}/{chunk_total} 未拿到可用 Gemini 响应: {last_exc}"
            ) from last_exc
        raise RuntimeError(f"分片 {chunk_index}/{chunk_total} 未拿到可用 Gemini 响应")

    def make_subtitles(
        self,
        audio_path: Path,
        subtitle_mode: str = "bilingual-zh",
        subtitle_format: str = "srt",
        timing_mode: str = "align",
        align_model: Optional[str] = None,
        align_cache_path: Optional[Path] = None,
    ) -> Tuple[str, Optional[str], List[str]]:
        if subtitle_mode not in {"bilingual-zh", "monolingual"}:
            raise ValueError(f"不支持的字幕模式: {subtitle_mode}")
        subtitle_format = (subtitle_format or "srt").lower()
        if subtitle_format not in self.SUPPORTED_SUBTITLE_FORMATS:
            raise ValueError(f"不支持的字幕格式: {subtitle_format}")
        timing_mode = (timing_mode or "align").lower()
        if timing_mode not in self.SUPPORTED_TIMING_MODES:
            raise ValueError(f"不支持的时间轴模式: {timing_mode}")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        duration = self._ffprobe_duration(audio_path)
        logger.info("开始生成字幕: %s (%.1fs, %s, timing=%s)", audio_path.name, duration, subtitle_format, timing_mode)

        if duration <= self.RELIABLE_SEGMENT_DURATION:
            workdir = Path(tempfile.mkdtemp(prefix="srt_work_", dir=str(audio_path.parent)))
            chunks = [AudioChunk(path=audio_path, start=0.0, end=duration)]
            logger.info("[srt] 音频较短，单段处理完成。")
        else:
            chunks, workdir = self._split_audio(audio_path)
            logger.info("[srt] 音频已切成 %s 段。", len(chunks))

        try:
            results: dict[int, List[SubtitleCue]] = {}
            error_chunks: list[int] = []

            with ThreadPoolExecutor(max_workers=self.parallelism) as executor:
                futures = {
                    executor.submit(self._render_chunk, chunk, subtitle_mode, index, len(chunks)): index
                    for index, chunk in enumerate(chunks, start=1)
                }
                done_count = 0
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        cues = future.result()
                    except Exception as exc:
                        logger.error("分片 %s 生成字幕失败: %s", idx, exc)
                        error_chunks.append(idx)
                        cues = []
                    results[idx] = cues
                    done_count += 1
                    logger.info("[srt] 分片进度: %s/%s", done_count, len(chunks))

            if error_chunks:
                raise RuntimeError(f"以下分片在重试后仍失败: {error_chunks}")

            merged: list[SubtitleCue] = []
            for idx, chunk in enumerate(chunks, start=1):
                merged.extend(self._offset_cues(results.get(idx, []), int(round(chunk.start * 1000))))

            if not merged:
                raise RuntimeError("模型未返回可解析的字幕内容")

            warnings: list[str] = []

            original_lines = []
            for cue in merged:
                first_line = cue.text.splitlines()[0].strip()
                if first_line:
                    original_lines.append(first_line)
            detected_language = _guess_language("\n".join(original_lines))

            if timing_mode == "placeholder":
                merged = self._placeholderize_cues(merged)
                warnings.append("生成阶段使用占位时间轴；最终时间应依赖后续强制对齐。")
            elif timing_mode == "align":
                merged, align_stats = self._force_align_cues(
                    merged,
                    audio_path,
                    detected_language=detected_language,
                    cache_path=align_cache_path,
                    align_model=align_model,
                )
                logger.info(
                    "[align] 本地对齐完成: matched=%s/%s, model=%s",
                    align_stats.get("matched_lines"),
                    align_stats.get("total_lines"),
                    align_stats.get("model"),
                )
                if align_stats.get("matched_lines", 0) < max(10, len(merged) // 8):
                    warnings.append(
                        "本地对齐锚点较少，部分时间轴可能仍依赖插值。"
                    )

            logger.info("[srt] 全部合并完成，共 %s 条字幕。", len(merged))
            return self._render_subtitle_text(merged, subtitle_format), detected_language, warnings
        finally:
            try:
                shutil.rmtree(workdir, ignore_errors=True)
            except Exception:
                pass

    def make_srt(
        self,
        audio_path: Path,
        subtitle_mode: str = "bilingual-zh",
    ) -> Tuple[str, Optional[str], List[str]]:
        return self.make_subtitles(audio_path, subtitle_mode=subtitle_mode, subtitle_format="srt")
