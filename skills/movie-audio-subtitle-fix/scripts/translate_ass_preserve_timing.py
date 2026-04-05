#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from _skill_paths import find_transcriber_dir

if TYPE_CHECKING:
    from google import genai
    from google.genai import types


THIS_FILE = Path(__file__).resolve()
SKILL_DIR = THIS_FILE.parents[1]
TRANSCRIBER_DIR = find_transcriber_dir(SKILL_DIR)


@dataclass
class DialogueEvent:
    line_index: int
    prefix: str
    fields: list[str]
    text: str


def load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return

    for path in (
        Path.cwd() / ".env.local",
        Path.cwd() / ".env",
        Path.home() / ".config" / "ai-video-transcriber" / ".env",
        TRANSCRIBER_DIR / ".env.local",
        TRANSCRIBER_DIR / ".env",
        SKILL_DIR / ".env.local",
        SKILL_DIR / ".env",
    ):
        if path.exists():
            load_dotenv(path, override=False)
            break


def extract_response_text(resp) -> str:
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
    return ""


def ass_text_to_plain(text: str) -> str:
    plain = text.replace(r"\N", "\n").replace(r"\n", "\n").replace(r"\h", " ")
    plain = re.sub(r"\{[^}]*\}", "", plain)
    plain = plain.replace(r"\\", "\\")
    return plain.strip()


def escape_ass_text(text: str) -> str:
    escaped_lines: list[str] = []
    for line in text.replace("\r", "").splitlines() or [""]:
        escaped = line.replace("\\", r"\\")
        escaped = escaped.replace("{", r"\{").replace("}", r"\}")
        escaped_lines.append(escaped)
    return r"\N".join(escaped_lines)


def first_display_line(text: str) -> str:
    for line in ass_text_to_plain(text).splitlines():
        candidate = line.strip()
        if candidate:
            return candidate
    return ""


def parse_ass(path: Path) -> tuple[list[str], list[DialogueEvent], dict[str, tuple[int, list[str]]]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    events: list[DialogueEvent] = []
    sections: dict[str, tuple[int, list[str]]] = {}

    current_section = ""
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped
            sections[current_section] = (idx, [])
            continue
        if current_section in sections:
            sections[current_section][1].append(line)

        if current_section != "[Events]":
            continue
        if not line.startswith("Dialogue:"):
            continue

        prefix, payload = line.split(":", 1)
        fields = payload.lstrip().split(",", 9)
        if len(fields) != 10:
            raise RuntimeError(f"无法解析 Dialogue 行: {line}")
        events.append(
            DialogueEvent(
                line_index=idx,
                prefix=prefix,
                fields=fields,
                text=fields[9],
            )
        )
    return lines, events, sections


def update_default_style(lines: list[str], *, font_name: str, font_size: int, outline: float, margin_v: int) -> None:
    in_styles = False
    style_fields: list[str] | None = None
    style_map: dict[str, int] | None = None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "[V4+ Styles]":
            in_styles = True
            style_fields = None
            style_map = None
            continue
        if in_styles and stripped.startswith("[") and stripped != "[V4+ Styles]":
            in_styles = False
            style_fields = None
            style_map = None
        if not in_styles:
            continue
        if stripped.startswith("Format:"):
            style_fields = [part.strip() for part in stripped[len("Format:"):].split(",")]
            style_map = {name: pos for pos, name in enumerate(style_fields)}
            continue
        if stripped.startswith("Style:") and style_map:
            payload = stripped[len("Style:"):].lstrip()
            parts = [part.strip() for part in payload.split(",")]
            if len(parts) != len(style_fields):
                continue
            if parts[style_map["Name"]] != "Default":
                continue
            parts[style_map["Fontname"]] = font_name
            parts[style_map["Fontsize"]] = str(font_size)
            if "Outline" in style_map:
                parts[style_map["Outline"]] = f"{outline:.1f}"
            if "MarginV" in style_map:
                parts[style_map["MarginV"]] = str(margin_v)
            lines[idx] = "Style: " + ",".join(parts)
            return
    raise RuntimeError("未找到可更新的 Default 样式。")


def build_batches(events: list[DialogueEvent], batch_size: int) -> list[list[tuple[int, str]]]:
    items: list[tuple[int, str]] = []
    for index, event in enumerate(events, start=1):
        source = first_display_line(event.text)
        if not source:
            source = ass_text_to_plain(event.text)
        items.append((index, source))
    return [items[pos:pos + batch_size] for pos in range(0, len(items), batch_size)]


def translate_batch(
    client: Any,
    *,
    model_name: str,
    batch: list[tuple[int, str]],
    retries: int,
) -> dict[int, str]:
    payload = [{"id": item_id, "text": text} for item_id, text in batch]
    prompt = (
        "Translate each subtitle line into concise, natural Simplified Chinese for movie subtitles.\n"
        "Rules:\n"
        "- Keep the original English text out of the output.\n"
        "- Return only a JSON array.\n"
        "- Each JSON item must be an object with keys: id, zh.\n"
        "- Preserve meaning, tone, names, and numbers.\n"
        "- Keep proper nouns or technical terms in English when Chinese subtitles normally do that.\n"
        "- If a line is already Chinese, copy it into zh.\n"
        "- Do not merge or split items.\n"
        "- Keep translations short enough for a two-line bilingual subtitle.\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )

    try:
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("需要安装 google-genai 才能调用 Gemini 翻译。") from exc

    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=8192,
        response_mime_type="application/json",
    )

    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            raw = extract_response_text(resp)
            data = json.loads(raw)
            if not isinstance(data, list):
                raise RuntimeError("模型没有返回 JSON 数组。")

            translated: dict[int, str] = {}
            for item in data:
                if not isinstance(item, dict):
                    continue
                item_id = item.get("id")
                zh = item.get("zh")
                if not isinstance(item_id, int) or not isinstance(zh, str):
                    continue
                translated[item_id] = zh.strip()

            expected_ids = {item_id for item_id, _ in batch}
            if set(translated) != expected_ids:
                missing = sorted(expected_ids - set(translated))
                extra = sorted(set(translated) - expected_ids)
                raise RuntimeError(f"批次返回项不完整: missing={missing} extra={extra}")
            return translated
        except Exception as exc:
            last_error = exc
            if attempt > retries:
                break
            time.sleep(min(10, 2 ** (attempt - 1)))

    raise RuntimeError(f"Gemini 翻译批次失败: {last_error}")


def apply_bilingual_text(lines: list[str], events: list[DialogueEvent], zh_map: dict[int, str]) -> None:
    for index, event in enumerate(events, start=1):
        fields = list(event.fields)
        english = event.text.strip()
        chinese = zh_map[index].strip()
        if not chinese:
            chinese = first_display_line(event.text)
        fields[9] = english + r"\N" + escape_ass_text(chinese)
        lines[event.line_index] = f"{event.prefix}: " + ",".join(fields)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Translate an ASS subtitle to bilingual EN+ZH while preserving all cue timings.")
    parser.add_argument("input", type=Path, help="输入 ASS 文件")
    parser.add_argument("--output", type=Path, default=None, help="输出 ASS 文件，默认覆盖输入文件")
    parser.add_argument("--backup", type=Path, default=None, help="备份当前输入文件")
    parser.add_argument("--translations-file", type=Path, default=None, help="离线翻译映射文件，格式为 id<TAB>zh")
    parser.add_argument("--checkpoint-file", type=Path, default=None, help="翻译进度检查点文件，格式为 id<TAB>zh")
    parser.add_argument("--batch-size", type=int, default=20, help="每次提交给 Gemini 的字幕条数")
    parser.add_argument("--font-name", default="Microsoft YaHei", help="双语字幕字体")
    parser.add_argument("--font-size", type=int, default=36, help="双语字幕字号")
    parser.add_argument("--outline", type=float, default=2.0, help="双语字幕描边粗细")
    parser.add_argument("--margin-v", type=int, default=32, help="双语字幕底边距")
    parser.add_argument("--model", default=None, help="覆盖 GEMINI_MODEL")
    parser.add_argument("--retries", type=int, default=2, help="单个翻译批次失败后的重试次数")
    return parser


def load_offline_translations(path: Path, expected_count: int) -> dict[int, str]:
    translations: dict[int, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            raise RuntimeError(f"离线翻译行格式错误: {raw}")
        try:
            item_id = int(parts[0])
        except ValueError as exc:
            raise RuntimeError(f"离线翻译编号无效: {parts[0]}") from exc
        translations[item_id] = parts[1].strip()

    expected_ids = set(range(1, expected_count + 1))
    got_ids = set(translations)
    if got_ids != expected_ids:
        missing = sorted(expected_ids - got_ids)
        extra = sorted(got_ids - expected_ids)
        raise RuntimeError(f"离线翻译映射不完整: missing={missing[:10]} extra={extra[:10]}")
    return translations


def load_partial_translations(path: Path) -> dict[int, str]:
    translations: dict[int, str] = {}
    if not path.exists():
        return translations
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        try:
            item_id = int(parts[0])
        except ValueError:
            continue
        translations[item_id] = parts[1].strip()
    return translations


def write_partial_translations(path: Path, translations: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{item_id}\t{translations[item_id]}" for item_id in sorted(translations)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    load_dotenv_if_present()
    args = build_parser().parse_args()

    input_path = args.input.expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"输入 ASS 不存在: {input_path}")

    output_path = (args.output or input_path).expanduser()
    backup_path = (args.backup or input_path.with_suffix(input_path.suffix + ".pre_bilingual.bak")).expanduser()

    lines, events, _sections = parse_ass(input_path)
    if not events:
        raise RuntimeError("ASS 中没有可处理的 Dialogue 行。")

    if not backup_path.exists():
        backup_path.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")

    if args.translations_file:
        zh_map = load_offline_translations(args.translations_file.expanduser(), len(events))
    else:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("需要安装 google-genai 才能调用 Gemini 翻译。") from exc

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("未设置 GEMINI_API_KEY。")
        model_name = args.model or os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
        if model_name.startswith("models/"):
            model_name = model_name.split("/", 1)[-1]

        client = genai.Client(api_key=api_key)
        batches = build_batches(events, max(1, args.batch_size))
        checkpoint_path = (
            args.checkpoint_file.expanduser()
            if args.checkpoint_file
            else output_path.with_suffix(output_path.suffix + ".translations.tsv")
        )
        zh_map = load_partial_translations(checkpoint_path)

        for batch_index, batch in enumerate(batches, start=1):
            batch_ids = {item_id for item_id, _ in batch}
            if batch_ids.issubset(zh_map):
                print(f"[translate] batch {batch_index}/{len(batches)} ({len(batch)} cues) skipped", flush=True)
                continue
            print(f"[translate] batch {batch_index}/{len(batches)} ({len(batch)} cues)", flush=True)
            translated = translate_batch(
                client,
                model_name=model_name,
                batch=batch,
                retries=max(0, args.retries),
            )
            zh_map.update(translated)
            write_partial_translations(checkpoint_path, zh_map)

        expected_ids = set(range(1, len(events) + 1))
        if set(zh_map) != expected_ids:
            missing = sorted(expected_ids - set(zh_map))
            extra = sorted(set(zh_map) - expected_ids)
            raise RuntimeError(f"翻译结果不完整: missing={missing[:10]} extra={extra[:10]}")

    apply_bilingual_text(lines, events, zh_map)
    update_default_style(
        lines,
        font_name=args.font_name,
        font_size=args.font_size,
        outline=args.outline,
        margin_v=args.margin_v,
    )

    text = "\n".join(lines) + "\n"
    output_path.write_text(text, encoding="utf-8")
    print(f"[done] output={output_path}", flush=True)
    print(f"[done] backup={backup_path}", flush=True)
    print(f"[done] events={len(events)}", flush=True)


if __name__ == "__main__":
    main()
