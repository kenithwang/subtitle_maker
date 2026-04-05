#!/usr/bin/env python3
from pathlib import Path


def is_transcriber_root(path: Path) -> bool:
    return path.is_dir() and (path / "cli.py").exists() and (path / "backend").is_dir()


def find_transcriber_dir(skill_dir: Path) -> Path:
    for parent in skill_dir.parents:
        if is_transcriber_root(parent):
            return parent

    sibling = skill_dir.parent / "AI-Video-Transcriber"
    if is_transcriber_root(sibling):
        return sibling

    raise RuntimeError(
        "未找到 AI-Video-Transcriber 根目录。"
        "请把 skill 放在仓库内，或放在与 AI-Video-Transcriber 同级的位置。"
    )
