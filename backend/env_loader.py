import os
from pathlib import Path


def iter_env_candidates(project_root: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path | None) -> None:
        if not path:
            return
        resolved = path.expanduser()
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(resolved)

    override = os.getenv("AI_VIDEO_TRANSCRIBER_ENV_FILE")
    if override:
        add(Path(override))

    cwd = Path.cwd()
    add(cwd / ".env.local")

    if project_root:
        add(project_root / ".env.local")

    add(Path.home() / ".config" / "ai-video-transcriber" / ".env")

    add(cwd / ".env")
    if project_root:
        add(project_root / ".env")

    return candidates


def load_env_if_present(project_root: Path | None = None) -> Path | None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return None

    for path in iter_env_candidates(project_root):
        if path.exists():
            load_dotenv(path, override=False)
            return path
    return None
