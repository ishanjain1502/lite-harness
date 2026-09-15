"""Resolve ffmpeg/ffprobe — system PATH, config path, or auto-download."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

_DURATION_RE = re.compile(
    r"Duration:\s*(\d+):(\d{2}):(\d{2}\.\d+)",
    re.IGNORECASE,
)
_VIDEO_STREAM_RE = re.compile(
    r"Stream #\d+:\d+.*?:\s*Video:\s*(\w+).*?(\d+)x(\d+)",
    re.IGNORECASE,
)
_AUDIO_STREAM_RE = re.compile(
    r"Stream #\d+:\d+.*?:\s*Audio:\s*(\w+)",
    re.IGNORECASE,
)


class FFmpegUnavailable(Exception):
    """Raised when ffmpeg cannot be resolved."""


def find_ffprobe_near(ffmpeg_path: str) -> str | None:
    ffmpeg = Path(ffmpeg_path)
    name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    candidate = ffmpeg.with_name(name)
    if candidate.is_file():
        return str(candidate)
    return None


def _resolve_from_path_mode(
    ffmpeg_path: str | None,
    ffprobe_path: str | None,
) -> tuple[str, str | None]:
    if not ffmpeg_path:
        raise FFmpegUnavailable("ffmpeg_mode=path requires ffmpeg_path in plugin config")
    probe = ffprobe_path or find_ffprobe_near(ffmpeg_path)
    return ffmpeg_path, probe


def _resolve_from_system() -> tuple[str, str | None] | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    ffprobe = shutil.which("ffprobe") or find_ffprobe_near(ffmpeg)
    return ffmpeg, ffprobe


def _resolve_bundled() -> tuple[str, str | None]:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise FFmpegUnavailable(
            "ffmpeg not found on PATH; install ffmpeg system-wide or run: "
            "pip install 'lite-ness[video]'"
        ) from exc

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    ffprobe = find_ffprobe_near(ffmpeg) or shutil.which("ffprobe")
    return ffmpeg, ffprobe


def resolve_ffmpeg_binaries(
    *,
    mode: str,
    temp_dir: Path,
    ffmpeg_path: str | None,
    ffprobe_path: str | None,
) -> tuple[str, str | None]:
    """Resolve ffmpeg and optionally ffprobe binary paths.

    Modes:
    - auto: explicit config → system PATH → imageio-ffmpeg download
    - system: explicit config → system PATH only
    - path: explicit ffmpeg_path (+ optional ffprobe_path)
    """
    if mode == "path":
        return _resolve_from_path_mode(ffmpeg_path, ffprobe_path)

    if ffmpeg_path:
        probe = ffprobe_path or find_ffprobe_near(ffmpeg_path)
        return ffmpeg_path, probe

    system = _resolve_from_system()
    if system is not None:
        return system

    if mode == "system":
        raise FFmpegUnavailable(
            "ffmpeg not found on PATH; install ffmpeg or set ffmpeg_mode: auto"
        )

    return _resolve_bundled()


def _ffmpeg_probe_output(ffmpeg: str, path: Path) -> str | None:
    try:
        completed = subprocess.run(
            [ffmpeg, "-i", str(path)],
            capture_output=True,
            text=True,
            timeout=60.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return f"{completed.stdout}\n{completed.stderr}"


def _parse_duration_from_probe_text(text: str) -> float | None:
    match = _DURATION_RE.search(text)
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def probe_duration_with_ffmpeg(ffmpeg: str, path: Path) -> float | None:
    """Fallback duration probe using ffmpeg -i stderr when ffprobe is missing."""
    text = _ffmpeg_probe_output(ffmpeg, path)
    if not text:
        return None
    return _parse_duration_from_probe_text(text)


def probe_video_info_with_ffmpeg(ffmpeg: str, path: Path) -> dict[str, Any] | None:
    """Build a minimal ffprobe-shaped payload from ffmpeg -i output."""
    text = _ffmpeg_probe_output(ffmpeg, path)
    if not text:
        return None

    duration = _parse_duration_from_probe_text(text)
    streams: list[dict[str, Any]] = []

    for match in _VIDEO_STREAM_RE.finditer(text):
        video_stream: dict[str, Any] = {
            "codec_type": "video",
            "codec_name": match.group(1),
            "width": int(match.group(2)),
            "height": int(match.group(3)),
        }
        if duration is not None:
            video_stream["duration"] = str(duration)
        streams.append(video_stream)

    for match in _AUDIO_STREAM_RE.finditer(text):
        streams.append(
            {
                "codec_type": "audio",
                "codec_name": match.group(1),
            }
        )

    if not streams:
        return None

    fmt: dict[str, Any] = {"format_name": "unknown"}
    if duration is not None:
        fmt["duration"] = str(duration)

    return {
        "format": fmt,
        "streams": streams,
    }
