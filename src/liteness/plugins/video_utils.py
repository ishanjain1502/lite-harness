"""Shared utilities for the video plugin."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from liteness.tools import ToolResult

_TIME_RE = re.compile(
    r"^(?:(\d+):)?(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?$|^(\d+(?:\.\d+)?)$"
)

DEFAULT_PLATFORM_HOSTS = (
    "youtube.com",
    "youtu.be",
    "vimeo.com",
    "dailymotion.com",
)

VALID_EDIT_ACTIONS = frozenset(
    {
        "trim_clip",
        "cut_segment",
        "concat_clips",
        "speed_change",
        "mute_audio",
        "extract_audio",
        "add_text_overlay",
        "crop",
        "resize",
    }
)


def invalid_args(call_id: str, name: str, message: str) -> ToolResult:
    return ToolResult(
        call_id=call_id,
        name=name,
        content=message,
        is_error=True,
        error_code="INVALID_ARGS",
    )


def parse_time(value: str | float | int) -> float:
    """Parse seconds float or HH:MM:SS[.ms] / MM:SS[.ms] string to seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("time must be a number or HH:MM:SS string")

    text = value.strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)

    match = _TIME_RE.match(text)
    if not match:
        raise ValueError(f"invalid time format: {value!r}")

    if match.group(5) is not None:
        return float(match.group(5))

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    millis = match.group(4)
    frac = int(millis.ljust(3, "0")[:3]) / 1000.0 if millis else 0.0
    return hours * 3600 + minutes * 60 + seconds + frac


def format_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    whole = int(seconds)
    frac_ms = int(round((seconds - whole) * 1000))
    if frac_ms == 1000:
        whole += 1
        frac_ms = 0
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}.{frac_ms:03d}".rstrip("0").rstrip(".")
    return f"{m}:{s:02d}.{frac_ms:03d}".rstrip("0").rstrip(".")


def resolve_path(raw: str, *, workspace: Path | None) -> Path:
    path = Path(raw)
    if not path.is_absolute() and workspace is not None:
        path = workspace / path
    return path.resolve()


def find_binary(name: str, configured: str | None) -> str:
    if configured:
        return configured
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"{name} not found on PATH; install ffmpeg or set ffmpeg_path in plugin config"
    )


def is_http_url(source: str) -> bool:
    try:
        parsed = urlparse(source.strip())
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def sanitize_filename(name: str, *, max_len: int = 80) -> str:
    safe = re.sub(r"[^\w.-]+", "_", name).strip("_")
    return (safe or "video")[:max_len]


def edited_output_path(input_path: Path, suffix: str = ".mp4") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return input_path.parent / f"{input_path.stem}_edited_{ts}{suffix}"


def step_output_path(
    input_path: Path,
    label: str,
    temp_dir: Path,
    suffix: str = ".mp4",
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = sanitize_filename(label)
    return temp_dir / f"{input_path.stem}_{safe_label}_{ts}{suffix}"


class VideoRuntime:
    """Per-install ffmpeg paths, workspace, downloads, and vision settings."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.ffmpeg = find_binary("ffmpeg", config.get("ffmpeg_path"))
        self.ffprobe = find_binary("ffprobe", config.get("ffprobe_path"))
        workspace = config.get("workspace")
        self.workspace = Path(workspace).resolve() if workspace else None

        temp_dir = config.get("temp_dir", "./.liteness-video-temp")
        self.temp_dir = Path(temp_dir)
        if not self.temp_dir.is_absolute() and self.workspace is not None:
            self.temp_dir = self.workspace / self.temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        output_dir = config.get("output_dir", "./video-output")
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute() and self.workspace is not None:
            self.output_dir = self.workspace / self.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.vision_provider = config.get("vision_provider")
        self.vision_model = config.get("vision_model", "gemini-2.0-flash")
        self.max_frame_count = int(config.get("max_frame_count", 12))
        self.download_timeout_s = float(config.get("download_timeout_s", 600))
        self.max_download_size_mb = int(config.get("max_download_size_mb", 500))
        hosts = config.get("platform_hosts", list(DEFAULT_PLATFORM_HOSTS))
        self.platform_hosts = tuple(str(h).lower() for h in hosts)
        self.cleanup_temp_on_uninstall = bool(config.get("cleanup_temp_on_uninstall", True))
        self.ytdlp_format = config.get(
            "ytdlp_format",
            "bestvideo+bestaudio/best",
        )

    def resolve(self, raw: str) -> Path:
        return resolve_path(raw, workspace=self.workspace)

    def default_output(self, input_path: Path, suffix: str = ".mp4") -> Path:
        return edited_output_path(input_path, suffix=suffix)

    def intermediate_output(self, input_path: Path, label: str, suffix: str = ".mp4") -> Path:
        return step_output_path(input_path, label, self.temp_dir, suffix=suffix)

    def is_platform_url(self, source: str) -> bool:
        if not is_http_url(source):
            return False
        host = urlparse(source.strip()).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return any(host == h or host.endswith(f".{h}") for h in self.platform_hosts)

    def cleanup_temp_dir(self) -> None:
        if not self.temp_dir.exists():
            return
        for path in self.temp_dir.iterdir():
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)


def run_command(
    args: list[str],
    *,
    timeout_s: float = 300.0,
) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"command timed out after {timeout_s}s"
    except OSError as exc:
        return False, str(exc)

    output = completed.stdout.strip()
    if completed.stderr.strip():
        output = f"{output}\n{completed.stderr.strip()}".strip()
    if completed.returncode != 0:
        return False, output or f"exit code {completed.returncode}"
    return True, output


def probe_duration(runtime: VideoRuntime, path: Path) -> float | None:
    ok, output = run_command(
        [
            runtime.ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        timeout_s=60.0,
    )
    if not ok:
        return None
    try:
        return float(output.strip())
    except ValueError:
        return None


def probe_video_info(runtime: VideoRuntime, path: Path) -> dict[str, Any] | None:
    ok, output = run_command(
        [
            runtime.ffprobe,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        timeout_s=60.0,
    )
    if not ok:
        return None
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return None


def resolve_output_path(
    runtime: VideoRuntime,
    input_path: Path,
    output_arg: Any,
    *,
    suffix: str = ".mp4",
) -> Path:
    if isinstance(output_arg, str) and output_arg:
        return runtime.resolve(output_arg)
    return runtime.default_output(input_path, suffix=suffix)
