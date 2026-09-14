"""Video acquisition — local paths, HTTP downloads, and platform URLs."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from liteness.plugins.video_utils import (
    VideoRuntime,
    invalid_args,
    is_http_url,
    probe_duration,
    probe_video_info,
    resolve_path,
    sanitize_filename,
)
from liteness.tools import ToolResult


def _download_http(
    runtime: VideoRuntime,
    url: str,
    *,
    dest_path: Path,
) -> tuple[bool, str]:
    try:
        import httpx
    except ImportError:
        return (
            False,
            "httpx not installed; pip install 'lite-ness[video-download]'",
        )

    max_bytes = runtime.max_download_size_mb * 1024 * 1024
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with httpx.stream(
            "GET",
            url,
            follow_redirects=True,
            timeout=runtime.download_timeout_s,
        ) as response:
            if response.status_code >= 400:
                return False, f"HTTP {response.status_code} for {url}"

            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > max_bytes:
                return (
                    False,
                    f"file exceeds max_download_size_mb ({runtime.max_download_size_mb})",
                )

            written = 0
            with dest_path.open("wb") as fh:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    written += len(chunk)
                    if written > max_bytes:
                        dest_path.unlink(missing_ok=True)
                        return (
                            False,
                            f"file exceeds max_download_size_mb ({runtime.max_download_size_mb})",
                        )
                    fh.write(chunk)
    except Exception as exc:  # noqa: BLE001 — surface download errors to agent
        dest_path.unlink(missing_ok=True)
        return False, f"download failed: {exc}"

    return True, str(dest_path)


def _download_ytdlp(
    runtime: VideoRuntime,
    url: str,
    *,
    dest_path: Path,
    format_selector: str,
) -> tuple[bool, str, str | None]:
    try:
        import yt_dlp
    except ImportError:
        return (
            False,
            "yt-dlp not installed; pip install 'lite-ness[video-download]'",
            None,
        )

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    out_template = str(dest_path.with_suffix("")) + ".%(ext)s"
    title: str | None = None

    opts: dict[str, Any] = {
        "format": format_selector,
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": runtime.download_timeout_s,
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info:
                title = info.get("title")
    except Exception as exc:  # noqa: BLE001
        return False, f"yt-dlp download failed: {exc}", None

    candidates = sorted(dest_path.parent.glob(dest_path.stem + "*"))
    if not candidates:
        return False, "yt-dlp did not produce an output file", title

    downloaded = candidates[0]
    if downloaded != dest_path:
        if dest_path.exists():
            dest_path.unlink(missing_ok=True)
        downloaded.rename(dest_path)

    return True, str(dest_path), title


def _guess_title_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    name = Path(path).name if path else "video"
    return sanitize_filename(name.rsplit(".", 1)[0] if "." in name else name)


def resolve_video_handler(runtime: VideoRuntime) -> Any:
    def handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
        source_arg = arguments.get("source")
        if not isinstance(source_arg, str) or not source_arg.strip():
            return invalid_args(call_id, "resolve_video", "source is required")

        source = source_arg.strip()
        format_arg = arguments.get("format")
        format_selector = (
            format_arg if isinstance(format_arg, str) and format_arg else runtime.ytdlp_format
        )

        if is_http_url(source):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            if runtime.is_platform_url(source):
                title_hint = _guess_title_from_url(source)
                dest = runtime.temp_dir / f"{title_hint}_{ts}.mp4"
                ok, message, title = _download_ytdlp(
                    runtime,
                    source,
                    dest_path=dest,
                    format_selector=format_selector,
                )
                if not ok:
                    code = (
                        "DOWNLOAD_UNAVAILABLE"
                        if "not installed" in message
                        else "DOWNLOAD_ERROR"
                    )
                    return ToolResult(
                        call_id=call_id,
                        name="resolve_video",
                        content=message,
                        is_error=True,
                        error_code=code,
                    )
                local_path = Path(message)
                duration = probe_duration(runtime, local_path)
                lines = [
                    f"local_path: {local_path}",
                    "source_type: platform",
                    f"source: {source}",
                ]
                if title:
                    lines.append(f"title: {title}")
                if duration is not None:
                    lines.append(f"duration_s: {duration:.3f}")
                return ToolResult(call_id=call_id, name="resolve_video", content="\n".join(lines))

            title_hint = _guess_title_from_url(source)
            url_hash = hashlib.sha256(source.encode()).hexdigest()[:8]
            dest = runtime.temp_dir / f"{title_hint}_{url_hash}_{ts}.mp4"
            ok, message = _download_http(runtime, source, dest_path=dest)
            if not ok:
                code = (
                    "DOWNLOAD_UNAVAILABLE"
                    if "not installed" in message
                    else "DOWNLOAD_TOO_LARGE"
                    if "exceeds max_download_size_mb" in message
                    else "DOWNLOAD_ERROR"
                )
                return ToolResult(
                    call_id=call_id,
                    name="resolve_video",
                    content=message,
                    is_error=True,
                    error_code=code,
                )
            local_path = Path(message)
            duration = probe_duration(runtime, local_path)
            lines = [
                f"local_path: {local_path}",
                "source_type: http",
                f"source: {source}",
            ]
            if duration is not None:
                lines.append(f"duration_s: {duration:.3f}")
            return ToolResult(call_id=call_id, name="resolve_video", content="\n".join(lines))

        path = resolve_path(source, workspace=runtime.workspace)
        if not path.exists():
            return ToolResult(
                call_id=call_id,
                name="resolve_video",
                content=(
                    f"file not found: {path}. If this is a URL, pass the full "
                    "http(s):// address to resolve_video."
                ),
                is_error=True,
                error_code="NOT_FOUND",
            )

        duration = probe_duration(runtime, path)
        info = probe_video_info(runtime, path)
        title = None
        if info:
            title = info.get("format", {}).get("tags", {}).get("title")

        lines = [
            f"local_path: {path}",
            "source_type: local",
        ]
        if title:
            lines.append(f"title: {title}")
        if duration is not None:
            lines.append(f"duration_s: {duration:.3f}")
        return ToolResult(call_id=call_id, name="resolve_video", content="\n".join(lines))

    return handler


def cleanup_temp_handler(runtime: VideoRuntime) -> Any:
    def handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
        stem_arg = arguments.get("stem")
        max_age_s = arguments.get("max_age_s")

        removed = 0
        now = time.time()
        for path in runtime.temp_dir.iterdir():
            if not path.is_file():
                continue
            if isinstance(stem_arg, str) and stem_arg and stem_arg not in path.name:
                continue
            if isinstance(max_age_s, (int, float)) and max_age_s > 0:
                if now - path.stat().st_mtime <= float(max_age_s):
                    continue
            path.unlink(missing_ok=True)
            removed += 1

        return ToolResult(
            call_id=call_id,
            name="cleanup_temp",
            content=f"removed {removed} file(s) from {runtime.temp_dir}",
        )

    return handler
