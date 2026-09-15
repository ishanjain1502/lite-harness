"""Tests for video download / resolve_video."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from liteness.plugins.video_download import (
    _download_ytdlp,
    _enrich_ytdlp_error,
    _format_fallback_chain,
    resolve_video_handler,
)
from liteness.plugins.video_utils import VideoRuntime


def _runtime(tmp_path: Path) -> VideoRuntime:
    runtime = VideoRuntime.__new__(VideoRuntime)
    runtime.ffmpeg = "ffmpeg"
    runtime.ffprobe = "ffprobe"
    runtime.workspace = tmp_path
    runtime.temp_dir = tmp_path / "temp"
    runtime.temp_dir.mkdir(parents=True, exist_ok=True)
    runtime.output_dir = tmp_path / "out"
    runtime.output_dir.mkdir(parents=True, exist_ok=True)
    runtime.ctx = None
    runtime.max_frame_count = 12
    runtime.download_timeout_s = 30.0
    runtime.max_download_size_mb = 500
    runtime.platform_hosts = ("youtube.com", "youtu.be", "vimeo.com")
    runtime.cleanup_temp_on_uninstall = True
    runtime.ytdlp_format = "best"
    return runtime


def test_resolve_video_local_path(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")

    def fake_probe_duration(runtime, path):  # noqa: ANN001
        return 60.0

    with patch(
        "liteness.plugins.video_download.probe_duration",
        side_effect=fake_probe_duration,
    ):
        result = resolve_video_handler(_runtime(tmp_path))(
            "c1",
            {"source": str(video)},
        )

    assert not result.is_error
    assert "source_type: local" in result.content
    assert f"local_path: {video}" in result.content
    assert "duration_s: 60.000" in result.content


def test_resolve_video_local_missing(tmp_path: Path) -> None:
    result = resolve_video_handler(_runtime(tmp_path))(
        "c1",
        {"source": str(tmp_path / "missing.mp4")},
    )
    assert result.is_error
    assert result.error_code == "NOT_FOUND"


def test_resolve_video_http_download(tmp_path: Path) -> None:
    url = "https://example.com/video.mp4"
    dest = _runtime(tmp_path).temp_dir / "downloaded.mp4"
    dest.write_bytes(b"video-bytes")

    with patch(
        "liteness.plugins.video_download._download_http",
        return_value=(True, str(dest)),
    ), patch(
        "liteness.plugins.video_download.probe_duration",
        return_value=10.0,
    ):
        result = resolve_video_handler(_runtime(tmp_path))(
            "c1",
            {"source": url},
        )

    assert not result.is_error
    assert "source_type: http" in result.content


def test_format_fallback_chain_dedupes() -> None:
    chain = _format_fallback_chain("best")
    assert chain[0] == "best"
    assert "worst" in chain
    assert len(chain) == len(set(chain))


def test_enrich_ytdlp_error_adds_upgrade_hint() -> None:
    msg = _enrich_ytdlp_error("ERROR: Requested format is not available")
    assert "upgrade yt-dlp" in msg


def test_download_ytdlp_tries_fallback_formats(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    calls: list[str] = []

    def fake_try(runtime, url, *, dest_path, format_selector):  # noqa: ANN001
        calls.append(format_selector)
        if format_selector == "worst":
            dest_path.write_bytes(b"ok")
            return True, str(dest_path), "title"
        return False, f"failed {format_selector}", None

    with patch(
        "liteness.plugins.video_download._try_download_ytdlp",
        side_effect=fake_try,
    ):
        ok, message, title = _download_ytdlp(
            runtime,
            "https://youtu.be/abc",
            dest_path=tmp_path / "out.mp4",
            format_selector="bestvideo+bestaudio/best",
        )

    assert ok
    assert title == "title"
    assert len(calls) > 1
    assert calls[-1] == "worst"


def test_resolve_video_platform_requires_ytdlp(tmp_path: Path) -> None:
    with patch(
        "liteness.plugins.video_download._download_ytdlp",
        return_value=(False, "yt-dlp not installed; pip install 'lite-ness[video-download]'", None),
    ):
        result = resolve_video_handler(_runtime(tmp_path))(
            "c1",
            {"source": "https://www.youtube.com/watch?v=abc123"},
        )

    assert result.is_error
    assert result.error_code == "DOWNLOAD_UNAVAILABLE"
