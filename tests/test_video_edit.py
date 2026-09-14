"""Tests for extended video edit handlers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from liteness.plugins.video_edit import (
    cut_segment_handler,
    mute_audio_handler,
    resize_handler,
    speed_change_handler,
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
    runtime.vision_provider = None
    runtime.vision_model = "gemini-2.0-flash"
    runtime.max_frame_count = 12
    runtime.download_timeout_s = 30.0
    runtime.max_download_size_mb = 500
    runtime.platform_hosts = ("youtube.com",)
    runtime.cleanup_temp_on_uninstall = True
    runtime.ytdlp_format = "best"
    return runtime


def test_cut_segment_requires_end(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    result = cut_segment_handler(_runtime(tmp_path))(
        "c1",
        {"input": str(video), "start": "0"},
    )
    assert result.is_error
    assert result.error_code == "INVALID_ARGS"


def test_cut_segment_middle_removal(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")

    with patch("liteness.plugins.video_edit.probe_duration", return_value=120.0), patch(
        "liteness.plugins.video_edit.run_command",
        return_value=(True, ""),
    ):
        result = cut_segment_handler(_runtime(tmp_path))(
            "c1",
            {"input": str(video), "start": "10", "end": "20"},
        )

    assert not result.is_error
    assert "cut segment removed" in result.content


def test_speed_change_requires_factor(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    result = speed_change_handler(_runtime(tmp_path))(
        "c1",
        {"input": str(video)},
    )
    assert result.is_error
    assert result.error_code == "INVALID_ARGS"


def test_mute_audio_success(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")

    with patch(
        "liteness.plugins.video_edit.run_command",
        return_value=(True, ""),
    ):
        result = mute_audio_handler(_runtime(tmp_path))(
            "c1",
            {"input": str(video)},
        )

    assert not result.is_error
    assert "audio muted" in result.content


def test_resize_invalid_dimensions(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    result = resize_handler(_runtime(tmp_path))(
        "c1",
        {"input": str(video), "width": 0, "height": 720},
    )
    assert result.is_error
    assert result.error_code == "INVALID_ARGS"
