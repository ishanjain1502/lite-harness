"""Tests for burned subtitle tooling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from liteness.plugins.video_subtitles import burn_subtitles_handler, ffmpeg_subtitles_path
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
    runtime.platform_hosts = ("youtube.com",)
    runtime.cleanup_temp_on_uninstall = True
    runtime.ytdlp_format = "best"
    runtime.subtitle_font_size = 24
    runtime.subtitle_margin_v = 30
    return runtime


def test_ffmpeg_subtitles_path_escapes_windows_drive(tmp_path: Path) -> None:
    path = tmp_path / "sub.srt"
    path.write_text("1\n", encoding="utf-8")
    escaped = ffmpeg_subtitles_path(path)
    assert "\\:" in escaped or "/" in escaped
    assert "sub.srt" in escaped


def test_burn_subtitles_requires_srt(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    result = burn_subtitles_handler(_runtime(tmp_path))(
        "c1",
        {"input": str(video)},
    )
    assert result.is_error
    assert result.error_code == "INVALID_ARGS"


def test_burn_subtitles_success(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    srt = tmp_path / "clip.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")

    with patch(
        "liteness.plugins.video_subtitles.run_command",
        return_value=(True, ""),
    ) as run_command:
        result = burn_subtitles_handler(_runtime(tmp_path))(
            "c1",
            {"input": str(video), "srt": str(srt)},
        )

    assert not result.is_error
    assert "subtitles burned" in result.content
    args = run_command.call_args[0][1]
    assert "-vf" in args
    vf_index = args.index("-vf")
    assert "subtitles=" in args[vf_index + 1]
