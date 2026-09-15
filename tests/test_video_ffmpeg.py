"""Tests for ffmpeg auto-resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from liteness.plugins.video_ffmpeg import (
    FFmpegUnavailable,
    probe_duration_with_ffmpeg,
    probe_video_info_with_ffmpeg,
    resolve_ffmpeg_binaries,
)
from liteness.plugins.video_utils import VideoRuntime
from liteness.testing import registry_with_plugins


def test_video_plugin_installs_without_system_ffmpeg() -> None:
    names = set(registry_with_plugins("video").names())
    assert "trim_clip" in names


def test_resolve_auto_uses_system_when_available(tmp_path: Path) -> None:
    with patch(
        "liteness.plugins.video_ffmpeg._resolve_from_system",
        return_value=("/usr/bin/ffmpeg", "/usr/bin/ffprobe"),
    ):
        ffmpeg, ffprobe = resolve_ffmpeg_binaries(
            mode="auto",
            temp_dir=tmp_path,
            ffmpeg_path=None,
            ffprobe_path=None,
        )
    assert ffmpeg == "/usr/bin/ffmpeg"
    assert ffprobe == "/usr/bin/ffprobe"


def test_resolve_auto_downloads_when_system_missing(tmp_path: Path) -> None:
    with patch(
        "liteness.plugins.video_ffmpeg._resolve_from_system",
        return_value=None,
    ), patch(
        "liteness.plugins.video_ffmpeg._resolve_bundled",
        return_value=("/cache/ffmpeg", None),
    ):
        ffmpeg, ffprobe = resolve_ffmpeg_binaries(
            mode="auto",
            temp_dir=tmp_path,
            ffmpeg_path=None,
            ffprobe_path=None,
        )
    assert ffmpeg == "/cache/ffmpeg"
    assert ffprobe is None


def test_resolve_system_raises_when_missing(tmp_path: Path) -> None:
    with patch(
        "liteness.plugins.video_ffmpeg._resolve_from_system",
        return_value=None,
    ):
        with pytest.raises(FFmpegUnavailable):
            resolve_ffmpeg_binaries(
                mode="system",
                temp_dir=tmp_path,
                ffmpeg_path=None,
                ffprobe_path=None,
            )


def test_runtime_lazy_ensure_binaries(tmp_path: Path) -> None:
    runtime = VideoRuntime(
        {
            "ffmpeg_mode": "auto",
            "temp_dir": str(tmp_path / "temp"),
            "output_dir": str(tmp_path / "out"),
        }
    )
    assert runtime.ffmpeg is None
    with patch(
        "liteness.plugins.video_utils.resolve_ffmpeg_binaries",
        return_value=("/bin/ffmpeg", "/bin/ffprobe"),
    ):
        runtime.ensure_binaries()
    assert runtime.ffmpeg == "/bin/ffmpeg"
    assert runtime.ffprobe == "/bin/ffprobe"


def test_probe_duration_with_ffmpeg_parses_stderr() -> None:
    stderr = "Input #0, mp4, from 'clip.mp4':\n  Duration: 00:01:30.50"
    with patch(
        "liteness.plugins.video_ffmpeg._ffmpeg_probe_output",
        return_value=stderr,
    ):
        duration = probe_duration_with_ffmpeg("/bin/ffmpeg", Path("clip.mp4"))
    assert duration == pytest.approx(90.5)


def test_probe_video_info_with_ffmpeg_includes_audio_stream() -> None:
    stderr = (
        "Input #0, mov,mp4, from 'clip.mp4':\n"
        "  Duration: 00:06:55.36, start: 0.000000, bitrate: 1234 kb/s\n"
        "  Stream #0:0[0x1](und): Video: h264 (High), yuv420p, 1920x1080, 30 fps\n"
        "  Stream #0:1[0x2](eng): Audio: aac (LC), 44100 Hz, stereo, fltp, 128 kb/s\n"
    )
    with patch(
        "liteness.plugins.video_ffmpeg._ffmpeg_probe_output",
        return_value=stderr,
    ):
        info = probe_video_info_with_ffmpeg("/bin/ffmpeg", Path("clip.mp4"))

    assert info is not None
    streams = info["streams"]
    assert any(s.get("codec_type") == "video" for s in streams)
    assert any(
        s.get("codec_type") == "audio" and s.get("codec_name") == "aac" for s in streams
    )
    assert float(info["format"]["duration"]) == pytest.approx(415.36)


def test_probe_video_info_with_ffmpeg_audio_only() -> None:
    stderr = (
        "Input #0, mp3, from 'clip.mp3':\n"
        "  Duration: 00:01:00.00, start: 0.000000, bitrate: 128 kb/s\n"
        "  Stream #0:0: Audio: mp3, 44100 Hz, stereo, fltp, 128 kb/s\n"
    )
    with patch(
        "liteness.plugins.video_ffmpeg._ffmpeg_probe_output",
        return_value=stderr,
    ):
        info = probe_video_info_with_ffmpeg("/bin/ffmpeg", Path("clip.mp3"))

    assert info is not None
    assert info["streams"] == [{"codec_type": "audio", "codec_name": "mp3"}]
