"""Tests for video transcription and SRT helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from liteness.plugins.video_transcribe import (
    seconds_to_srt_timestamp,
    transcribe_video_handler,
    write_srt,
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
    runtime.platform_hosts = ("youtube.com",)
    runtime.cleanup_temp_on_uninstall = True
    runtime.ytdlp_format = "best"
    runtime.whisper_model = "base"
    runtime.whisper_device = "cpu"
    runtime.whisper_compute_type = "int8"
    runtime.whisper_language = None
    return runtime


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00:00,000"),
        (3.2, "00:00:03,200"),
        (90.5, "00:01:30,500"),
        (3661.001, "01:01:01,001"),
    ],
)
def test_seconds_to_srt_timestamp(seconds: float, expected: str) -> None:
    assert seconds_to_srt_timestamp(seconds) == expected


def test_write_srt(tmp_path: Path) -> None:
    srt_path = tmp_path / "clip.srt"
    write_srt(
        srt_path,
        [
            {"index": 1, "start": 0.0, "end": 3.2, "text": "Hello world"},
            {"index": 2, "start": 3.2, "end": 7.1, "text": "Second line"},
        ],
    )
    text = srt_path.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:03,200" in text
    assert "Hello world" in text
    assert "00:00:03,200 --> 00:00:07,100" in text
    assert "Second line" in text


def test_transcribe_video_requires_input(tmp_path: Path) -> None:
    result = transcribe_video_handler(_runtime(tmp_path))("c1", {})
    assert result.is_error
    assert result.error_code == "INVALID_ARGS"


def test_transcribe_video_no_audio_stream(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    probe_data = {"streams": [{"codec_type": "video"}], "format": {"duration": "10"}}

    with patch(
        "liteness.plugins.video_transcribe.probe_video_info",
        return_value=probe_data,
    ):
        result = transcribe_video_handler(_runtime(tmp_path))(
            "c1",
            {"input": str(video)},
        )

    assert result.is_error
    assert result.error_code == "NO_AUDIO"


def test_transcribe_video_success(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    probe_data = {
        "streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
        "format": {"duration": "12.5"},
    }
    fake_segments = [
        SimpleNamespace(start=0.0, end=2.5, text=" Hello "),
        SimpleNamespace(start=2.5, end=5.0, text="World"),
    ]
    fake_info = SimpleNamespace(language="en", duration=12.5)

    with patch(
        "liteness.plugins.video_transcribe.probe_video_info",
        return_value=probe_data,
    ), patch(
        "liteness.plugins.video_transcribe.run_command",
        return_value=(True, ""),
    ), patch(
        "liteness.plugins.video_transcribe._load_whisper_model"
    ) as load_model:
        model = MagicMock()
        model.transcribe.return_value = (iter(fake_segments), fake_info)
        load_model.return_value = model

        result = transcribe_video_handler(_runtime(tmp_path))(
            "c1",
            {"input": str(video)},
        )

    assert not result.is_error
    payload = json.loads(result.content)
    assert payload["language"] == "en"
    assert payload["segment_count"] == 2
    assert payload["segments"][0]["text"] == "Hello"
    srt_path = Path(payload["srt_path"])
    assert srt_path.exists()
    assert "Hello" in srt_path.read_text(encoding="utf-8")
