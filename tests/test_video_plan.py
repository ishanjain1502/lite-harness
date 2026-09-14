"""Tests for plan_edits and edit plan validation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from liteness.plugins.video_plan import plan_edits_handler, validate_edit_plan
from liteness.plugins.video_utils import VideoRuntime, edited_output_path


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


def test_validate_edit_plan_rejects_invalid_action() -> None:
    plan = {
        "steps": [
            {
                "step": 1,
                "action": "explode",
                "start": "0",
                "end": "5",
                "params": {},
            }
        ]
    }
    errors = validate_edit_plan(plan, duration_s=60.0)
    assert any("invalid action" in e for e in errors)


def test_validate_edit_plan_rejects_time_beyond_duration() -> None:
    plan = {
        "steps": [
            {
                "step": 1,
                "action": "cut_segment",
                "start": "0",
                "end": "120",
                "params": {},
            }
        ]
    }
    errors = validate_edit_plan(plan, duration_s=60.0)
    assert any("exceeds" in e for e in errors)


def test_plan_edits_metadata_fallback(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")

    analyze_result = MagicMock()
    analyze_result.is_error = False
    analyze_result.content = "duration_s: 60.0\nscene info"
    analyze_result.error_code = None

    def fake_analyze(call_id, args):  # noqa: ANN001
        return analyze_result

    with patch(
        "liteness.plugins.video_plan.probe_duration",
        return_value=60.0,
    ):
        result = plan_edits_handler(_runtime(tmp_path), fake_analyze)(
            "c1",
            {"input": str(video), "prompt": "remove the intro"},
        )

    assert not result.is_error
    assert "warning:" in result.content
    plan = json.loads(result.content.split("\n\n", 1)[-1])
    assert plan["video_path"] == str(video.resolve())
    assert plan["steps"]
    assert plan["steps"][0]["action"] == "cut_segment"


def test_plan_edits_requires_prompt(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")

    def fake_analyze(call_id, args):  # noqa: ANN001
        return MagicMock(is_error=False, content="ok", error_code=None)

    result = plan_edits_handler(_runtime(tmp_path), fake_analyze)(
        "c1",
        {"input": str(video)},
    )
    assert result.is_error
    assert result.error_code == "INVALID_ARGS"


def test_edited_output_path_suffix(tmp_path: Path) -> None:
    video = tmp_path / "vacation.mp4"
    out = edited_output_path(video)
    assert out.parent == video.parent
    assert out.name.startswith("vacation_edited_")
    assert out.suffix == ".mp4"
