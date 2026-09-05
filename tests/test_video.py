"""Video plugin tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from liteness.plugins.video import (
    VideoRuntime,
    _parse_time,
    analyze_video_handler,
    concat_clips_handler,
    extract_frames_handler,
    get_video_info_handler,
    trim_clip_handler,
)
from liteness.session import Session
from liteness.testing import registry_with_plugins


def test_video_plugin_registers_all_tools() -> None:
    with patch("liteness.plugins.video._find_binary", return_value="ffmpeg"):
        names = set(registry_with_plugins("video").names())
    assert names == {
        "get_video_info",
        "analyze_video",
        "extract_frames",
        "trim_clip",
        "concat_clips",
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("12.5", 12.5),
        ("90", 90.0),
        ("1:30", 90.0),
        ("0:01:30", 90.0),
        ("1:02:03.5", 3723.5),
    ],
)
def test_parse_time(raw: str, expected: float) -> None:
    assert _parse_time(raw) == pytest.approx(expected)


def test_parse_time_invalid() -> None:
    with pytest.raises(ValueError):
        _parse_time("not-a-time")


def _runtime(tmp_path: Path) -> VideoRuntime:
    runtime = VideoRuntime.__new__(VideoRuntime)
    runtime.ffmpeg = "ffmpeg"
    runtime.ffprobe = "ffprobe"
    runtime.workspace = tmp_path
    runtime.output_dir = tmp_path / "out"
    runtime.output_dir.mkdir(parents=True, exist_ok=True)
    runtime.vision_provider = None
    runtime.vision_model = "gemini-2.0-flash"
    runtime.max_frame_count = 12
    return runtime


def test_get_video_info_success(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")

    probe_payload = {
        "format": {"duration": "120.5", "size": "1000", "format_name": "mp4"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30/1",
                "duration": "120.5",
            },
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    }

    def fake_run(args, **kwargs):  # noqa: ANN001
        completed = MagicMock()
        completed.returncode = 0
        completed.stdout = json.dumps(probe_payload)
        completed.stderr = ""
        return completed

    with patch("liteness.plugins.video.subprocess.run", side_effect=fake_run):
        result = get_video_info_handler(_runtime(tmp_path))(
            "c1", {"path": str(video)}
        )

    assert not result.is_error
    assert "duration_s: 120.500" in result.content
    assert "1920x1080" in result.content
    assert "audio_codec: aac" in result.content


def test_trim_clip_requires_input(tmp_path: Path) -> None:
    result = trim_clip_handler(_runtime(tmp_path))("c1", {})
    assert result.is_error
    assert result.error_code == "INVALID_ARGS"


def test_trim_clip_end_before_start(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    result = trim_clip_handler(_runtime(tmp_path))(
        "c1",
        {"input": str(video), "start": "1:00", "end": "0:30"},
    )
    assert result.is_error
    assert result.error_code == "INVALID_ARGS"


def test_trim_clip_writes_output(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")

    def fake_run(args, **kwargs):  # noqa: ANN001
        out_path = Path(args[-1])
        out_path.write_bytes(b"edited")
        completed = MagicMock()
        completed.returncode = 0
        completed.stdout = ""
        completed.stderr = ""
        return completed

    with patch("liteness.plugins.video.subprocess.run", side_effect=fake_run):
        result = trim_clip_handler(_runtime(tmp_path))(
            "c1",
            {"input": str(video), "start": "10", "end": "30"},
        )

    assert not result.is_error
    assert "trimmed video written to" in result.content


def test_concat_clips_requires_inputs(tmp_path: Path) -> None:
    result = concat_clips_handler(_runtime(tmp_path))("c1", {"inputs": []})
    assert result.is_error
    assert result.error_code == "INVALID_ARGS"


def test_concat_clips_success(tmp_path: Path) -> None:
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_bytes(b"a")
    b.write_bytes(b"b")

    def fake_run(args, **kwargs):  # noqa: ANN001
        out_path = Path(args[-1])
        out_path.write_bytes(b"ab")
        completed = MagicMock()
        completed.returncode = 0
        completed.stdout = ""
        completed.stderr = ""
        return completed

    with patch("liteness.plugins.video.subprocess.run", side_effect=fake_run):
        result = concat_clips_handler(_runtime(tmp_path))(
            "c1",
            {"inputs": [str(a), str(b)]},
        )

    assert not result.is_error
    assert "concatenated 2 clips" in result.content


def test_extract_frames_evenly_spaced(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")

    calls: list[list[str]] = []

    def fake_run(args, **kwargs):  # noqa: ANN001
        calls.append(list(args))
        completed = MagicMock()
        completed.returncode = 0
        if "format=duration" in " ".join(args):
            completed.stdout = "100.0"
        else:
            out_path = Path(args[-1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"jpg")
            completed.stdout = ""
        completed.stderr = ""
        return completed

    with patch("liteness.plugins.video.subprocess.run", side_effect=fake_run):
        result = extract_frames_handler(_runtime(tmp_path))(
            "c1",
            {"input": str(video), "count": 2},
        )

    assert not result.is_error
    assert result.content.count("->") == 2


def test_analyze_video_without_vision(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")

    def fake_run(args, **kwargs):  # noqa: ANN001
        completed = MagicMock()
        completed.returncode = 0
        if "-print_format" in args:
            completed.stdout = json.dumps(
                {
                    "format": {"duration": "60.0", "size": "1", "format_name": "mp4"},
                    "streams": [
                        {
                            "codec_type": "video",
                            "codec_name": "h264",
                            "width": 1280,
                            "height": 720,
                            "r_frame_rate": "24/1",
                        }
                    ],
                }
            )
        elif "format=duration" in " ".join(args):
            completed.stdout = "60.0"
        else:
            out_path = Path(args[-1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"jpg")
            completed.stdout = ""
        completed.stderr = ""
        return completed

    with patch("liteness.plugins.video.subprocess.run", side_effect=fake_run):
        result = analyze_video_handler(_runtime(tmp_path))(
            "c1",
            {"input": str(video), "frame_count": 2},
        )

    assert not result.is_error
    assert "extracted_frames" in result.content
    assert "vision_provider not configured" in result.content


def test_load_video_editor_preset() -> None:
    from liteness.presets import load_preset

    preset = load_preset("video_editor")
    assert preset.name == "video_editor"
    assert [p.name for p in preset.plugins] == ["filesystem", "video"]
    assert preset.system_prompt is not None
    assert "analyze_video" in preset.system_prompt


def test_video_editor_preset_run_with_mock_llm(tmp_path: Path) -> None:
    from unittest.mock import MagicMock, patch

    from liteness.harness import create_runtime, dispose_runtime
    from liteness.llm import MockLLMProvider, MockStep, ToolCallDraft
    from liteness.loop import AgentLoop, LoopConfig

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")

    probe_payload = {
        "format": {"duration": "60.0", "size": "100", "format_name": "mp4"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "r_frame_rate": "30/1",
            }
        ],
    }

    def fake_run(args, **kwargs):  # noqa: ANN001
        completed = MagicMock()
        completed.returncode = 0
        if "-print_format" in args:
            completed.stdout = json.dumps(probe_payload)
        else:
            completed.stdout = ""
        completed.stderr = ""
        return completed

    llm = MockLLMProvider(
        steps=[
            MockStep(
                step=1,
                tool_calls=[
                    ToolCallDraft(
                        call_id="info1",
                        name="get_video_info",
                        arguments={"path": str(video)},
                    )
                ],
            ),
            MockStep(step=2, content="Video is 60 seconds, 1280x720."),
        ]
    )

    session = Session()
    with patch("liteness.plugins.video._find_binary", return_value="ffmpeg"), patch(
        "liteness.plugins.video.subprocess.run",
        side_effect=fake_run,
    ):
        runtime = create_runtime(preset_name="video_editor", session=session)
        try:
            loop = AgentLoop(
                llm=llm,
                tools=runtime.ctx.tools,
                config=LoopConfig(system_prompt=runtime.preset.system_prompt),
            )
            result = loop.run_turn(session, f"How long is {video.name}?")
            assert result.status == "completed"
            assert any(
                e.type == "tool/call" and e.payload["name"] == "get_video_info"
                for e in session.events
            )
        finally:
            dispose_runtime(runtime)
