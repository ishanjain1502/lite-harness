"""Burn SRT subtitles into video via ffmpeg."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from liteness.plugins.video_utils import (
    VideoRuntime,
    invalid_args,
    run_command,
    subtitled_output_path,
)
from liteness.tools import ToolResult


def ffmpeg_subtitles_path(path: Path) -> str:
    text = path.resolve().as_posix()
    text = text.replace(":", "\\:")
    text = text.replace("'", "\\'")
    return text


def _subtitle_force_style(runtime: VideoRuntime, arguments: dict[str, Any]) -> str:
    font_size = arguments.get("font_size", runtime.subtitle_font_size)
    margin_v = arguments.get("margin_v", runtime.subtitle_margin_v)
    if not isinstance(font_size, int) or font_size < 8:
        font_size = runtime.subtitle_font_size
    if not isinstance(margin_v, int) or margin_v < 0:
        margin_v = runtime.subtitle_margin_v
    return f"FontSize={font_size},MarginV={margin_v},Alignment=2"


def burn_subtitles_handler(runtime: VideoRuntime) -> Any:
    def handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
        input_arg = arguments.get("input")
        srt_arg = arguments.get("srt")
        if not isinstance(input_arg, str) or not input_arg:
            return invalid_args(call_id, "burn_subtitles", "input is required")
        if not isinstance(srt_arg, str) or not srt_arg:
            return invalid_args(call_id, "burn_subtitles", "srt is required")

        input_path = runtime.resolve(input_arg)
        if not input_path.exists():
            return ToolResult(
                call_id=call_id,
                name="burn_subtitles",
                content=f"input not found: {input_path}",
                is_error=True,
                error_code="NOT_FOUND",
            )

        srt_path = runtime.resolve(srt_arg)
        if not srt_path.exists():
            return ToolResult(
                call_id=call_id,
                name="burn_subtitles",
                content=f"srt not found: {srt_path}",
                is_error=True,
                error_code="NOT_FOUND",
            )

        output_arg = arguments.get("output")
        if isinstance(output_arg, str) and output_arg:
            output_path = runtime.resolve(output_arg)
        else:
            output_path = subtitled_output_path(input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        style = _subtitle_force_style(runtime, arguments)
        escaped_srt = ffmpeg_subtitles_path(srt_path)
        vf = f"subtitles='{escaped_srt}':force_style='{style}'"

        ok, output = run_command(
            runtime,
            [
                runtime.ffmpeg,
                "-y",
                "-i",
                str(input_path),
                "-vf",
                vf,
                "-codec:v",
                "libx264",
                "-crf",
                "23",
                "-preset",
                "veryfast",
                "-codec:a",
                "copy",
                str(output_path),
            ],
            timeout_s=1200.0,
        )
        if not ok:
            return ToolResult(
                call_id=call_id,
                name="burn_subtitles",
                content=output,
                is_error=True,
                error_code="FFMPEG_ERROR",
            )

        return ToolResult(
            call_id=call_id,
            name="burn_subtitles",
            content=f"subtitles burned; output written to {output_path}",
        )

    return handler
