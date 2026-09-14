"""Additional ffmpeg editing operations."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from liteness.plugins.video_utils import (
    VideoRuntime,
    format_time,
    invalid_args,
    parse_time,
    probe_duration,
    resolve_output_path,
    run_command,
)
from liteness.tools import ToolResult


def _ensure_input(
    call_id: str,
    tool_name: str,
    runtime: VideoRuntime,
    input_arg: Any,
) -> tuple[ToolResult | None, Path | None]:
    if not isinstance(input_arg, str) or not input_arg:
        return invalid_args(call_id, tool_name, "input is required"), None
    input_path = runtime.resolve(input_arg)
    if not input_path.exists():
        return (
            ToolResult(
                call_id=call_id,
                name=tool_name,
                content=f"input not found: {input_path}",
                is_error=True,
                error_code="NOT_FOUND",
            ),
            None,
        )
    return None, input_path


def _ffmpeg_error(call_id: str, tool_name: str, output: str) -> ToolResult:
    code = "FFMPEG_FILTER_ERROR" if "drawtext" in output.lower() else "FFMPEG_ERROR"
    return ToolResult(
        call_id=call_id,
        name=tool_name,
        content=output,
        is_error=True,
        error_code=code,
    )


def _concat_paths(
    runtime: VideoRuntime,
    paths: list[Path],
    output_path: Path,
) -> tuple[bool, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    ) as list_file:
        for path in paths:
            escaped = str(path).replace("'", "'\\''")
            list_file.write(f"file '{escaped}'\n")
        list_path = list_file.name

    try:
        return run_command(
            [
                runtime.ffmpeg,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                "-c",
                "copy",
                str(output_path),
            ],
            timeout_s=600.0,
        )
    finally:
        Path(list_path).unlink(missing_ok=True)


def _trim_to_file(
    runtime: VideoRuntime,
    input_path: Path,
    output_path: Path,
    *,
    start_s: float,
    end_s: float | None,
) -> tuple[bool, str]:
    args = [
        runtime.ffmpeg,
        "-y",
        "-ss",
        str(start_s),
        "-i",
        str(input_path),
    ]
    if end_s is not None:
        if end_s <= start_s:
            return False, "end must be greater than start"
        args.extend(["-t", str(end_s - start_s)])
    args.extend(["-c", "copy", str(output_path)])
    return run_command(args, timeout_s=600.0)


def cut_segment_handler(runtime: VideoRuntime) -> Any:
    def handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
        err, input_path = _ensure_input(call_id, "cut_segment", runtime, arguments.get("input"))
        if err:
            return err

        try:
            start_s = parse_time(arguments.get("start", 0))
            end_s = parse_time(arguments["end"])
        except (KeyError, ValueError) as exc:
            return invalid_args(call_id, "cut_segment", str(exc))

        duration = probe_duration(runtime, input_path)
        if duration is not None and end_s <= start_s:
            return invalid_args(call_id, "cut_segment", "end must be greater than start")
        if duration is not None and start_s >= duration:
            return invalid_args(call_id, "cut_segment", "start is beyond video duration")

        output_path = resolve_output_path(
            runtime,
            input_path,
            arguments.get("output"),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if start_s <= 0:
            ok, output = _trim_to_file(
                runtime,
                input_path,
                output_path,
                start_s=end_s,
                end_s=None,
            )
        elif duration is not None and end_s >= duration:
            ok, output = _trim_to_file(
                runtime,
                input_path,
                output_path,
                start_s=0.0,
                end_s=start_s,
            )
        else:
            part_a = runtime.intermediate_output(input_path, "cut_a")
            part_b = runtime.intermediate_output(input_path, "cut_b")
            ok_a, out_a = _trim_to_file(
                runtime,
                input_path,
                part_a,
                start_s=0.0,
                end_s=start_s,
            )
            if not ok_a:
                return _ffmpeg_error(call_id, "cut_segment", out_a)
            ok_b, out_b = _trim_to_file(
                runtime,
                input_path,
                part_b,
                start_s=end_s,
                end_s=None,
            )
            if not ok_b:
                part_a.unlink(missing_ok=True)
                return _ffmpeg_error(call_id, "cut_segment", out_b)
            ok, output = _concat_paths(runtime, [part_a, part_b], output_path)
            part_a.unlink(missing_ok=True)
            part_b.unlink(missing_ok=True)

        if not ok:
            return _ffmpeg_error(call_id, "cut_segment", output)

        return ToolResult(
            call_id=call_id,
            name="cut_segment",
            content=(
                f"cut segment removed; output written to {output_path}\n"
                f"removed: {format_time(start_s)} to {format_time(end_s)}"
            ),
        )

    return handler


def _atempo_chain(factor: float) -> str:
    if factor <= 0:
        raise ValueError("factor must be positive")
    parts: list[str] = []
    remaining = factor
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.6f}".rstrip("0").rstrip("."))
    return ",".join(parts)


def speed_change_handler(runtime: VideoRuntime) -> Any:
    def handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
        err, input_path = _ensure_input(call_id, "speed_change", runtime, arguments.get("input"))
        if err:
            return err

        factor = arguments.get("factor")
        if not isinstance(factor, (int, float)) or factor <= 0:
            return invalid_args(call_id, "speed_change", "factor must be a positive number")

        output_path = resolve_output_path(runtime, input_path, arguments.get("output"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            atempo = _atempo_chain(float(factor))
        except ValueError as exc:
            return invalid_args(call_id, "speed_change", str(exc))

        filter_complex = (
            f"[0:v]setpts=PTS/{float(factor)}[v];[0:a]{atempo}[a]"
        )
        ok, output = run_command(
            [
                runtime.ffmpeg,
                "-y",
                "-i",
                str(input_path),
                "-filter_complex",
                filter_complex,
                "-map",
                "[v]",
                "-map",
                "[a]",
                str(output_path),
            ],
            timeout_s=600.0,
        )
        if not ok:
            return _ffmpeg_error(call_id, "speed_change", output)

        return ToolResult(
            call_id=call_id,
            name="speed_change",
            content=f"speed changed by factor {factor}; output written to {output_path}",
        )

    return handler


def mute_audio_handler(runtime: VideoRuntime) -> Any:
    def handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
        err, input_path = _ensure_input(call_id, "mute_audio", runtime, arguments.get("input"))
        if err:
            return err

        output_path = resolve_output_path(runtime, input_path, arguments.get("output"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ok, output = run_command(
            [
                runtime.ffmpeg,
                "-y",
                "-i",
                str(input_path),
                "-an",
                "-c:v",
                "copy",
                str(output_path),
            ],
            timeout_s=600.0,
        )
        if not ok:
            return _ffmpeg_error(call_id, "mute_audio", output)

        return ToolResult(
            call_id=call_id,
            name="mute_audio",
            content=f"audio muted; output written to {output_path}",
        )

    return handler


def extract_audio_handler(runtime: VideoRuntime) -> Any:
    def handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
        err, input_path = _ensure_input(call_id, "extract_audio", runtime, arguments.get("input"))
        if err:
            return err

        output_path = resolve_output_path(
            runtime,
            input_path,
            arguments.get("output"),
            suffix=".mp3",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ok, output = run_command(
            [
                runtime.ffmpeg,
                "-y",
                "-i",
                str(input_path),
                "-vn",
                "-acodec",
                "libmp3lame",
                str(output_path),
            ],
            timeout_s=600.0,
        )
        if not ok:
            return _ffmpeg_error(call_id, "extract_audio", output)

        return ToolResult(
            call_id=call_id,
            name="extract_audio",
            content=f"audio extracted to {output_path}",
        )

    return handler


def _text_position_expr(position: str) -> tuple[str, str]:
    mapping = {
        "top": ("(w-text_w)/2", "50"),
        "center": ("(w-text_w)/2", "(h-text_h)/2"),
        "bottom": ("(w-text_w)/2", "h-text_h-50"),
    }
    return mapping.get(position, mapping["bottom"])


def add_text_overlay_handler(runtime: VideoRuntime) -> Any:
    def handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
        err, input_path = _ensure_input(
            call_id,
            "add_text_overlay",
            runtime,
            arguments.get("input"),
        )
        if err:
            return err

        text = arguments.get("text")
        if not isinstance(text, str) or not text:
            return invalid_args(call_id, "add_text_overlay", "text is required")

        try:
            start_s = parse_time(arguments.get("start", 0))
            end_raw = arguments.get("end")
            end_s = parse_time(end_raw) if end_raw is not None else None
        except ValueError as exc:
            return invalid_args(call_id, "add_text_overlay", str(exc))

        position = arguments.get("position", "bottom")
        if not isinstance(position, str):
            position = "bottom"
        position = position.lower()
        if position not in {"top", "center", "bottom"}:
            return invalid_args(
                call_id,
                "add_text_overlay",
                "position must be top, center, or bottom",
            )

        font_size = arguments.get("font_size", 24)
        if not isinstance(font_size, int) or font_size < 8:
            return invalid_args(call_id, "add_text_overlay", "font_size must be an integer >= 8")

        output_path = resolve_output_path(runtime, input_path, arguments.get("output"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        x_expr, y_expr = _text_position_expr(position)
        escaped = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        if end_s is not None:
            enable = f"between(t\\,{start_s}\\,{end_s})"
        else:
            enable = f"gte(t\\,{start_s})"

        vf = (
            f"drawtext=text='{escaped}':fontsize={font_size}:"
            f"x={x_expr}:y={y_expr}:enable='{enable}'"
        )
        ok, output = run_command(
            [
                runtime.ffmpeg,
                "-y",
                "-i",
                str(input_path),
                "-vf",
                vf,
                "-codec:a",
                "copy",
                str(output_path),
            ],
            timeout_s=600.0,
        )
        if not ok:
            return _ffmpeg_error(call_id, "add_text_overlay", output)

        return ToolResult(
            call_id=call_id,
            name="add_text_overlay",
            content=f"text overlay added; output written to {output_path}",
        )

    return handler


def crop_handler(runtime: VideoRuntime) -> Any:
    def handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
        err, input_path = _ensure_input(call_id, "crop", runtime, arguments.get("input"))
        if err:
            return err

        width = arguments.get("width")
        height = arguments.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1:
            return invalid_args(call_id, "crop", "width and height must be positive integers")

        x = arguments.get("x", 0)
        y = arguments.get("y", 0)
        if not isinstance(x, int) or not isinstance(y, int) or x < 0 or y < 0:
            return invalid_args(call_id, "crop", "x and y must be non-negative integers")

        output_path = resolve_output_path(runtime, input_path, arguments.get("output"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ok, output = run_command(
            [
                runtime.ffmpeg,
                "-y",
                "-i",
                str(input_path),
                "-vf",
                f"crop={width}:{height}:{x}:{y}",
                "-codec:a",
                "copy",
                str(output_path),
            ],
            timeout_s=600.0,
        )
        if not ok:
            return _ffmpeg_error(call_id, "crop", output)

        return ToolResult(
            call_id=call_id,
            name="crop",
            content=f"cropped to {width}x{height}; output written to {output_path}",
        )

    return handler


def resize_handler(runtime: VideoRuntime) -> Any:
    def handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
        err, input_path = _ensure_input(call_id, "resize", runtime, arguments.get("input"))
        if err:
            return err

        width = arguments.get("width")
        height = arguments.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1:
            return invalid_args(call_id, "resize", "width and height must be positive integers")

        output_path = resolve_output_path(runtime, input_path, arguments.get("output"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ok, output = run_command(
            [
                runtime.ffmpeg,
                "-y",
                "-i",
                str(input_path),
                "-vf",
                f"scale={width}:{height}",
                "-codec:a",
                "copy",
                str(output_path),
            ],
            timeout_s=600.0,
        )
        if not ok:
            return _ffmpeg_error(call_id, "resize", output)

        return ToolResult(
            call_id=call_id,
            name="resize",
            content=f"resized to {width}x{height}; output written to {output_path}",
        )

    return handler
