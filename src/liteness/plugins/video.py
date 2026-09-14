"""Video plugin — natural-language video editing via ffmpeg tools and vision analysis."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

from liteness.context import Context
from liteness.plugins.video_download import cleanup_temp_handler, resolve_video_handler
from liteness.plugins.video_edit import (
    add_text_overlay_handler,
    crop_handler,
    cut_segment_handler,
    extract_audio_handler,
    mute_audio_handler,
    resize_handler,
    speed_change_handler,
)
from liteness.plugins.video_plan import plan_edits_handler
from liteness.plugins.video_utils import (
    VideoRuntime,
    find_binary,
    format_time,
    invalid_args,
    parse_time,
    probe_video_info,
    resolve_output_path,
    run_command,
)
from liteness.tools import ToolDefinition, ToolResult

# Backward-compatible aliases for tests and external imports.
_parse_time = parse_time
_format_time = format_time
_find_binary = find_binary
_invalid_args = invalid_args


def _describe_frames_with_gemini(
    frame_paths: list[Path],
    timestamps: list[float],
    *,
    model: str,
) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "analyze_video with vision requires google-genai: pip install 'lite-ness[google]'"
        ) from exc

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set")

    client = genai.Client(api_key=api_key)
    parts: list[Any] = [
        types.Part(
            text=(
                "You are helping a video editor agent. Describe what happens in this "
                "video at each timestamp. For each frame, note the timestamp, visible "
                "scene, people/objects, on-screen text, and whether it looks like an "
                "intro/outro/transition. Be concise and structured."
            )
        )
    ]
    for path, t in zip(frame_paths, timestamps, strict=True):
        image_bytes = path.read_bytes()
        parts.append(types.Part(text=f"Timestamp {format_time(t)} ({t:.3f}s):"))
        parts.append(
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        )

    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=parts)],
    )
    text = getattr(response, "text", None)
    if not text:
        return "Vision model returned no description."
    return text


def get_video_info_handler(runtime: VideoRuntime) -> Any:
    def handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
        path_arg = arguments.get("path")
        if not isinstance(path_arg, str) or not path_arg:
            return invalid_args(call_id, "get_video_info", "path is required")

        path = runtime.resolve(path_arg)
        if not path.exists():
            return ToolResult(
                call_id=call_id,
                name="get_video_info",
                content=f"file not found: {path}",
                is_error=True,
                error_code="NOT_FOUND",
            )

        data = probe_video_info(runtime, path)
        if data is None:
            return ToolResult(
                call_id=call_id,
                name="get_video_info",
                content="failed to probe video",
                is_error=True,
                error_code="PROBE_ERROR",
            )

        video_stream = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
            None,
        )
        audio_stream = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "audio"),
            None,
        )
        fmt = data.get("format", {})
        duration = float(fmt.get("duration") or (video_stream or {}).get("duration") or 0)

        lines = [
            f"path: {path}",
            f"duration_s: {duration:.3f}",
            f"duration: {format_time(duration)}",
            f"size_bytes: {fmt.get('size', 'unknown')}",
            f"format: {fmt.get('format_long_name', fmt.get('format_name', 'unknown'))}",
        ]
        if video_stream:
            lines.extend(
                [
                    f"video_codec: {video_stream.get('codec_name', 'unknown')}",
                    f"resolution: {video_stream.get('width')}x{video_stream.get('height')}",
                    f"fps: {video_stream.get('r_frame_rate', 'unknown')}",
                ]
            )
        if audio_stream:
            lines.append(f"audio_codec: {audio_stream.get('codec_name', 'unknown')}")

        return ToolResult(
            call_id=call_id,
            name="get_video_info",
            content="\n".join(lines),
        )

    return handler


def trim_clip_handler(runtime: VideoRuntime) -> Any:
    def handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
        input_arg = arguments.get("input")
        if not isinstance(input_arg, str) or not input_arg:
            return invalid_args(call_id, "trim_clip", "input is required")

        try:
            start_s = parse_time(arguments.get("start", 0))
            end_raw = arguments.get("end")
            end_s = parse_time(end_raw) if end_raw is not None else None
        except ValueError as exc:
            return invalid_args(call_id, "trim_clip", str(exc))

        input_path = runtime.resolve(input_arg)
        if not input_path.exists():
            return ToolResult(
                call_id=call_id,
                name="trim_clip",
                content=f"input not found: {input_path}",
                is_error=True,
                error_code="NOT_FOUND",
            )

        output_path = resolve_output_path(runtime, input_path, arguments.get("output"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

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
                return invalid_args(call_id, "trim_clip", "end must be greater than start")
            args.extend(["-t", str(end_s - start_s)])
        args.extend(["-c", "copy", str(output_path)])

        ok, output = run_command(args, timeout_s=600.0)
        if not ok:
            return ToolResult(
                call_id=call_id,
                name="trim_clip",
                content=output,
                is_error=True,
                error_code="FFMPEG_ERROR",
            )

        return ToolResult(
            call_id=call_id,
            name="trim_clip",
            content=(
                f"trimmed video written to {output_path}\n"
                f"start: {format_time(start_s)}\n"
                f"end: {format_time(end_s) if end_s is not None else 'EOF'}"
            ),
        )

    return handler


def concat_clips_handler(runtime: VideoRuntime) -> Any:
    def handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
        inputs = arguments.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            return invalid_args(call_id, "concat_clips", "inputs must be a non-empty list")
        if not all(isinstance(p, str) and p for p in inputs):
            return invalid_args(
                call_id, "concat_clips", "each input must be a non-empty string path"
            )

        paths = [runtime.resolve(p) for p in inputs]
        for path in paths:
            if not path.exists():
                return ToolResult(
                    call_id=call_id,
                    name="concat_clips",
                    content=f"input not found: {path}",
                    is_error=True,
                    error_code="NOT_FOUND",
                )

        anchor = paths[0]
        output_path = resolve_output_path(runtime, anchor, arguments.get("output"))
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
            ok, output = run_command(
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

        if not ok:
            return ToolResult(
                call_id=call_id,
                name="concat_clips",
                content=output,
                is_error=True,
                error_code="FFMPEG_ERROR",
            )

        return ToolResult(
            call_id=call_id,
            name="concat_clips",
            content=f"concatenated {len(paths)} clips to {output_path}",
        )

    return handler


def extract_frames_handler(runtime: VideoRuntime) -> Any:
    def handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
        input_arg = arguments.get("input")
        if not isinstance(input_arg, str) or not input_arg:
            return invalid_args(call_id, "extract_frames", "input is required")

        input_path = runtime.resolve(input_arg)
        if not input_path.exists():
            return ToolResult(
                call_id=call_id,
                name="extract_frames",
                content=f"input not found: {input_path}",
                is_error=True,
                error_code="NOT_FOUND",
            )

        count = arguments.get("count", 8)
        if not isinstance(count, int) or count < 1:
            return invalid_args(call_id, "extract_frames", "count must be a positive integer")

        count = min(count, runtime.max_frame_count)
        timestamps = arguments.get("timestamps")
        times: list[float] = []

        if timestamps is not None:
            if not isinstance(timestamps, list) or not timestamps:
                return invalid_args(
                    call_id,
                    "extract_frames",
                    "timestamps must be a non-empty list when provided",
                )
            try:
                times = [parse_time(t) for t in timestamps]
            except ValueError as exc:
                return invalid_args(call_id, "extract_frames", str(exc))
            times = times[: runtime.max_frame_count]
        else:
            from liteness.plugins.video_utils import probe_duration

            duration = probe_duration(runtime, input_path)
            if duration is None or duration <= 0:
                times = [0.0]
            elif count == 1:
                times = [duration / 2]
            else:
                step = duration / (count + 1)
                times = [step * (i + 1) for i in range(count)]

        out_dir_arg = arguments.get("output_dir")
        if isinstance(out_dir_arg, str) and out_dir_arg:
            out_dir = runtime.resolve(out_dir_arg)
        else:
            out_dir = runtime.output_dir / f"{input_path.stem}_frames"
        out_dir.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        for idx, t in enumerate(times):
            frame_path = out_dir / f"frame_{idx:03d}_{format_time(t).replace(':', '-')}.jpg"
            ok, output = run_command(
                [
                    runtime.ffmpeg,
                    "-y",
                    "-ss",
                    str(t),
                    "-i",
                    str(input_path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(frame_path),
                ],
                timeout_s=120.0,
            )
            if not ok:
                return ToolResult(
                    call_id=call_id,
                    name="extract_frames",
                    content=f"failed at t={format_time(t)}: {output}",
                    is_error=True,
                    error_code="FFMPEG_ERROR",
                )
            lines.append(f"t={format_time(t)} ({t:.3f}s) -> {frame_path}")

        return ToolResult(
            call_id=call_id,
            name="extract_frames",
            content="\n".join(lines),
        )

    return handler


def analyze_video_handler(runtime: VideoRuntime) -> Any:
    def handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
        input_arg = arguments.get("input")
        if not isinstance(input_arg, str) or not input_arg:
            return invalid_args(call_id, "analyze_video", "input is required")

        input_path = runtime.resolve(input_arg)
        if not input_path.exists():
            return ToolResult(
                call_id=call_id,
                name="analyze_video",
                content=f"input not found: {input_path}",
                is_error=True,
                error_code="NOT_FOUND",
            )

        count = arguments.get("frame_count", 8)
        if not isinstance(count, int) or count < 1:
            return invalid_args(
                call_id, "analyze_video", "frame_count must be a positive integer"
            )
        count = min(count, runtime.max_frame_count)

        frames_result = extract_frames_handler(runtime)(
            call_id,
            {"input": str(input_path), "count": count},
        )
        if frames_result.is_error:
            return ToolResult(
                call_id=call_id,
                name="analyze_video",
                content=frames_result.content,
                is_error=True,
                error_code=frames_result.error_code,
            )

        frame_lines = frames_result.content.splitlines()
        frame_paths: list[Path] = []
        timestamps: list[float] = []
        for line in frame_lines:
            if "->" not in line:
                continue
            left, right = line.split("->", 1)
            frame_paths.append(Path(right.strip()))
            ts_match = re.search(r"\(([\d.]+)s\)", left)
            if ts_match:
                timestamps.append(float(ts_match.group(1)))

        info_result = get_video_info_handler(runtime)(call_id, {"path": str(input_path)})
        header = info_result.content if not info_result.is_error else ""

        if runtime.vision_provider == "google":
            try:
                description = _describe_frames_with_gemini(
                    frame_paths,
                    timestamps,
                    model=runtime.vision_model,
                )
            except RuntimeError as exc:
                return ToolResult(
                    call_id=call_id,
                    name="analyze_video",
                    content=str(exc),
                    is_error=True,
                    error_code="VISION_UNAVAILABLE",
                )
            body = (
                f"{header}\n\n"
                f"extracted_frames:\n{frames_result.content}\n\n"
                f"scene_analysis:\n{description}"
            )
        else:
            body = (
                f"{header}\n\n"
                f"extracted_frames:\n{frames_result.content}\n\n"
                "scene_analysis: (vision_provider not configured — set vision_provider: "
                "google in plugin config and GOOGLE_API_KEY to enable automatic scene "
                "descriptions; otherwise infer edits from frame timestamps and user prompt)"
            )

        return ToolResult(call_id=call_id, name="analyze_video", content=body)

    return handler


def _register_core_tools(ctx: Context, runtime: VideoRuntime) -> None:
    analyze_handler = analyze_video_handler(runtime)

    ctx.register_tool(
        ToolDefinition(
            name="resolve_video",
            description=(
                "Acquire a local video path from a file path, direct HTTP URL, or "
                "platform URL (YouTube, Vimeo, etc.). Always call this first when "
                "the user provides a URL or before editing."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Local path, http(s) file URL, or platform URL",
                    },
                    "format": {
                        "type": "string",
                        "description": "yt-dlp format selector for platform URLs",
                    },
                },
                "required": ["source"],
            },
            handler=resolve_video_handler(runtime),
            timeout_s=runtime.download_timeout_s + 60.0,
            idempotent=True,
        )
    )
    ctx.register_tool(
        ToolDefinition(
            name="get_video_info",
            description=(
                "Get duration, resolution, codecs, and format for a video file."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the video file"},
                },
                "required": ["path"],
            },
            handler=get_video_info_handler(runtime),
            timeout_s=60.0,
            idempotent=True,
        )
    )
    ctx.register_tool(
        ToolDefinition(
            name="analyze_video",
            description=(
                "Understand video content for natural-language editing. Extracts "
                "sample frames and (when vision_provider is configured) describes "
                "what happens at each timestamp."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Path to the video file"},
                    "frame_count": {
                        "type": "integer",
                        "description": "Number of frames to sample (default 8)",
                    },
                },
                "required": ["input"],
            },
            handler=analyze_handler,
            timeout_s=300.0,
            idempotent=True,
        )
    )
    ctx.register_tool(
        ToolDefinition(
            name="plan_edits",
            description=(
                "Generate a structured JSON edit plan from a natural-language prompt. "
                "Returns steps with actions, timestamps, and params. Always call "
                "before executing edits."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Resolved local video path"},
                    "prompt": {
                        "type": "string",
                        "description": "User editing instructions",
                    },
                    "frame_count": {
                        "type": "integer",
                        "description": "Frames for scene analysis (default from config)",
                    },
                },
                "required": ["input", "prompt"],
            },
            handler=plan_edits_handler(runtime, analyze_handler),
            timeout_s=600.0,
            idempotent=True,
        )
    )
    ctx.register_tool(
        ToolDefinition(
            name="extract_frames",
            description=(
                "Extract JPEG frames at evenly spaced times or explicit timestamps."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Path to video"},
                    "count": {
                        "type": "integer",
                        "description": "Evenly spaced frame count when timestamps omitted",
                    },
                    "timestamps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Explicit times (seconds or HH:MM:SS)",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory for frame images",
                    },
                },
                "required": ["input"],
            },
            handler=extract_frames_handler(runtime),
            timeout_s=300.0,
            idempotent=True,
        )
    )
    ctx.register_tool(
        ToolDefinition(
            name="trim_clip",
            description=(
                "Trim a video to a time range. Times accept seconds (12.5) or "
                "MM:SS / HH:MM:SS. Omit end to keep until EOF."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Source video path"},
                    "start": {"type": "string", "description": "Start time (default 0)"},
                    "end": {"type": "string", "description": "End time"},
                    "output": {
                        "type": "string",
                        "description": "Output path (auto: {stem}_edited_{timestamp}.mp4)",
                    },
                },
                "required": ["input"],
            },
            handler=trim_clip_handler(runtime),
            timeout_s=600.0,
        )
    )
    ctx.register_tool(
        ToolDefinition(
            name="cut_segment",
            description=(
                "Remove a time range from a video (keeps everything except the segment)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Source video path"},
                    "start": {"type": "string", "description": "Start of segment to remove"},
                    "end": {"type": "string", "description": "End of segment to remove"},
                    "output": {"type": "string", "description": "Output path"},
                },
                "required": ["input", "start", "end"],
            },
            handler=cut_segment_handler(runtime),
            timeout_s=600.0,
        )
    )
    ctx.register_tool(
        ToolDefinition(
            name="concat_clips",
            description="Concatenate multiple video files in order (stream copy).",
            parameters={
                "type": "object",
                "properties": {
                    "inputs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered list of video paths",
                    },
                    "output": {"type": "string", "description": "Output video path"},
                },
                "required": ["inputs"],
            },
            handler=concat_clips_handler(runtime),
            timeout_s=600.0,
        )
    )
    ctx.register_tool(
        ToolDefinition(
            name="speed_change",
            description="Change playback speed of the full video (factor > 1 = faster).",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "factor": {"type": "number", "description": "Speed multiplier"},
                    "output": {"type": "string"},
                },
                "required": ["input", "factor"],
            },
            handler=speed_change_handler(runtime),
            timeout_s=600.0,
        )
    )
    ctx.register_tool(
        ToolDefinition(
            name="mute_audio",
            description="Remove audio track from a video.",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "output": {"type": "string"},
                },
                "required": ["input"],
            },
            handler=mute_audio_handler(runtime),
            timeout_s=600.0,
        )
    )
    ctx.register_tool(
        ToolDefinition(
            name="extract_audio",
            description="Extract audio from a video as MP3.",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "output": {"type": "string"},
                },
                "required": ["input"],
            },
            handler=extract_audio_handler(runtime),
            timeout_s=600.0,
        )
    )
    ctx.register_tool(
        ToolDefinition(
            name="add_text_overlay",
            description="Add text overlay to a video using ffmpeg drawtext.",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "text": {"type": "string"},
                    "start": {"type": "string", "description": "Show from (default 0)"},
                    "end": {"type": "string", "description": "Hide after (default EOF)"},
                    "position": {
                        "type": "string",
                        "enum": ["top", "center", "bottom"],
                    },
                    "font_size": {"type": "integer"},
                    "output": {"type": "string"},
                },
                "required": ["input", "text"],
            },
            handler=add_text_overlay_handler(runtime),
            timeout_s=600.0,
        )
    )
    ctx.register_tool(
        ToolDefinition(
            name="crop",
            description="Crop video to width x height at optional x,y offset.",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "output": {"type": "string"},
                },
                "required": ["input", "width", "height"],
            },
            handler=crop_handler(runtime),
            timeout_s=600.0,
        )
    )
    ctx.register_tool(
        ToolDefinition(
            name="resize",
            description="Resize video to width x height.",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                    "output": {"type": "string"},
                },
                "required": ["input", "width", "height"],
            },
            handler=resize_handler(runtime),
            timeout_s=600.0,
        )
    )
    ctx.register_tool(
        ToolDefinition(
            name="cleanup_temp",
            description="Remove intermediate/downloaded files from the video temp directory.",
            parameters={
                "type": "object",
                "properties": {
                    "stem": {
                        "type": "string",
                        "description": "Only remove files containing this stem",
                    },
                    "max_age_s": {
                        "type": "number",
                        "description": "Only remove files older than this many seconds",
                    },
                },
            },
            handler=cleanup_temp_handler(runtime),
            timeout_s=60.0,
            idempotent=True,
        )
    )


class VideoPlugin:
    name = "video"

    def install(self, ctx: Context, config: dict[str, Any]) -> None:
        try:
            runtime = VideoRuntime(config)
        except FileNotFoundError as exc:
            raise RuntimeError(str(exc)) from exc

        ctx.services["video"] = runtime
        _register_core_tools(ctx, runtime)

    def uninstall(self, ctx: Context) -> None:
        runtime = ctx.services.get("video")
        if isinstance(runtime, VideoRuntime) and runtime.cleanup_temp_on_uninstall:
            runtime.cleanup_temp_dir()
        ctx.services.pop("video", None)
