"""Video plugin — natural-language video editing via ffmpeg tools and vision analysis."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from liteness.context import Context
from liteness.tools import ToolDefinition, ToolResult

_TIME_RE = re.compile(
    r"^(?:(\d+):)?(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?$|^(\d+(?:\.\d+)?)$"
)


def _invalid_args(call_id: str, name: str, message: str) -> ToolResult:
    return ToolResult(
        call_id=call_id,
        name=name,
        content=message,
        is_error=True,
        error_code="INVALID_ARGS",
    )


def _parse_time(value: str | float | int) -> float:
    """Parse seconds float or HH:MM:SS[.ms] / MM:SS[.ms] string to seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("time must be a number or HH:MM:SS string")

    text = value.strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)

    match = _TIME_RE.match(text)
    if not match:
        raise ValueError(f"invalid time format: {value!r}")

    if match.group(5) is not None:
        return float(match.group(5))

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    millis = match.group(4)
    frac = int(millis.ljust(3, "0")[:3]) / 1000.0 if millis else 0.0
    return hours * 3600 + minutes * 60 + seconds + frac


def _format_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    whole = int(seconds)
    frac_ms = int(round((seconds - whole) * 1000))
    if frac_ms == 1000:
        whole += 1
        frac_ms = 0
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}.{frac_ms:03d}".rstrip("0").rstrip(".")
    return f"{m}:{s:02d}.{frac_ms:03d}".rstrip("0").rstrip(".")


def _resolve_path(raw: str, *, workspace: Path | None) -> Path:
    path = Path(raw)
    if not path.is_absolute() and workspace is not None:
        path = workspace / path
    return path.resolve()


def _find_binary(name: str, configured: str | None) -> str:
    if configured:
        return configured
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"{name} not found on PATH; install ffmpeg or set ffmpeg_path in plugin config"
    )


class VideoRuntime:
    """Per-install ffmpeg paths, workspace, and optional vision settings."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.ffmpeg = _find_binary("ffmpeg", config.get("ffmpeg_path"))
        self.ffprobe = _find_binary("ffprobe", config.get("ffprobe_path"))
        workspace = config.get("workspace")
        self.workspace = Path(workspace).resolve() if workspace else None
        output_dir = config.get("output_dir", "./video-output")
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute() and self.workspace is not None:
            self.output_dir = self.workspace / self.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vision_provider = config.get("vision_provider")
        self.vision_model = config.get("vision_model", "gemini-2.0-flash")
        self.max_frame_count = int(config.get("max_frame_count", 12))

    def resolve(self, raw: str) -> Path:
        return _resolve_path(raw, workspace=self.workspace)

    def default_output(self, stem: str, suffix: str = ".mp4") -> Path:
        safe = re.sub(r"[^\w.-]+", "_", stem).strip("_") or "output"
        return self.output_dir / f"{safe}{suffix}"


def _run_command(
    call_id: str,
    name: str,
    args: list[str],
    *,
    timeout_s: float = 300.0,
) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"command timed out after {timeout_s}s"
    except OSError as exc:
        return False, str(exc)

    output = completed.stdout.strip()
    if completed.stderr.strip():
        output = f"{output}\n{completed.stderr.strip()}".strip()
    if completed.returncode != 0:
        return False, output or f"exit code {completed.returncode}"
    return True, output


def get_video_info_handler(
    runtime: VideoRuntime,
) -> Any:
    def handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
        path_arg = arguments.get("path")
        if not isinstance(path_arg, str) or not path_arg:
            return _invalid_args(call_id, "get_video_info", "path is required")

        path = runtime.resolve(path_arg)
        if not path.exists():
            return ToolResult(
                call_id=call_id,
                name="get_video_info",
                content=f"file not found: {path}",
                is_error=True,
                error_code="NOT_FOUND",
            )

        ok, output = _run_command(
            call_id,
            "get_video_info",
            [
                runtime.ffprobe,
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
        )
        if not ok:
            return ToolResult(
                call_id=call_id,
                name="get_video_info",
                content=output,
                is_error=True,
                error_code="PROBE_ERROR",
            )

        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return ToolResult(
                call_id=call_id,
                name="get_video_info",
                content=output,
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
            f"duration: {_format_time(duration)}",
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
            return _invalid_args(call_id, "trim_clip", "input is required")

        try:
            start_s = _parse_time(arguments.get("start", 0))
            end_raw = arguments.get("end")
            end_s = _parse_time(end_raw) if end_raw is not None else None
        except ValueError as exc:
            return _invalid_args(call_id, "trim_clip", str(exc))

        input_path = runtime.resolve(input_arg)
        if not input_path.exists():
            return ToolResult(
                call_id=call_id,
                name="trim_clip",
                content=f"input not found: {input_path}",
                is_error=True,
                error_code="NOT_FOUND",
            )

        output_arg = arguments.get("output")
        if isinstance(output_arg, str) and output_arg:
            output_path = runtime.resolve(output_arg)
        else:
            output_path = runtime.default_output(f"{input_path.stem}_trimmed")

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
                return _invalid_args(call_id, "trim_clip", "end must be greater than start")
            args.extend(["-t", str(end_s - start_s)])
        args.extend(["-c", "copy", str(output_path)])

        ok, output = _run_command(call_id, "trim_clip", args, timeout_s=600.0)
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
                f"start: {_format_time(start_s)}\n"
                f"end: {_format_time(end_s) if end_s is not None else 'EOF'}"
            ),
        )

    return handler


def concat_clips_handler(runtime: VideoRuntime) -> Any:
    def handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
        inputs = arguments.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            return _invalid_args(call_id, "concat_clips", "inputs must be a non-empty list")
        if not all(isinstance(p, str) and p for p in inputs):
            return _invalid_args(
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

        output_arg = arguments.get("output")
        if isinstance(output_arg, str) and output_arg:
            output_path = runtime.resolve(output_arg)
        else:
            output_path = runtime.default_output("concatenated")

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
            ok, output = _run_command(
                call_id,
                "concat_clips",
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
            return _invalid_args(call_id, "extract_frames", "input is required")

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
            return _invalid_args(call_id, "extract_frames", "count must be a positive integer")

        count = min(count, runtime.max_frame_count)
        timestamps = arguments.get("timestamps")
        times: list[float] = []

        if timestamps is not None:
            if not isinstance(timestamps, list) or not timestamps:
                return _invalid_args(
                    call_id,
                    "extract_frames",
                    "timestamps must be a non-empty list when provided",
                )
            try:
                times = [_parse_time(t) for t in timestamps]
            except ValueError as exc:
                return _invalid_args(call_id, "extract_frames", str(exc))
            times = times[: runtime.max_frame_count]
        else:
            info_ok, info_out = _run_command(
                call_id,
                "extract_frames",
                [
                    runtime.ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(input_path),
                ],
            )
            if not info_ok:
                return ToolResult(
                    call_id=call_id,
                    name="extract_frames",
                    content=info_out,
                    is_error=True,
                    error_code="PROBE_ERROR",
                )
            try:
                duration = float(info_out.strip())
            except ValueError:
                duration = 0.0
            if duration <= 0:
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
            frame_path = out_dir / f"frame_{idx:03d}_{_format_time(t).replace(':', '-')}.jpg"
            ok, output = _run_command(
                call_id,
                "extract_frames",
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
                    content=f"failed at t={_format_time(t)}: {output}",
                    is_error=True,
                    error_code="FFMPEG_ERROR",
                )
            lines.append(f"t={_format_time(t)} ({t:.3f}s) -> {frame_path}")

        return ToolResult(
            call_id=call_id,
            name="extract_frames",
            content="\n".join(lines),
        )

    return handler


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
        parts.append(types.Part(text=f"Timestamp {_format_time(t)} ({t:.3f}s):"))
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


def analyze_video_handler(runtime: VideoRuntime) -> Any:
    def handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
        input_arg = arguments.get("input")
        if not isinstance(input_arg, str) or not input_arg:
            return _invalid_args(call_id, "analyze_video", "input is required")

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
            return _invalid_args(
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


class VideoPlugin:
    name = "video"

    def install(self, ctx: Context, config: dict[str, Any]) -> None:
        try:
            runtime = VideoRuntime(config)
        except FileNotFoundError as exc:
            raise RuntimeError(str(exc)) from exc

        ctx.services["video"] = runtime

        ctx.register_tool(
            ToolDefinition(
                name="get_video_info",
                description=(
                    "Get duration, resolution, codecs, and format for a video file. "
                    "Call this first before editing."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the video file",
                        },
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
                    "what happens at each timestamp. Use before trim/concat when the "
                    "user gives semantic instructions like 'remove the intro'."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Path to the video file",
                        },
                        "frame_count": {
                            "type": "integer",
                            "description": "Number of frames to sample (default 8)",
                        },
                    },
                    "required": ["input"],
                },
                handler=analyze_video_handler(runtime),
                timeout_s=300.0,
                idempotent=True,
            )
        )
        ctx.register_tool(
            ToolDefinition(
                name="extract_frames",
                description=(
                    "Extract JPEG frames at evenly spaced times or explicit timestamps. "
                    "Useful to inspect specific moments before cutting."
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
                        "start": {
                            "type": "string",
                            "description": "Start time (default 0)",
                        },
                        "end": {
                            "type": "string",
                            "description": "End time (exclusive segment end)",
                        },
                        "output": {
                            "type": "string",
                            "description": "Output video path (auto-generated if omitted)",
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
                        "output": {
                            "type": "string",
                            "description": "Output video path",
                        },
                    },
                    "required": ["inputs"],
                },
                handler=concat_clips_handler(runtime),
                timeout_s=600.0,
            )
        )

    def uninstall(self, ctx: Context) -> None:
        ctx.services.pop("video", None)
