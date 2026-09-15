"""Local Whisper transcription and SRT generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from liteness.plugins.video_utils import (
    VideoRuntime,
    invalid_args,
    probe_video_info,
    run_command,
)
from liteness.tools import ToolResult


def seconds_to_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(path: Path, segments: list[dict[str, Any]]) -> None:
    blocks: list[str] = []
    for segment in segments:
        index = segment["index"]
        start = seconds_to_srt_timestamp(float(segment["start"]))
        end = seconds_to_srt_timestamp(float(segment["end"]))
        text = str(segment["text"]).strip()
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
    path.write_text("\n".join(blocks) + "\n", encoding="utf-8")


def _has_audio_stream(probe_data: dict[str, Any]) -> bool:
    return any(
        stream.get("codec_type") == "audio"
        for stream in probe_data.get("streams", [])
        if isinstance(stream, dict)
    )


def _load_whisper_model(runtime: VideoRuntime) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed — run: pip install -e '.[video-whisper]'"
        ) from exc

    return WhisperModel(
        runtime.whisper_model,
        device=runtime.whisper_device,
        compute_type=runtime.whisper_compute_type,
    )


def _transcribe_segments(
    runtime: VideoRuntime,
    wav_path: Path,
    *,
    language: str | None,
) -> tuple[list[dict[str, Any]], str, float | None]:
    model = _load_whisper_model(runtime)
    kwargs: dict[str, Any] = {}
    if language:
        kwargs["language"] = language
    raw_segments, info = model.transcribe(str(wav_path), **kwargs)

    segments: list[dict[str, Any]] = []
    for idx, segment in enumerate(_iter_segments(raw_segments), start=1):
        segments.append(
            {
                "index": idx,
                "start": float(segment.start),
                "end": float(segment.end),
                "text": str(segment.text).strip(),
            }
        )

    detected_language = str(getattr(info, "language", "") or language or "unknown")
    duration = getattr(info, "duration", None)
    duration_s = float(duration) if duration is not None else None
    return segments, detected_language, duration_s


def _iter_segments(raw_segments: Any) -> Iterator[Any]:
    if hasattr(raw_segments, "__iter__"):
        yield from raw_segments
        return
    raise RuntimeError("whisper transcription returned no segments")


def transcribe_video_handler(runtime: VideoRuntime) -> Any:
    def handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
        input_arg = arguments.get("input")
        if not isinstance(input_arg, str) or not input_arg:
            return invalid_args(call_id, "transcribe_video", "input is required")

        input_path = runtime.resolve(input_arg)
        if not input_path.exists():
            return ToolResult(
                call_id=call_id,
                name="transcribe_video",
                content=f"input not found: {input_path}",
                is_error=True,
                error_code="NOT_FOUND",
            )

        probe_data = probe_video_info(runtime, input_path)
        if probe_data is None:
            return ToolResult(
                call_id=call_id,
                name="transcribe_video",
                content="failed to probe video",
                is_error=True,
                error_code="PROBE_ERROR",
            )
        if not _has_audio_stream(probe_data):
            return ToolResult(
                call_id=call_id,
                name="transcribe_video",
                content="video has no audio track",
                is_error=True,
                error_code="NO_AUDIO",
            )

        output_srt_arg = arguments.get("output_srt")
        if isinstance(output_srt_arg, str) and output_srt_arg:
            srt_path = runtime.resolve(output_srt_arg)
        else:
            srt_path = runtime.temp_dir / f"{input_path.stem}.srt"
        srt_path.parent.mkdir(parents=True, exist_ok=True)

        wav_path = runtime.temp_dir / f"{input_path.stem}_whisper.wav"
        ok, output = run_command(
            runtime,
            [
                runtime.ffmpeg,
                "-y",
                "-i",
                str(input_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                str(wav_path),
            ],
            timeout_s=600.0,
        )
        if not ok:
            return ToolResult(
                call_id=call_id,
                name="transcribe_video",
                content=output,
                is_error=True,
                error_code="FFMPEG_ERROR",
            )

        language_arg = arguments.get("language")
        language: str | None
        if isinstance(language_arg, str) and language_arg.strip():
            language = language_arg.strip()
        else:
            language = runtime.whisper_language

        try:
            segments, detected_language, duration_s = _transcribe_segments(
                runtime,
                wav_path,
                language=language,
            )
        except RuntimeError as exc:
            return ToolResult(
                call_id=call_id,
                name="transcribe_video",
                content=str(exc),
                is_error=True,
                error_code="WHISPER_ERROR",
            )

        write_srt(srt_path, segments)

        if duration_s is None:
            fmt = probe_data.get("format", {})
            try:
                duration_s = float(fmt.get("duration") or 0) or None
            except (TypeError, ValueError):
                duration_s = None

        payload = {
            "video_path": str(input_path),
            "srt_path": str(srt_path),
            "language": detected_language,
            "duration_s": duration_s,
            "segment_count": len(segments),
            "segments": segments,
        }
        return ToolResult(
            call_id=call_id,
            name="transcribe_video",
            content=json.dumps(payload, indent=2),
        )

    return handler
