"""Structured edit planning from natural-language prompts."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from liteness.plugins.video_utils import (
    VALID_EDIT_ACTIONS,
    VideoRuntime,
    edited_output_path,
    format_time,
    invalid_args,
    parse_time,
    probe_duration,
)
from liteness.tools import ToolResult


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model response")
    return json.loads(text[start : end + 1])


def validate_edit_plan(plan: dict[str, Any], *, duration_s: float | None) -> list[str]:
    errors: list[str] = []
    steps = plan.get("steps")
    if not isinstance(steps, list):
        errors.append("steps must be a list")
        return errors

    for idx, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            errors.append(f"step {idx} must be an object")
            continue

        action = step.get("action")
        if action not in VALID_EDIT_ACTIONS:
            errors.append(f"step {idx} has invalid action: {action!r}")

        for field in ("start", "end"):
            if field in step and step[field] is not None:
                try:
                    parsed = parse_time(step[field])
                except (TypeError, ValueError):
                    errors.append(f"step {idx} has invalid {field}: {step[field]!r}")
                    continue
                if duration_s is not None and parsed > duration_s:
                    errors.append(
                        f"step {idx} {field} ({format_time(parsed)}) exceeds "
                        f"video duration ({format_time(duration_s)})"
                    )

        if action in {"trim_clip", "cut_segment"}:
            if "start" not in step or "end" not in step:
                errors.append(f"step {idx} ({action}) requires start and end")
            else:
                try:
                    start_s = parse_time(step["start"])
                    end_s = parse_time(step["end"])
                    if end_s <= start_s:
                        errors.append(f"step {idx} end must be greater than start")
                except (TypeError, ValueError):
                    pass

        if action == "speed_change":
            factor = (step.get("params") or {}).get("factor")
            if not isinstance(factor, (int, float)) or factor <= 0:
                errors.append(f"step {idx} speed_change requires params.factor > 0")

        if action == "add_text_overlay":
            text = (step.get("params") or {}).get("text")
            if not isinstance(text, str) or not text:
                errors.append(f"step {idx} add_text_overlay requires params.text")

        if action in {"crop", "resize"}:
            params = step.get("params") or {}
            if not isinstance(params.get("width"), int) or not isinstance(params.get("height"), int):
                errors.append(f"step {idx} {action} requires params.width and params.height")

    return errors


def _metadata_fallback_plan(
    video_path: Path,
    prompt: str,
    duration_s: float | None,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    lower = prompt.lower()
    if duration_s and any(word in lower for word in ("intro", "opening", "first")):
        intro_end = min(10.0, duration_s * 0.1)
        steps.append(
            {
                "step": 1,
                "action": "cut_segment",
                "description": "Remove intro (metadata fallback — verify timestamps)",
                "start": "0:00",
                "end": format_time(intro_end),
                "params": {},
            }
        )
    return {
        "video_path": str(video_path),
        "duration_s": duration_s,
        "steps": steps,
        "output_path": str(edited_output_path(video_path)),
        "warning": (
            "Vision unavailable — plan generated from prompt keywords only. "
            "Review timestamps carefully before executing."
        ),
    }


def _plan_with_gemini(
    *,
    model: str,
    video_path: Path,
    prompt: str,
    analysis: str,
    duration_s: float | None,
) -> dict[str, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "plan_edits with vision requires google-genai: pip install 'lite-ness[google]'"
        ) from exc

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set")

    duration_hint = f"{duration_s:.3f}s" if duration_s is not None else "unknown"
    system = (
        "You are a video editing planner. Given video analysis and a user prompt, "
        "return ONLY a JSON object with keys: video_path, duration_s, steps, output_path.\n"
        "Each step must have: step (int), action, description, start, end, params.\n"
        f"Valid actions: {', '.join(sorted(VALID_EDIT_ACTIONS))}.\n"
        "Use MM:SS or seconds for start/end. params holds action-specific fields.\n"
        f"video_path must be {video_path}. output_path must be next to the input with "
        "_edited_TIMESTAMP suffix.\n"
        "Order steps for sequential execution."
    )
    user_text = (
        f"Video path: {video_path}\n"
        f"Duration: {duration_hint}\n"
        f"User prompt: {prompt}\n\n"
        f"Video analysis:\n{analysis}"
    )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text=f"{system}\n\n{user_text}")],
            )
        ],
    )
    text = getattr(response, "text", None) or ""
    plan = _extract_json_object(text)
    plan.setdefault("video_path", str(video_path))
    if duration_s is not None:
        plan.setdefault("duration_s", duration_s)
    plan.setdefault("output_path", str(edited_output_path(video_path)))
    return plan


def plan_edits_handler(
    runtime: VideoRuntime,
    analyze_video_handler: Any,
) -> Any:
    def handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
        input_arg = arguments.get("input")
        prompt = arguments.get("prompt")
        if not isinstance(input_arg, str) or not input_arg:
            return invalid_args(call_id, "plan_edits", "input is required")
        if not isinstance(prompt, str) or not prompt.strip():
            return invalid_args(call_id, "plan_edits", "prompt is required")

        input_path = runtime.resolve(input_arg)
        if not input_path.exists():
            return ToolResult(
                call_id=call_id,
                name="plan_edits",
                content=f"input not found: {input_path}",
                is_error=True,
                error_code="NOT_FOUND",
            )

        frame_count = arguments.get("frame_count", runtime.max_frame_count)
        if not isinstance(frame_count, int) or frame_count < 1:
            return invalid_args(call_id, "plan_edits", "frame_count must be a positive integer")
        frame_count = min(frame_count, runtime.max_frame_count)

        analysis_result = analyze_video_handler(
            call_id,
            {"input": str(input_path), "frame_count": frame_count},
        )
        if analysis_result.is_error:
            return ToolResult(
                call_id=call_id,
                name="plan_edits",
                content=analysis_result.content,
                is_error=True,
                error_code=analysis_result.error_code,
            )

        duration_s = probe_duration(runtime, input_path)
        warning: str | None = None

        if runtime.vision_provider == "google":
            try:
                plan = _plan_with_gemini(
                    model=runtime.vision_model,
                    video_path=input_path,
                    prompt=prompt.strip(),
                    analysis=analysis_result.content,
                    duration_s=duration_s,
                )
            except RuntimeError as exc:
                plan = _metadata_fallback_plan(input_path, prompt.strip(), duration_s)
                warning = str(exc)
        else:
            plan = _metadata_fallback_plan(input_path, prompt.strip(), duration_s)
            warning = (
                "vision_provider not configured — using metadata-only fallback plan"
            )

        errors = validate_edit_plan(plan, duration_s=duration_s)
        if errors:
            return ToolResult(
                call_id=call_id,
                name="plan_edits",
                content="invalid edit plan:\n" + "\n".join(f"- {e}" for e in errors),
                is_error=True,
                error_code="INVALID_PLAN",
            )

        body = json.dumps(plan, indent=2)
        if warning:
            body = f"warning: {warning}\n\n{body}"
        return ToolResult(call_id=call_id, name="plan_edits", content=body)

    return handler
