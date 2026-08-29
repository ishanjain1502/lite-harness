"""Agent decision seam: validate tool calls and map LLM output to actions."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from liteness.llm import ToolCallDraft, ToolSchema
from liteness.session import Message


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AgentContext:
    session_id: str
    turn: int
    step: int
    user_goal: str
    messages: list[Message]
    tool_schemas: list[ToolSchema]
    allowed_tools: list[str]
    llm_content: str
    tool_call_drafts: list[ToolCallDraft]
    max_steps: int
    steps_remaining: int


@dataclass
class ToolCallsAction:
    kind: Literal["tool_calls"] = "tool_calls"
    calls: list[ToolCall] | None = None
    rationale: str | None = None


@dataclass
class InjectContextAction:
    kind: Literal["inject_context"] = "inject_context"
    content: str = ""
    rationale: str | None = None


@dataclass
class FinalAnswerStop:
    kind: Literal["final_answer"] = "final_answer"
    content: str = ""
    rationale: str | None = None


@dataclass
class ErrorStop:
    kind: Literal["error"] = "error"
    code: str = ""
    message: str = ""


@dataclass
class PolicyHaltStop:
    kind: Literal["policy_halt"] = "policy_halt"
    reason: str = ""


AgentAction = ToolCallsAction | InjectContextAction
AgentStop = FinalAnswerStop | ErrorStop | PolicyHaltStop
AgentDecision = AgentAction | AgentStop


def _validate_type(value: Any, schema: dict[str, Any]) -> str | None:
    expected = schema.get("type")
    if expected == "string" and not isinstance(value, str):
        return f"expected string, got {type(value).__name__}"
    if expected == "integer" and not isinstance(value, int):
        return f"expected integer, got {type(value).__name__}"
    if expected == "number" and not isinstance(value, (int, float)):
        return f"expected number, got {type(value).__name__}"
    if expected == "boolean" and not isinstance(value, bool):
        return f"expected boolean, got {type(value).__name__}"
    return None


def validate_tool_arguments(
    name: str,
    arguments: dict[str, Any],
    schemas: list[ToolSchema],
) -> list[str]:
    schema_by_name = {s.name: s for s in schemas}
    tool_schema = schema_by_name.get(name)
    if tool_schema is None:
        return [f"unknown tool: {name}"]

    errors: list[str] = []
    params = tool_schema.parameters
    required = params.get("required", [])
    properties = params.get("properties", {})

    for key in required:
        if key not in arguments:
            errors.append(f"missing required argument: {key}")

    for key, value in arguments.items():
        prop = properties.get(key)
        if prop is None:
            errors.append(f"unknown argument: {key}")
            continue
        type_err = _validate_type(value, prop)
        if type_err:
            errors.append(f"{key}: {type_err}")

    return errors


class Agent:
    """Default agent: prefer tool calls from LLM; validate args; stop on final text."""

    def decide(self, ctx: AgentContext) -> AgentDecision:
        if ctx.tool_call_drafts:
            return self._decide_tool_calls(ctx)

        content = ctx.llm_content.strip()
        if content:
            return FinalAnswerStop(content=content, rationale="model returned text")

        return ErrorStop(code="EMPTY_RESPONSE", message="model returned no text or tools")

    def _decide_tool_calls(self, ctx: AgentContext) -> AgentDecision:
        validated: list[ToolCall] = []
        errors: list[str] = []

        for draft in ctx.tool_call_drafts:
            if draft.name not in ctx.allowed_tools:
                errors.append(f"tool not allowed: {draft.name}")
                continue

            arg_errors = validate_tool_arguments(
                draft.name, draft.arguments, ctx.tool_schemas
            )
            if arg_errors:
                errors.extend(arg_errors)
                continue

            validated.append(
                ToolCall(
                    call_id=draft.call_id or _new_call_id(),
                    name=draft.name,
                    arguments=draft.arguments,
                )
            )

        if validated:
            return ToolCallsAction(
                calls=validated,
                rationale=json.dumps({"validated_calls": len(validated)}),
            )

        if errors:
            return InjectContextAction(
                content="Tool validation failed:\n- " + "\n- ".join(errors),
                rationale="validation_errors",
            )

        return ErrorStop(code="NO_VALID_TOOLS", message="no valid tool calls")
