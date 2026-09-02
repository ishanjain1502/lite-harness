"""Canonical harness error taxonomy (domain + code)."""

from __future__ import annotations

from enum import Enum


class ErrorDomain(str, Enum):
    LLM = "LLM"
    TOOL = "TOOL"
    PLUGIN = "PLUGIN"
    CONTROL = "CONTROL"
    POLICY = "POLICY"


class ErrorCode(str, Enum):
    LLM_RATE_LIMITED = "LLM_RATE_LIMITED"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    LLM_INVALID_RESPONSE = "LLM_INVALID_RESPONSE"
    LLM_ERROR = "LLM_ERROR"

    TOOL_INVALID_ARGS = "TOOL_INVALID_ARGS"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    TOOL_PERMISSION_DENIED = "TOOL_PERMISSION_DENIED"
    TOOL_EXCEPTION = "TOOL_EXCEPTION"
    TOOL_CANCELLED = "TOOL_CANCELLED"

    PLUGIN_UNAVAILABLE = "PLUGIN_UNAVAILABLE"
    PLUGIN_EXCEPTION = "PLUGIN_EXCEPTION"

    CONTROL_CANCELLED = "CONTROL_CANCELLED"
    CONTROL_BUDGET_EXCEEDED = "CONTROL_BUDGET_EXCEEDED"
    CONTROL_STEP_LIMIT = "CONTROL_STEP_LIMIT"
    CONTROL_TIMEOUT = "CONTROL_TIMEOUT"

    POLICY_DENIED = "POLICY_DENIED"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


def normalize_tool_error_code(raw: str | None) -> ErrorCode:
    """Map tool result error_code strings to canonical ErrorCode."""
    if raw is None:
        return ErrorCode.UNKNOWN_ERROR
    mapping = {
        "UNKNOWN_TOOL": ErrorCode.TOOL_NOT_FOUND,
        "TOOL_TIMEOUT": ErrorCode.TOOL_TIMEOUT,
        "TOOL_EXCEPTION": ErrorCode.TOOL_EXCEPTION,
        "CANCELLED": ErrorCode.TOOL_CANCELLED,
        "INVALID_ARGS": ErrorCode.TOOL_INVALID_ARGS,
    }
    try:
        return ErrorCode(raw)
    except ValueError:
        return mapping.get(raw, ErrorCode.UNKNOWN_ERROR)
