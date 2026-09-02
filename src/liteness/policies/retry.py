"""Retry policy — shared LLM and tool retry decisions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Literal

from liteness.errors import ErrorCode, normalize_tool_error_code
from liteness.tools import ToolDefinition

RetryDomain = Literal["llm", "tool"]

EventEmitter = Callable[[str, dict], None]


@dataclass
class DomainRetryConfig:
    max_attempts: int = 3
    backoff_s: float = 3.0


@dataclass
class RetryPolicyConfig:
    llm: DomainRetryConfig = field(default_factory=lambda: DomainRetryConfig(max_attempts=3, backoff_s=3.0))
    tool: DomainRetryConfig = field(default_factory=lambda: DomainRetryConfig(max_attempts=2, backoff_s=0.0))


_LLM_RETRYABLE = {
    ErrorCode.LLM_RATE_LIMITED,
    ErrorCode.LLM_TIMEOUT,
    ErrorCode.LLM_UNAVAILABLE,
}

_LLM_LEGACY_RETRYABLE = {
    "NETWORK": ErrorCode.LLM_UNAVAILABLE,
    "RATE_LIMIT": ErrorCode.LLM_RATE_LIMITED,
}

_TOOL_NEVER_RETRY = {
    ErrorCode.TOOL_INVALID_ARGS,
    ErrorCode.TOOL_NOT_FOUND,
    ErrorCode.TOOL_PERMISSION_DENIED,
    ErrorCode.TOOL_CANCELLED,
}


class RetryPolicy:
    def __init__(
        self,
        config: RetryPolicyConfig | None = None,
        *,
        emit_event: EventEmitter | None = None,
    ) -> None:
        self.config = config or RetryPolicyConfig()
        self.emit_event = emit_event

    def config_for(self, domain: RetryDomain) -> DomainRetryConfig:
        return self.config.llm if domain == "llm" else self.config.tool

    def should_retry_llm(self, code: str, *, attempt: int) -> bool:
        cfg = self.config.llm
        if attempt > cfg.max_attempts:
            return False
        try:
            error_code = ErrorCode(code)
        except ValueError:
            mapped = _LLM_LEGACY_RETRYABLE.get(code)
            if mapped is None:
                return False
            error_code = mapped
        return error_code in _LLM_RETRYABLE

    def should_retry_tool(
        self,
        error_code: str | None,
        *,
        tool: ToolDefinition | None,
        attempt: int,
    ) -> bool:
        cfg = self.config.tool
        if attempt > cfg.max_attempts:
            return False
        code = normalize_tool_error_code(error_code)
        if code in _TOOL_NEVER_RETRY:
            return False
        if code == ErrorCode.TOOL_TIMEOUT:
            return tool is not None and tool.idempotent
        if code in {ErrorCode.TOOL_EXCEPTION, ErrorCode.UNKNOWN_ERROR}:
            return tool is not None and tool.idempotent
        return False

    def schedule_retry(
        self,
        domain: RetryDomain,
        *,
        attempt: int,
        reason_code: str,
        delay_s: float | None = None,
        call_id: str | None = None,
    ) -> None:
        cfg = self.config_for(domain)
        wait = cfg.backoff_s if delay_s is None else delay_s
        event_type = "llm/retry" if domain == "llm" else "tool/retry"
        payload: dict = {
            "attempt": attempt,
            "reason_code": reason_code,
            "delay_s": wait,
        }
        if call_id is not None:
            payload["call_id"] = call_id
        if self.emit_event is not None:
            self.emit_event(event_type, payload)
        if wait > 0:
            time.sleep(wait)
