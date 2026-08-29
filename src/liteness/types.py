"""Shared harness types: stop reasons, cancellation, errors."""

from __future__ import annotations

from enum import Enum
from typing import Literal


class StopReason(str, Enum):
    COMPLETED = "completed"
    USER_CANCELLED = "user_cancelled"
    STEP_LIMIT = "step_limit"
    TURN_LIMIT = "turn_limit"
    CONTEXT_LIMIT = "context_limit"
    LLM_ERROR = "llm_error"
    AGENT_ERROR = "agent_error"
    POLICY_DENIED = "policy_denied"


def stop_reason_to_status(
    reason: StopReason,
) -> Literal["completed", "stopped", "error"]:
    if reason == StopReason.COMPLETED:
        return "completed"
    if reason in (StopReason.LLM_ERROR, StopReason.AGENT_ERROR):
        return "error"
    return "stopped"


class CancelledError(Exception):
    """Raised when execution is cancelled via CancelToken."""


class CancelToken:
    """Cooperative cancellation signal passed into run_turn."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def check(self) -> None:
        if self._cancelled:
            raise CancelledError()


class LlmError(Exception):
    """Structured LLM failure for retry classification."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
