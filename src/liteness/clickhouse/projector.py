"""Project session events and eval results into ClickHouse row shapes."""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from liteness.eval.models import EvalCaseResult, EvalReport, EvaluationResult
from liteness.session import SessionEvent

REDACT_PAYLOAD_ALLOWLIST: frozenset[str] = frozenset(
    {
        "name",
        "call_id",
        "id",
        "is_error",
        "error_code",
        "code",
        "model",
        "stop_reason",
        "status",
        "reason",
        "turn",
        "step",
        "retry_count",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "duration_ms",
        "total_tokens",
        "total_cost_usd",
        "tool_calls",
    }
)

_ERROR_STOP_REASONS = frozenset({"llm_error", "agent_error", "retry_exhausted"})


def _format_clickhouse_datetime(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    millis = dt.microsecond // 1000
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{millis:03d}"


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (_redact_value(nested) if key in REDACT_PAYLOAD_ALLOWLIST else "[redacted]")
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


def redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _redact_value(copy.deepcopy(payload))


def _extract_dimensions(event: SessionEvent) -> dict[str, Any]:
    payload = event.payload
    tool_name = ""
    if event.type in ("tool/call", "tool/result"):
        tool_name = str(payload.get("name") or "")

    call_id = str(payload.get("call_id") or "")
    error_code = str(payload.get("error_code") or payload.get("code") or "")

    is_error = 0
    if event.type == "tool/result" and payload.get("is_error") is True:
        is_error = 1

    stop_reason = ""
    if event.type in ("turn/end", "policy/stop"):
        stop_reason = str(payload.get("stop_reason") or payload.get("reason") or "")

    model = str(payload.get("model") or "")
    return {
        "tool_name": tool_name,
        "call_id": call_id,
        "error_code": error_code,
        "is_error": is_error,
        "stop_reason": stop_reason,
        "model": model,
    }


def _severity_for_event(event: SessionEvent, *, is_error: int, stop_reason: str) -> str:
    if event.type == "error":
        return "error"
    if event.type == "tool/result" and is_error:
        return "error"
    if event.type == "turn/end" and stop_reason in _ERROR_STOP_REASONS:
        return "error"
    return "info"


def _payload_json(payload: dict[str, Any], mode: str) -> str:
    if mode == "full":
        exported = payload
    else:
        exported = redact_payload(payload)
    return json.dumps(exported, ensure_ascii=False)


@dataclass
class SessionEventRow:
    event_id: str
    channel: str
    event_type: str
    timestamp: datetime
    session_id: str
    turn: int
    step: int
    severity: str
    tool_name: str
    call_id: str
    error_code: str
    is_error: int
    stop_reason: str
    model: str
    payload: str

    def to_insert_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "channel": self.channel,
            "event_type": self.event_type,
            "timestamp": _format_clickhouse_datetime(self.timestamp),
            "session_id": self.session_id,
            "turn": self.turn,
            "step": self.step,
            "severity": self.severity,
            "tool_name": self.tool_name,
            "call_id": self.call_id,
            "error_code": self.error_code,
            "is_error": self.is_error,
            "stop_reason": self.stop_reason,
            "model": self.model,
            "payload": self.payload,
        }


@dataclass
class EvalResultRow:
    result_id: str
    timestamp: datetime
    suite: str
    dataset: str
    case_id: str
    session_id: str
    evaluator: str
    passed: int | None
    score: float
    reason: str
    status: str
    session_path: str
    preset: str
    provider: str
    model: str
    git_commit: str

    def to_insert_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "timestamp": _format_clickhouse_datetime(self.timestamp),
            "suite": self.suite,
            "dataset": self.dataset,
            "case_id": self.case_id,
            "session_id": self.session_id,
            "evaluator": self.evaluator,
            "passed": self.passed,
            "score": self.score,
            "reason": self.reason,
            "status": self.status,
            "session_path": self.session_path,
            "preset": self.preset,
            "provider": self.provider,
            "model": self.model,
            "git_commit": self.git_commit,
        }


class ClickHouseProjector:
    def __init__(self, mode: str = "redacted") -> None:
        self.mode = mode
        self._chunk_seen: set[tuple[str, int, int]] = set()

    def project_event(self, event: SessionEvent) -> SessionEventRow | None:
        if event.type == "assistant/chunk":
            key = (event.session_id, event.turn, event.step)
            if key in self._chunk_seen:
                return None
            self._chunk_seen.add(key)

        dims = _extract_dimensions(event)
        severity = _severity_for_event(
            event,
            is_error=dims["is_error"],
            stop_reason=dims["stop_reason"],
        )
        return SessionEventRow(
            event_id=event.id,
            channel="ledger",
            event_type=event.type,
            timestamp=event.timestamp,
            session_id=event.session_id,
            turn=event.turn,
            step=event.step,
            severity=severity,
            tool_name=dims["tool_name"],
            call_id=dims["call_id"],
            error_code=dims["error_code"],
            is_error=dims["is_error"],
            stop_reason=dims["stop_reason"],
            model=dims["model"],
            payload=_payload_json(event.payload, self.mode),
        )

    def project_ops(
        self,
        event_type: str,
        session_id: str,
        payload: dict[str, Any],
    ) -> SessionEventRow:
        severity = "error" if event_type == "ops/persistence_degraded" else "info"
        return SessionEventRow(
            event_id=uuid.uuid4().hex,
            channel="ops",
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            session_id=session_id,
            turn=0,
            step=0,
            severity=severity,
            tool_name="",
            call_id="",
            error_code="",
            is_error=0,
            stop_reason="",
            model="",
            payload=_payload_json(payload, self.mode),
        )


def eval_result_id(
    suite: str,
    case_id: str,
    evaluator: str,
    session_id: str,
    timestamp: str,
) -> str:
    digest_input = f"{suite}|{case_id}|{evaluator}|{session_id}|{timestamp}"
    return hashlib.sha256(digest_input.encode()).hexdigest()


def project_eval_result(
    *,
    report: EvalReport,
    case: EvalCaseResult,
    result: EvaluationResult,
    session_id: str,
    mode: str,
    preset: str = "",
    provider: str = "",
    model: str = "",
) -> EvalResultRow:
    passed: int | None
    if result.passed is None:
        passed = None
    else:
        passed = int(result.passed)

    reason = result.reason if mode == "full" else "[redacted]"
    return EvalResultRow(
        result_id=eval_result_id(
            report.suite,
            case.id,
            result.evaluator,
            session_id,
            report.timestamp,
        ),
        timestamp=datetime.fromisoformat(report.timestamp),
        suite=report.suite,
        dataset=report.dataset or "",
        case_id=case.id,
        session_id=session_id,
        evaluator=result.evaluator,
        passed=passed,
        score=result.score,
        reason=reason,
        status=case.status,
        session_path=Path(case.session_path).name,
        preset=preset,
        provider=provider,
        model=model,
        git_commit=report.git_commit or "",
    )
