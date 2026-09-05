from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from liteness.clickhouse.projector import (
    ClickHouseProjector,
    eval_result_id,
    project_eval_result,
    redact_payload,
)
from liteness.eval.models import EvalCaseResult, EvalReport, EvaluationResult
from liteness.session import SessionEvent


def _event(event_type: str, payload: dict, **kwargs) -> SessionEvent:
    return SessionEvent(
        id=kwargs.get("id", "e1"),
        type=event_type,
        timestamp=datetime(2026, 9, 5, 12, 0, 0, tzinfo=timezone.utc),
        session_id=kwargs.get("session_id", "s1"),
        turn=kwargs.get("turn", 1),
        step=kwargs.get("step", 1),
        payload=payload,
    )


def test_redact_strips_content_keeps_original() -> None:
    original = {"content": "secret", "name": "weather"}
    payload = dict(original)
    redacted = redact_payload(payload)
    assert redacted["content"] == "[redacted]"
    assert redacted["name"] == "weather"
    assert payload["content"] == "secret"


def test_redact_unknown_key_stripped() -> None:
    redacted = redact_payload({"content": "x", "mystery": "leak", "call_id": "c1"})
    assert redacted["mystery"] == "[redacted]"
    assert redacted["call_id"] == "c1"


def test_redact_keeps_tool_name_stop_reason() -> None:
    projector = ClickHouseProjector(mode="redacted")
    row = projector.project_event(
        _event("tool/call", {"call_id": "c1", "name": "weather", "arguments": {"q": "Delhi"}})
    )
    assert row is not None
    assert row.tool_name == "weather"
    assert row.call_id == "c1"
    body = json.loads(row.payload)
    assert body["arguments"] == "[redacted]"
    assert body["name"] == "weather"


def test_full_mode_round_trips_payload() -> None:
    projector = ClickHouseProjector(mode="full")
    row = projector.project_event(
        _event("user/message", {"content": "hello"}, step=0)
    )
    assert row is not None
    assert json.loads(row.payload)["content"] == "hello"


def test_chunk_elision() -> None:
    projector = ClickHouseProjector(mode="full")
    first = projector.project_event(
        _event("assistant/chunk", {"content_delta": "Hel"}, id="c1")
    )
    second = projector.project_event(
        _event("assistant/chunk", {"content_delta": "lo"}, id="c2")
    )
    assert first is not None
    assert second is None


def test_severity_tool_error() -> None:
    row = ClickHouseProjector().project_event(
        _event("tool/result", {"call_id": "c1", "name": "x", "content": "boom", "is_error": True, "error_code": "TOOL_TIMEOUT"})
    )
    assert row is not None
    assert row.severity == "error"
    assert row.error_code == "TOOL_TIMEOUT"
    assert row.is_error == 1


def test_eval_reason_redacted() -> None:
    report = EvalReport(
        schema_version=1,
        suite="basic",
        timestamp="2026-09-05T00:00:00+00:00",
        summary={},
        evaluators_summary={},
        cases=[],
        failures=[],
        dataset="basic",
        git_commit="abc",
    )
    case = EvalCaseResult(
        id="weather-001",
        passed=False,
        session_path=str(Path("/tmp/sessions/weather-001.jsonl")),
        status="complete",
        results=[],
    )
    result = EvaluationResult(
        evaluator="stop_reason",
        evaluator_version="1.0.0",
        case_id="weather-001",
        passed=False,
        score=0.0,
        reason="expected stop_reason completed, got llm_error",
    )
    row = project_eval_result(
        report=report, case=case, result=result, session_id="s1", mode="redacted"
    )
    assert row.reason == "[redacted]"
    assert row.session_path == "weather-001.jsonl"
    assert row.result_id == eval_result_id(
        "basic", "weather-001", "stop_reason", "s1", report.timestamp
    )


def test_eval_reason_full_mode_kept() -> None:
    report = EvalReport(
        schema_version=1,
        suite="basic",
        timestamp="2026-09-05T00:00:00+00:00",
        summary={},
        evaluators_summary={},
        cases=[],
        failures=[],
    )
    case = EvalCaseResult(
        id="c1", passed=True, session_path="C:/x/y.jsonl", status="complete"
    )
    result = EvaluationResult(
        evaluator="tool_lifecycle",
        evaluator_version="1.0.0",
        case_id="c1",
        passed=True,
        score=1.0,
        reason="ok",
    )
    row = project_eval_result(
        report=report, case=case, result=result, session_id="s1", mode="full"
    )
    assert row.reason == "ok"
    assert row.session_path == "y.jsonl"
