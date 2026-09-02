"""Day 8 tests — telemetry, budgets, retries, and control-plane events."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Iterator

import pytest

from liteness.context import Context
from liteness.events import EventBus
from liteness.llm import LLMProvider, LlmChunk, LlmRequest, MockLLMProvider, MockStep, ToolCallDraft
from liteness.loop import AgentLoop, LoopConfig
from liteness.plugins.telemetry import TelemetryPlugin
from liteness.policies.budget import BudgetConfig, BudgetManager
from liteness.policies.retry import RetryPolicy
from liteness.session import Session
from liteness.telemetry.projector import format_trace_report, project_events
from liteness.tools import ToolDefinition, ToolRegistry, ToolResult
from liteness.testing import registry_with_plugins
from liteness.types import CancelToken, LlmError, StopReason


def _readme_mock(readme: str) -> MockLLMProvider:
    return MockLLMProvider(
        steps=[
            MockStep(
                step=1,
                tool_calls=[
                    ToolCallDraft(
                        call_id="c1",
                        name="read_file",
                        arguments={"path": readme},
                    )
                ],
            ),
            MockStep(step=2, content="done"),
        ]
    )


def test_demo_success_trace_and_metrics(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("hello", encoding="utf-8")

    session = Session()
    loop = AgentLoop(llm=_readme_mock(str(readme)), tools=registry_with_plugins())
    result = loop.run_turn(session, "read")

    assert result.stop_reason == StopReason.COMPLETED
    report = project_events(session.events)
    assert report.metrics.llm_requests >= 1
    assert report.metrics.tool_calls == 1
    assert report.metrics.tool_success_rate == 1.0
    assert any(s.kind == "turn" for s in report.trace.spans)
    assert any(s.kind == "tool" for s in report.trace.spans)
    assert any(e.type == "llm/response" for e in session.events)


class FailingLLM(LLMProvider):
    def __init__(self, *, fail_times: int = 1) -> None:
        self.fail_times = fail_times
        self.calls = 0

    def stream(self, request: LlmRequest) -> Iterator[LlmChunk]:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise LlmError("NETWORK", "connection reset", retryable=True)
        yield LlmChunk(content_delta="recovered", done=True)


def test_demo_llm_retry_events_and_spans() -> None:
    session = Session()
    llm = FailingLLM(fail_times=2)
    AgentLoop(
        llm=llm,
        tools=registry_with_plugins(),
        config=LoopConfig(llm_max_retries=3, llm_retry_delay_s=0),
    ).run_turn(session, "retry")

    retries = [e for e in session.events if e.type == "llm/retry"]
    assert len(retries) == 2
    report = project_events(session.events)
    assert report.metrics.llm_retries == 2
    assert report.retry_count == 2
    assert any(s.kind == "retry-attempt" for s in report.trace.spans)


def test_demo_tool_timeout_inv18_note() -> None:
    def slow_handler(call_id: str, _args: dict) -> ToolResult:
        time.sleep(5)
        return ToolResult(call_id=call_id, name="slow_tool", content="done")

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="slow_tool",
            description="slow",
            parameters={"type": "object", "properties": {}},
            handler=slow_handler,
        )
    )
    llm = MockLLMProvider(
        steps=[
            MockStep(
                step=1,
                tool_calls=[ToolCallDraft(call_id="t1", name="slow_tool", arguments={})],
            ),
            MockStep(step=2, content="after timeout"),
        ]
    )
    session = Session()
    result = AgentLoop(
        llm=llm,
        tools=registry,
        config=LoopConfig(default_tool_timeout_s=0.2, llm_retry_delay_s=0),
    ).run_turn(session, "timeout")

    tool_results = [e for e in session.events if e.type == "tool/result"]
    assert tool_results[0].payload["error_code"] == "TOOL_TIMEOUT"
    assert result.stop_reason == StopReason.COMPLETED
    report = project_events(session.events)
    tool_span = next(s for s in report.trace.spans if s.kind == "tool")
    assert tool_span.status == "timeout"
    rendered = format_trace_report(report)
    assert "INV-18" in rendered


def test_demo_user_cancel_emits_event() -> None:
    llm = MockLLMProvider(
        steps=[
            MockStep(
                step=1,
                tool_calls=[ToolCallDraft(call_id="slow", name="slow_tool", arguments={})],
            )
        ]
    )

    def slow_handler(call_id: str, _args: dict) -> ToolResult:
        time.sleep(2)
        return ToolResult(call_id=call_id, name="slow_tool", content="late")

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="slow_tool",
            description="slow",
            parameters={"type": "object", "properties": {}},
            handler=slow_handler,
        )
    )

    cancel = CancelToken()
    session = Session()
    loop = AgentLoop(llm=llm, tools=registry)

    def run() -> None:
        loop.run_turn(session, "cancel", cancel=cancel)

    thread = threading.Thread(target=run)
    thread.start()
    time.sleep(0.05)
    cancel.cancel()
    thread.join(timeout=5)

    assert any(e.type == "cancel/requested" for e in session.events)
    turn_end = [e for e in session.events if e.type == "turn/end"][-1]
    assert turn_end.payload["stop_reason"] == "user_cancelled"


def test_demo_budget_exhausted() -> None:
    session = Session()
    budget = BudgetManager(
        config=BudgetConfig(max_tokens_session=10),
    )
    llm = MockLLMProvider(steps=[MockStep(step=1, content="x" * 200)])
    result = AgentLoop(
        llm=llm,
        tools=registry_with_plugins(),
        budget_manager=budget,
    ).run_turn(session, "budget")

    assert result.stop_reason == StopReason.BUDGET_EXCEEDED
    assert any(e.type == "budget/exhausted" for e in session.events)


def test_tool_retry_idempotent_only() -> None:
    attempts = {"count": 0}

    def flaky(call_id: str, _args: dict) -> ToolResult:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return ToolResult(
                call_id=call_id,
                name="flaky",
                content="timeout",
                is_error=True,
                error_code="TOOL_TIMEOUT",
            )
        return ToolResult(call_id=call_id, name="flaky", content="ok")

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="flaky",
            description="flaky",
            parameters={"type": "object", "properties": {}},
            handler=flaky,
            idempotent=True,
        )
    )
    llm = MockLLMProvider(
        steps=[
            MockStep(
                step=1,
                tool_calls=[ToolCallDraft(call_id="f1", name="flaky", arguments={})],
            ),
            MockStep(step=2, content="done"),
        ]
    )
    session = Session()
    AgentLoop(
        llm=llm,
        tools=registry,
        config=LoopConfig(tool_max_retries=2, tool_retry_delay_s=0, llm_retry_delay_s=0),
    ).run_turn(session, "retry tool")

    assert any(e.type == "tool/retry" for e in session.events)
    results = [e for e in session.events if e.type == "tool/result"]
    assert results[-1].payload["content"] == "ok"


def test_telemetry_plugin_live_projection() -> None:
    ctx = Context()
    plugin = TelemetryPlugin()
    plugin.install(ctx, {})

    session = Session()
    event = session.append("turn/start", {"turn": 1}, turn=1, step=0)
    ctx.emit("session/event", event.type, event.payload, session_event=event)

    report = plugin.report(session.session_id)
    assert any(s.kind == "turn" for s in report.trace.spans)


def test_telemetry_handler_failure_does_not_break_emitter() -> None:
    bus = EventBus()

    def boom(*_args, **_kwargs) -> None:
        raise RuntimeError("metrics crashed")

    bus.subscribe("session/event", boom)
    bus.emit("session/event", "tool/call", {"name": "x"})
    # no exception propagated


def test_offline_report_matches_live_telemetry(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("data", encoding="utf-8")

    session = Session()
    ctx = Context()
    plugin = TelemetryPlugin()
    plugin.install(ctx, {})

    loop = AgentLoop(
        llm=_readme_mock(str(readme)),
        tools=registry_with_plugins(),
        event_sink=lambda event: ctx.emit(
            "session/event", event.type, event.payload, session_event=event
        ),
    )
    loop.run_turn(session, "read")

    live = plugin.report(session.session_id)
    offline = project_events(session.events)
    assert live.metrics.llm_requests == offline.metrics.llm_requests
    assert live.metrics.tool_calls == offline.metrics.tool_calls


def test_report_command_roundtrip(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("x", encoding="utf-8")
    log = tmp_path / "session.jsonl"

    session = Session.open(log)
    AgentLoop(llm=_readme_mock(str(readme)), tools=registry_with_plugins()).run_turn(
        session, "read"
    )

    loaded = Session.load_from_jsonl(log)
    report = project_events(loaded.events)
    assert report.metrics.tool_calls == 1
    assert format_trace_report(report)
