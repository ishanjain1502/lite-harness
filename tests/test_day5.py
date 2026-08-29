"""Day 5 loop tests — stop reasons, cancellation, concurrency, timeouts."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Iterator

import pytest

from liteness.llm import LLMProvider, LlmChunk, LlmRequest, MockLLMProvider, MockStep, ToolCallDraft
from liteness.loop import AgentLoop, LoopConfig
from liteness.session import Session
from liteness.tools import (
    READ_FILE_TOOL,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
    default_registry,
)
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


def test_stop_reason_completed(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("hi", encoding="utf-8")

    session = Session()
    result = AgentLoop(llm=_readme_mock(str(readme)), tools=default_registry()).run_turn(
        session, "read"
    )

    assert result.stop_reason == StopReason.COMPLETED
    assert result.status == "completed"
    turn_end = [e for e in session.events if e.type == "turn/end"][-1]
    assert turn_end.payload["stop_reason"] == "completed"


def test_stop_reason_step_limit() -> None:
    llm = MockLLMProvider(
        steps=[
            MockStep(
                step=n,
                tool_calls=[
                    ToolCallDraft(
                        call_id=f"c{n}",
                        name="read_file",
                        arguments={"path": "/nonexistent"},
                    )
                ],
            )
            for n in range(1, 20)
        ]
    )
    session = Session()
    result = AgentLoop(
        llm=llm,
        tools=default_registry(),
        config=LoopConfig(max_steps_per_turn=3),
    ).run_turn(session, "loop forever")

    assert result.stop_reason == StopReason.STEP_LIMIT
    assert result.status == "stopped"
    assert result.final_output is None


def test_stop_reason_turn_limit(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("x", encoding="utf-8")
    loop = AgentLoop(
        llm=_readme_mock(str(readme)),
        tools=default_registry(),
        config=LoopConfig(max_turns_per_session=1),
    )
    session = Session()
    first = loop.run_turn(session, "one")
    second = loop.run_turn(session, "two")

    assert first.stop_reason == StopReason.COMPLETED
    assert second.stop_reason == StopReason.TURN_LIMIT
    assert second.steps_run == 0


def test_stop_reason_context_limit(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("x", encoding="utf-8")
    session = Session()
    result = AgentLoop(
        llm=_readme_mock(str(readme)),
        tools=default_registry(),
        config=LoopConfig(max_context_chars=5),
    ).run_turn(session, "this message alone exceeds the tiny context budget")

    assert result.stop_reason == StopReason.CONTEXT_LIMIT
    assert result.status == "stopped"


def test_stop_reason_agent_error() -> None:
    llm = MockLLMProvider(steps=[MockStep(step=1, content="")])
    session = Session()
    result = AgentLoop(llm=llm, tools=default_registry()).run_turn(session, "empty")

    assert result.stop_reason == StopReason.AGENT_ERROR
    assert result.status == "error"


class FailingLLM(LLMProvider):
    def __init__(self, *, retryable: bool = False, fail_times: int = 1) -> None:
        self.retryable = retryable
        self.fail_times = fail_times
        self.calls = 0

    def stream(self, request: LlmRequest) -> Iterator[LlmChunk]:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise LlmError("NETWORK", "connection reset", retryable=self.retryable)
        yield LlmChunk(content_delta="recovered", done=True)


def test_llm_error_stop_reason() -> None:
    session = Session()
    result = AgentLoop(
        llm=FailingLLM(retryable=False),
        tools=default_registry(),
    ).run_turn(session, "fail")

    assert result.stop_reason == StopReason.LLM_ERROR
    assert result.status == "error"
    assert any(e.type == "error" for e in session.events)


def test_llm_retry_then_success() -> None:
    session = Session()
    llm = FailingLLM(retryable=True, fail_times=2)
    result = AgentLoop(
        llm=llm,
        tools=default_registry(),
        config=LoopConfig(llm_max_retries=3, llm_retry_delay_s=0),
    ).run_turn(session, "retry")

    assert result.stop_reason == StopReason.COMPLETED
    assert llm.calls == 3


def test_cancellation_before_step() -> None:
    llm = MockLLMProvider(
        steps=[
            MockStep(
                step=1,
                tool_calls=[
                    ToolCallDraft(
                        call_id="slow",
                        name="slow_tool",
                        arguments={},
                    )
                ],
            )
        ]
    )

    def slow_handler(call_id: str, args: dict) -> ToolResult:
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
        loop.run_turn(session, "cancel me", cancel=cancel)

    thread = threading.Thread(target=run)
    thread.start()
    time.sleep(0.05)
    cancel.cancel()
    thread.join(timeout=5)

    result_events = [e for e in session.events if e.type == "turn/end"]
    assert result_events
    assert result_events[-1].payload["stop_reason"] == "user_cancelled"


def test_tool_timeout() -> None:
    def slow_handler(call_id: str, args: dict) -> ToolResult:
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


def test_concurrent_tool_calls(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("A", encoding="utf-8")
    b.write_text("B", encoding="utf-8")

    timings: list[float] = []

    def timed_read(call_id: str, args: dict) -> ToolResult:
        start = time.monotonic()
        time.sleep(0.3)
        timings.append(time.monotonic() - start)
        path = args["path"]
        return ToolResult(call_id=call_id, name="read_file", content=Path(path).read_text())

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="read_file",
            description="timed read",
            parameters=READ_FILE_TOOL.parameters,
            handler=timed_read,
        )
    )
    llm = MockLLMProvider(
        steps=[
            MockStep(
                step=1,
                tool_calls=[
                    ToolCallDraft(
                        call_id="a",
                        name="read_file",
                        arguments={"path": str(a)},
                    ),
                    ToolCallDraft(
                        call_id="b",
                        name="read_file",
                        arguments={"path": str(b)},
                    ),
                ],
            ),
            MockStep(step=2, content="both read"),
        ]
    )

    session = Session()
    start = time.monotonic()
    AgentLoop(llm=llm, tools=registry).run_turn(session, "parallel reads")
    elapsed = time.monotonic() - start

    assert len(timings) == 2
    assert elapsed < 0.55  # parallel ~0.3s, serial would be ~0.6s+
    calls = [e for e in session.events if e.type == "tool/call"]
    assert len(calls) == 2


def test_tool_exception_normalized() -> None:
    def boom(_call_id: str, _args: dict) -> ToolResult:
        raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="boom",
            description="explodes",
            parameters={"type": "object", "properties": {}},
            handler=boom,
        )
    )
    llm = MockLLMProvider(
        steps=[
            MockStep(
                step=1,
                tool_calls=[ToolCallDraft(call_id="x", name="boom", arguments={})],
            ),
            MockStep(step=2, content="handled"),
        ]
    )
    session = Session()
    AgentLoop(llm=llm, tools=registry).run_turn(session, "boom")

    results = [e for e in session.events if e.type == "tool/result"]
    assert results[0].payload["is_error"] is True
    assert results[0].payload["error_code"] == "TOOL_EXCEPTION"
