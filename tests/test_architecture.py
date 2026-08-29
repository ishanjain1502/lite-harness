"""Day 4 architecture tests — verify contracts, not just happy path."""

from __future__ import annotations

from pathlib import Path

import pytest

from liteness.agent import Agent
from liteness.llm import MockLLMProvider, MockStep, ToolCallDraft
from liteness.loop import AgentLoop, LoopConfig
from liteness.session import Session
from liteness.tools import (
    READ_FILE_TOOL,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
    default_registry,
)


def _readme_mock(readme: str = "README.md") -> MockLLMProvider:
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
            MockStep(step=2, content="The repository does harness things."),
        ]
    )


def test_full_turn_produces_expected_events(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# Demo\n\nA tiny harness.", encoding="utf-8")

    session = Session()
    loop = AgentLoop(llm=_readme_mock(str(readme)), tools=default_registry())
    result = loop.run_turn(session, "Read README.md and summarize.")

    assert result.status == "completed"
    assert result.final_output == "The repository does harness things."
    event_types = [e.type for e in session.events]
    assert "turn/start" in event_types
    assert "step/start" in event_types
    assert "assistant/chunk" in event_types
    assert "assistant/message" in event_types
    assert "tool/call" in event_types
    assert "tool/result" in event_types
    assert "step/end" in event_types
    assert "turn/end" in event_types


def test_can_replace_llm_without_touching_loop(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("hello", encoding="utf-8")

    loop_a = AgentLoop(llm=_readme_mock(str(readme)), tools=default_registry())
    loop_b = AgentLoop(
        llm=MockLLMProvider(steps=[MockStep(step=1, content="direct answer")]),
        tools=default_registry(),
    )

    session_a = Session()
    session_b = Session()
    loop_a.run_turn(session_a, "task a")
    loop_b.run_turn(session_b, "task b")

    assert any(e.type == "tool/call" for e in session_a.events)
    assert not any(e.type == "tool/call" for e in session_b.events)


def test_can_replace_tool_without_touching_agent(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("fake content", encoding="utf-8")

    def fake_read(call_id: str, args: dict) -> ToolResult:
        return ToolResult(
            call_id=call_id,
            name="read_file",
            content="INJECTED",
        )

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="read_file",
            description="fake",
            parameters=READ_FILE_TOOL.parameters,
            handler=fake_read,
        )
    )

    loop = AgentLoop(llm=_readme_mock(str(readme)), tools=registry)
    session = Session()
    loop.run_turn(session, "read")

    tool_results = [e for e in session.events if e.type == "tool/result"]
    assert tool_results[0].payload["content"] == "INJECTED"


def test_can_inspect_entire_execution(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("x", encoding="utf-8")

    session = Session()
    loop = AgentLoop(llm=_readme_mock(str(readme)), tools=default_registry())
    loop.run_turn(session, "inspect me")

    assert len(session.events) > 0
    assert all(e.session_id == session.session_id for e in session.events)
    assert session.derive_messages()  # projection works


def test_can_force_tool_call(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("data", encoding="utf-8")

    llm = MockLLMProvider(
        steps=[
            MockStep(
                step=1,
                tool_calls=[
                    ToolCallDraft(
                        call_id="forced",
                        name="read_file",
                        arguments={"path": str(readme)},
                    )
                ],
            ),
            MockStep(step=2, content="done"),
        ]
    )
    session = Session()
    AgentLoop(llm=llm, tools=default_registry()).run_turn(session, "force tool")

    calls = [e for e in session.events if e.type == "tool/call"]
    assert len(calls) == 1
    assert calls[0].payload["name"] == "read_file"


def test_can_force_tool_failure() -> None:
    llm = MockLLMProvider(
        steps=[
            MockStep(
                step=1,
                tool_calls=[
                    ToolCallDraft(
                        call_id="bad",
                        name="read_file",
                        arguments={"path": "/nonexistent/file.txt"},
                    )
                ],
            ),
            MockStep(step=2, content="handled failure"),
        ]
    )
    session = Session()
    AgentLoop(llm=llm, tools=default_registry()).run_turn(session, "fail")

    results = [e for e in session.events if e.type == "tool/result"]
    assert results[0].payload["is_error"] is True


def test_deterministic_mock_replay() -> None:
    steps = [MockStep(step=1, content="same every time")]
    llm = MockLLMProvider(steps=steps)

    s1, s2 = Session(), Session()
    AgentLoop(llm=llm, tools=default_registry()).run_turn(s1, "a")
    llm2 = MockLLMProvider(steps=steps)
    AgentLoop(llm=llm2, tools=default_registry()).run_turn(s2, "a")

    msgs1 = [e for e in s1.events if e.type == "assistant/message"]
    msgs2 = [e for e in s2.events if e.type == "assistant/message"]
    assert msgs1[0].payload == msgs2[0].payload


def test_malformed_tool_args_inject_context() -> None:
    llm = MockLLMProvider(
        steps=[
            MockStep(
                step=1,
                tool_calls=[
                    ToolCallDraft(
                        call_id="bad",
                        name="read_file",
                        arguments={"path": 123},
                    )
                ],
            ),
            MockStep(step=2, content="recovered"),
        ]
    )
    session = Session()
    AgentLoop(llm=llm, tools=default_registry()).run_turn(session, "bad args")

    assert any(e.type == "inject/context" for e in session.events)
    assert not any(e.type == "tool/call" for e in session.events)


def test_agent_validates_allowed_tools() -> None:
    llm = MockLLMProvider(
        steps=[
            MockStep(
                step=1,
                tool_calls=[
                    ToolCallDraft(
                        call_id="x",
                        name="read_file",
                        arguments={"path": "x"},
                    )
                ],
            )
        ]
    )
    registry = default_registry()
    loop = AgentLoop(
        llm=llm,
        tools=registry,
        config=LoopConfig(allowed_tools=[]),
    )
    session = Session()
    result = loop.run_turn(session, "denied")

    assert not any(e.type == "tool/call" for e in session.events)
    assert result.status in ("completed", "error", "stopped")
