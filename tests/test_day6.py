"""Day 6 tests — persistence, recovery, replay, resume."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from liteness.llm import LlmRequest, MockLLMProvider, MockStep, ToolCallDraft
from liteness.loop import AgentLoop, LoopConfig
from liteness.replay import ReplayLLMProvider
from liteness.session import (
    Session,
    SessionEvent,
    find_orphan_tool_calls,
    recover_orphans,
)
from liteness.tools import default_registry


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
            MockStep(step=2, content="summary from disk"),
        ]
    )


def test_persist_events_to_jsonl(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("persist me", encoding="utf-8")
    log_path = tmp_path / "session.jsonl"

    session = Session.open(log_path)
    loop = AgentLoop(llm=_readme_mock(str(readme)), tools=default_registry())
    loop.run_turn(session, "read and summarize")

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(session.events)
    assert session.events[0].type == "turn/start"


def test_load_session_from_jsonl(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("reload", encoding="utf-8")
    log_path = tmp_path / "session.jsonl"

    original = Session.open(log_path)
    AgentLoop(llm=_readme_mock(str(readme)), tools=default_registry()).run_turn(
        original, "task"
    )

    loaded = Session.load_from_jsonl(log_path)
    assert loaded.session_id == original.session_id
    assert len(loaded.events) == len(original.events)
    assert loaded.events[-1].type == "turn/end"
    assert loaded._turn == original._turn


def test_find_orphan_tool_calls() -> None:
    events = [
        SessionEvent(
            id="1",
            type="tool/call",
            timestamp=SessionEvent.from_dict(
                {
                    "id": "x",
                    "type": "tool/call",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "session_id": "s",
                    "turn": 1,
                    "step": 1,
                    "payload": {
                        "call_id": "orphan-1",
                        "name": "read_file",
                        "arguments": {"path": "a"},
                    },
                }
            ).timestamp,
            session_id="s",
            turn=1,
            step=1,
            payload={
                "call_id": "orphan-1",
                "name": "read_file",
                "arguments": {"path": "a"},
            },
        ),
        SessionEvent(
            id="2",
            type="tool/call",
            timestamp=SessionEvent.from_dict(
                {
                    "id": "y",
                    "type": "tool/call",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "session_id": "s",
                    "turn": 1,
                    "step": 1,
                    "payload": {
                        "call_id": "ok-1",
                        "name": "read_file",
                        "arguments": {"path": "b"},
                    },
                }
            ).timestamp,
            session_id="s",
            turn=1,
            step=1,
            payload={
                "call_id": "ok-1",
                "name": "read_file",
                "arguments": {"path": "b"},
            },
        ),
        SessionEvent(
            id="3",
            type="tool/result",
            timestamp=SessionEvent.from_dict(
                {
                    "id": "z",
                    "type": "tool/result",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "session_id": "s",
                    "turn": 1,
                    "step": 1,
                    "payload": {
                        "call_id": "ok-1",
                        "name": "read_file",
                        "content": "b",
                        "is_error": False,
                    },
                }
            ).timestamp,
            session_id="s",
            turn=1,
            step=1,
            payload={
                "call_id": "ok-1",
                "name": "read_file",
                "content": "b",
                "is_error": False,
            },
        ),
    ]

    orphans = find_orphan_tool_calls(events)
    assert len(orphans) == 1
    assert orphans[0].call_id == "orphan-1"


def test_recover_orphans_appends_synthetic_result() -> None:
    session = Session(session_id="s")
    session.append(
        "tool/call",
        {
            "call_id": "orphan-1",
            "name": "read_file",
            "arguments": {"path": "missing"},
        },
        turn=1,
        step=1,
    )

    recovered = recover_orphans(session)
    assert len(recovered) == 1
    assert recovered[0].type == "tool/result"
    assert recovered[0].payload["error_code"] == "RECOVERY_INCOMPLETE"
    assert find_orphan_tool_calls(session.events) == []


def test_resume_next_turn_from_jsonl(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("resume test", encoding="utf-8")
    log_path = tmp_path / "session.jsonl"

    session = Session.open(log_path)
    loop = AgentLoop(llm=_readme_mock(str(readme)), tools=default_registry())
    first = loop.run_turn(session, "first turn")
    assert first.stop_reason.value == "completed"

    resumed = Session.load_from_jsonl(log_path, recover=True)
    assert resumed._turn == 1

    llm = MockLLMProvider(steps=[MockStep(step=1, content="second turn answer")])
    second = AgentLoop(llm=llm, tools=default_registry()).run_turn(
        resumed, "second turn"
    )

    assert second.turn == 2
    assert second.final_output == "second turn answer"
    assert any(e.type == "turn/start" and e.turn == 2 for e in resumed.events)


def test_load_with_recover_heals_orphan_on_disk(tmp_path: Path) -> None:
    log_path = tmp_path / "broken.jsonl"
    orphan_event = {
        "id": "e1",
        "type": "tool/call",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "session_id": "sess1",
        "turn": 1,
        "step": 1,
        "payload": {
            "call_id": "call-orphan",
            "name": "read_file",
            "arguments": {"path": "x"},
        },
    }
    log_path.write_text(json.dumps(orphan_event) + "\n", encoding="utf-8")

    session = Session.load_from_jsonl(log_path, recover=True)
    results = [e for e in session.events if e.type == "tool/result"]
    assert len(results) == 1
    assert results[0].payload["error_code"] == "RECOVERY_INCOMPLETE"
    assert log_path.read_text(encoding="utf-8").count("tool/result") == 1


def test_replay_llm_provider_from_session(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("replay", encoding="utf-8")
    log_path = tmp_path / "session.jsonl"

    session = Session.open(log_path)
    AgentLoop(llm=_readme_mock(str(readme)), tools=default_registry()).run_turn(
        session, "record for replay"
    )

    replay = ReplayLLMProvider.from_session(session)
    request = LlmRequest(
        session_id=session.session_id,
        turn=1,
        step=1,
        messages=[],
        tools=[],
    )
    step1 = list(replay.stream(request))
    assert any(c.tool_call_deltas for c in step1)


def test_export_jsonl(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("export", encoding="utf-8")
    log_path = tmp_path / "session.jsonl"
    export_path = tmp_path / "export.jsonl"

    session = Session.open(log_path)
    AgentLoop(llm=_readme_mock(str(readme)), tools=default_registry()).run_turn(
        session, "export me"
    )

    session.export_jsonl(export_path)
    exported = Session.load_from_jsonl(export_path)
    assert len(exported.events) == len(session.events)
    assert exported.session_id == session.session_id


def test_tool_call_persisted_before_result_in_log(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("ordering", encoding="utf-8")
    log_path = tmp_path / "session.jsonl"

    session = Session.open(log_path)
    AgentLoop(llm=_readme_mock(str(readme)), tools=default_registry()).run_turn(
        session, "check order"
    )

    types = [json.loads(line)["type"] for line in log_path.read_text().splitlines()]
    call_idx = types.index("tool/call")
    result_idx = types.index("tool/result")
    assert call_idx < result_idx
