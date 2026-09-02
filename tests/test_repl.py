from __future__ import annotations

import threading
from pathlib import Path

import pytest

import liteness.repl as repl_module
from liteness.repl import ReplConfig, ReplSession
from liteness.llm import MockLLMProvider, MockStep, ToolCallDraft
from liteness.loop import AgentLoop
from liteness.session import Session
from liteness.harness import create_runtime
from liteness.testing import registry_with_plugins
from liteness.types import CancelledError, LlmError


def _config(**overrides) -> ReplConfig:
    base = dict(
        preset=None, provider="mock", model=None, project_id="default",
        max_steps=10, session_file=None, readme="README.md",
        telemetry=False, verbose=False,
    )
    base.update(overrides)
    return ReplConfig(**base)


def _iter_input(lines: list[str], *, eof: bool = False):
    it = iter(lines)

    def _fn(_prompt: str = "") -> str:
        try:
            return next(it)
        except StopIteration as e:
            raise EOFError from e

    return _fn


def test_repl_exits_on_quit_command(tmp_path: Path) -> None:
    outputs: list[str] = []
    repl = ReplSession(_config(), input_fn=_iter_input([":q"]), output_fn=outputs.append)
    code = repl.run()
    assert code == 0
    joined = "\n".join(outputs)
    assert "liteness" in joined.lower()
    assert repl.session._turn == 0


def test_repl_quit_disposes_injected_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    disposed: list[object] = []
    runtime = object()
    repl = ReplSession(_config(), input_fn=_iter_input([":q"]), output_fn=lambda _: None)
    repl.session = Session()
    repl.runtime = runtime  # type: ignore[assignment]
    repl.loop = AgentLoop(
        llm=MockLLMProvider(steps=[MockStep(step=1, content="ok")]),
        tools=registry_with_plugins(),
    )
    monkeypatch.setattr(repl_module, "dispose_runtime", disposed.append)

    assert repl.run() == 0

    assert disposed == [runtime]
    assert repl.runtime is None


def test_repl_run_preserves_injected_components(monkeypatch: pytest.MonkeyPatch) -> None:
    session = Session()
    runtime = object()
    loop = AgentLoop(
        llm=MockLLMProvider(steps=[MockStep(step=1, content="ok")]),
        tools=registry_with_plugins(),
    )
    repl = ReplSession(_config(), input_fn=_iter_input([]), output_fn=lambda _: None)
    repl.session = session
    repl.runtime = runtime  # type: ignore[assignment]
    repl.loop = loop
    monkeypatch.setattr(repl, "_open_session", lambda: pytest.fail("session was rebuilt"))
    monkeypatch.setattr(repl, "_build_runtime", lambda: pytest.fail("runtime was rebuilt"))
    monkeypatch.setattr(repl, "_build_loop", lambda: pytest.fail("loop was rebuilt"))
    monkeypatch.setattr(repl, "_read_loop", lambda: 0)

    assert repl.run() == 0

    assert repl.session is session
    assert repl.runtime is runtime
    assert repl.loop is loop


def test_repl_exits_on_eof(tmp_path: Path) -> None:
    outputs: list[str] = []
    repl = ReplSession(_config(), input_fn=_iter_input([]), output_fn=outputs.append)
    assert repl.run() == 0


def _wire_mock(repl: ReplSession, llm: MockLLMProvider) -> None:
    repl.session = Session()
    repl.runtime = None
    repl.loop = AgentLoop(llm=llm, tools=registry_with_plugins())


def test_repl_runs_turn_and_streams(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("hello world", encoding="utf-8")
    outputs: list[str] = []
    repl = ReplSession(_config(), input_fn=_iter_input(["read it", ":q"]), output_fn=outputs.append)
    _wire_mock(repl, MockLLMProvider(steps=[
        MockStep(step=1, tool_calls=[ToolCallDraft(call_id="c1", name="read_file", arguments={"path": str(readme)})]),
        MockStep(step=2, content="done reading"),
    ]))
    assert repl.run() == 0
    joined = "\n".join(outputs)
    assert "-> tool: read_file" in joined
    assert "-> result:" in joined
    rendered = "".join(outputs)
    assert "done reading" in rendered
    assert rendered.count("done reading") == 1
    assert repl.session._turn == 1


def test_repl_turn_composes_existing_event_sink() -> None:
    outputs: list[str] = []
    recorded_events: list[str] = []
    repl = ReplSession(_config(), output_fn=outputs.append)
    repl.session = Session()
    repl.loop = AgentLoop(
        llm=MockLLMProvider(steps=[MockStep(step=1, content="recorded answer")]),
        tools=registry_with_plugins(),
        event_sink=lambda event: recorded_events.append(event.type),
    )

    original_sink = repl.loop.event_sink
    repl._run_turn("record this")

    assert "assistant/chunk" in recorded_events
    assert "assistant/message" in recorded_events
    assert repl.loop.event_sink is original_sink


def test_repl_blank_lines_are_ignored(tmp_path: Path) -> None:
    outputs: list[str] = []
    repl = ReplSession(_config(), input_fn=_iter_input(["   ", ":q"]), output_fn=outputs.append)
    _wire_mock(repl, MockLLMProvider(steps=[MockStep(step=1, content="ok")]))
    assert repl.run() == 0
    assert repl.session._turn == 0


def test_repl_preset_switch_in_place(tmp_path: Path) -> None:
    outputs: list[str] = []
    repl = ReplSession(
        _config(preset="coder"),
        input_fn=_iter_input(["hi", ":preset researcher", ":tools", ":history", ":q"]),
        output_fn=outputs.append,
    )
    # Build an initial coder runtime + loop so the switch has something to dispose.
    repl.session = Session()
    repl.runtime = create_runtime(preset_name="coder", session=repl.session, project_id="default")
    repl.loop = AgentLoop(
        llm=MockLLMProvider(steps=[MockStep(step=1, content="ok")]),
        tools=repl.runtime.ctx.tools,
    )
    before_id = repl.session.session_id
    repl.run()
    joined = "\n".join(outputs)
    assert "switched to preset: researcher" in joined
    assert ("memory" in joined) or ("rag" in joined)
    assert "turn=1" in joined
    assert repl.session.session_id == before_id  # same session retained
    assert repl.session._turn == 1

def test_repl_resumes_from_session_file(tmp_path: Path) -> None:
    log = tmp_path / "repl.jsonl"
    readme = tmp_path / "README.md"
    readme.write_text("resume me", encoding="utf-8")

    # First REPL session: two turns, then quit.
    outputs1: list[str] = []
    repl1 = ReplSession(
        _config(session_file=str(log)),
        input_fn=_iter_input(["read it", "again", ":q"]),
        output_fn=outputs1.append,
    )
    repl1.session = Session.open(log)
    repl1.runtime = None
    repl1.loop = AgentLoop(
        llm=MockLLMProvider(steps=[
            MockStep(step=1, tool_calls=[ToolCallDraft(call_id="c1", name="read_file", arguments={"path": str(readme)})]),
            MockStep(step=2, content="first answer"),
        ]),
        tools=registry_with_plugins(),
    )
    repl1.run()
    # MockLLMProvider scripts produce multiple steps per turn; two prompts => two turns.
    assert repl1.session._turn == 2

    # Second REPL session: same --session-file, one more turn → continues at turn 2.
    outputs2: list[str] = []
    repl2 = ReplSession(
        _config(session_file=str(log)),
        input_fn=_iter_input(["continue", ":q"]),
        output_fn=outputs2.append,
    )
    repl2.session = Session.open(log)
    repl2.runtime = None
    repl2.loop = AgentLoop(
        llm=MockLLMProvider(steps=[MockStep(step=1, content="second answer")]),
        tools=registry_with_plugins(),
    )
    repl2.run()
    # Continuing from the existing log should advance to turn 3.
    assert repl2.session._turn == 3
    assert "second answer" in "".join(outputs2)
    # no runtime assertions here


def test_repl_failed_preset_creation_keeps_old_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs: list[str] = []
    old_runtime = object()
    old_loop = object()
    repl = ReplSession(_config(preset="coder"), output_fn=outputs.append)
    repl.session = Session()
    repl.runtime = old_runtime  # type: ignore[assignment]
    repl.loop = old_loop  # type: ignore[assignment]
    disposed: list[object] = []
    monkeypatch.setattr(repl_module, "dispose_runtime", disposed.append)

    def fail_create_runtime(**_kwargs):
        raise RuntimeError("preset creation failed")

    monkeypatch.setattr(repl_module, "create_runtime", fail_create_runtime)

    repl._do_preset("researcher")

    assert repl.runtime is old_runtime
    assert repl.loop is old_loop
    assert disposed == []
    assert "Error switching preset: preset creation failed" in outputs


def test_repl_failed_preset_loop_build_keeps_old_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs: list[str] = []
    old_runtime = object()
    new_runtime = object()
    old_loop = object()
    repl = ReplSession(_config(preset="coder"), output_fn=outputs.append)
    repl.session = Session()
    repl.runtime = old_runtime  # type: ignore[assignment]
    repl.loop = old_loop  # type: ignore[assignment]
    disposed: list[object] = []
    monkeypatch.setattr(repl_module, "dispose_runtime", disposed.append)
    monkeypatch.setattr(repl_module, "create_runtime", lambda **_kwargs: new_runtime)

    def fail_build_loop(*, runtime=None):
        assert runtime is new_runtime
        raise RuntimeError("loop creation failed")

    monkeypatch.setattr(repl, "_build_loop", fail_build_loop)

    repl._do_preset("researcher")

    assert repl.runtime is old_runtime
    assert repl.loop is old_loop
    assert disposed == [new_runtime]
    assert "Error switching preset: loop creation failed" in outputs

def test_repl_continues_after_turn_error() -> None:
    outputs: list[str] = []
    repl = ReplSession(_config(), input_fn=_iter_input(["bad", "good", ":q"]), output_fn=outputs.append)

    repl.session = Session()
    repl.runtime = None
    repl.loop = AgentLoop(
        llm=MockLLMProvider(steps=[MockStep(step=1, content="recovered")]),
        tools=registry_with_plugins(),
    )

    original_run_turn = repl.loop.run_turn
    calls = {"n": 0}

    def _run_turn(session, prompt, cancel=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise LlmError("LLM_ERROR", "boom", retryable=False)
        return original_run_turn(session, prompt, cancel=cancel)

    repl.loop.run_turn = _run_turn  # type: ignore[assignment]
    code = repl.run()
    joined = "\n".join(outputs)
    assert code == 0
    assert "error" in joined.lower() or "llm_error" in joined.lower()
    assert "recovered" in "".join(outputs)
    assert repl.session._turn == 1


def test_repl_reraises_cancelled_error() -> None:
    outputs: list[str] = []
    repl = ReplSession(_config(), output_fn=outputs.append)
    repl.session = Session()
    repl.loop = AgentLoop(
        llm=MockLLMProvider(steps=[]),
        tools=registry_with_plugins(),
    )
    original_sink = repl.loop.event_sink

    def _cancelled_turn(session, prompt, cancel=None):
        raise CancelledError()

    repl.loop.run_turn = _cancelled_turn  # type: ignore[assignment]

    with pytest.raises(CancelledError):
        repl._run_turn("cancel me")

    assert repl.loop.event_sink is original_sink
    assert not any("unexpected error" in output for output in outputs)


def test_repl_cancel_mid_turn_returns_to_prompt(tmp_path: Path) -> None:
    outputs: list[str] = []
    repl = ReplSession(_config(), input_fn=_iter_input(["slow", ":q"]), output_fn=outputs.append)

    started = threading.Event()
    cancel_event = threading.Event()

    class _BlockingLLM(MockLLMProvider):
        def stream(self, request):
            started.set()
            cancel_event.wait(5.0)
            from liteness.types import LlmError
            raise LlmError("LLM_ERROR", "should have been cancelled", retryable=False)

    repl.session = Session()
    repl.runtime = None
    repl.loop = AgentLoop(llm=_BlockingLLM(steps=[]), tools=registry_with_plugins())

    # Thread that trips the per-turn CancelToken once the turn starts.
    def _trip():
        started.wait(5.0)
        assert repl._current_cancel is not None
        repl._current_cancel.cancel()

    t = threading.Thread(target=_trip)
    t.start()
    code = repl.run()
    t.join(timeout=5)
    joined = "\n".join(outputs)
    assert code == 0
    assert "user_cancelled" in joined
    # REPL returned to the prompt and processed :q
    assert repl.session._turn == 1


def test_repl_fatal_unknown_preset(tmp_path: Path) -> None:
    outputs: list[str] = []
    repl = ReplSession(
        _config(preset="nope"),
        input_fn=_iter_input([]),
        output_fn=outputs.append,
    )
    assert repl.run() == 1
    assert "Error" in "\n".join(outputs)


def test_repl_fatal_unknown_provider(tmp_path: Path) -> None:
    outputs: list[str] = []
    repl = ReplSession(
        _config(provider="bogus"),
        input_fn=_iter_input([]),
        output_fn=outputs.append,
    )
    assert repl.run() == 1
    assert "Error" in "\n".join(outputs)

