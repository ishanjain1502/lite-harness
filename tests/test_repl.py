from __future__ import annotations

from pathlib import Path

import pytest

import liteness.repl as repl_module
from liteness.repl import ReplConfig, ReplSession
from liteness.llm import MockLLMProvider, MockStep, ToolCallDraft
from liteness.loop import AgentLoop
from liteness.session import Session
from liteness.harness import create_runtime
from liteness.testing import registry_with_plugins


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
    assert "done reading" in joined
    assert repl.session._turn == 1


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
    assert repl.runtime is None


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

