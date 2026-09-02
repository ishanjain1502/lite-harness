from __future__ import annotations

from pathlib import Path

import pytest

from liteness.repl import ReplConfig, ReplSession
from liteness.llm import MockLLMProvider, MockStep, ToolCallDraft
from liteness.loop import AgentLoop
from liteness.session import Session
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

