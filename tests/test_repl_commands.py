from __future__ import annotations

from pathlib import Path

from liteness.repl import ReplConfig, ReplSession
from liteness.llm import MockLLMProvider, MockStep
from liteness.testing import registry_with_plugins
from liteness.session import Session


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


def _wire_mock(repl: ReplSession, llm: MockLLMProvider) -> None:
    repl.session = Session()
    repl.runtime = None
    repl.loop = llm and __import__("liteness.loop", fromlist=["AgentLoop"]).AgentLoop(llm=llm, tools=registry_with_plugins())


def test_repl_tools_command(tmp_path: Path) -> None:
    outputs: list[str] = []
    repl = ReplSession(_config(), input_fn=_iter_input([":tools", ":q"]), output_fn=outputs.append)
    _wire_mock(repl, MockLLMProvider(steps=[MockStep(step=1, content="ok")]))
    repl.run()
    joined = "\n".join(outputs)
    assert "read_file" in joined  # from registry_with_plugins filesystem


def test_repl_history_command(tmp_path: Path) -> None:
    outputs: list[str] = []
    repl = ReplSession(_config(), input_fn=_iter_input(["hi", ":history", ":q"]), output_fn=outputs.append)
    _wire_mock(repl, MockLLMProvider(steps=[MockStep(step=1, content="answer")]))
    repl.run()
    joined = "\n".join(outputs)
    assert "user/message" in joined or "turn/start" in joined


def test_repl_report_command(tmp_path: Path) -> None:
    outputs: list[str] = []
    repl = ReplSession(_config(), input_fn=_iter_input(["hi", ":report", ":q"]), output_fn=outputs.append)
    _wire_mock(repl, MockLLMProvider(steps=[MockStep(step=1, content="answer")]))
    repl.run()
    joined = "\n".join(outputs)
    assert "trace" in joined.lower() or "turn" in joined.lower()


def test_repl_unknown_command(tmp_path: Path) -> None:
    outputs: list[str] = []
    repl = ReplSession(_config(), input_fn=_iter_input([":bogus", ":q"]), output_fn=outputs.append)
    _wire_mock(repl, MockLLMProvider(steps=[MockStep(step=1, content="ok")]))
    repl.run()
    assert any("unknown command: :bogus" in o for o in outputs)
    assert repl.session._turn == 0  # :bogus did not run a turn

