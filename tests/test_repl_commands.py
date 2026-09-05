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
        telemetry=False, verbose=False, debug=False, eval_file=None,
        clickhouse=False, clickhouse_full=False,
        clickhouse_url=None, clickhouse_database="liteness",
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


def test_repl_reset_clears_in_memory_session(tmp_path: Path) -> None:
    outputs: list[str] = []
    repl = ReplSession(_config(), input_fn=_iter_input(["hi", ":reset", ":q"]), output_fn=outputs.append)
    _wire_mock(repl, MockLLMProvider(steps=[MockStep(step=1, content="answer")]))
    repl.run()
    assert repl.session._turn == 0
    assert repl.session.events == []
    assert "session reset" in "\n".join(outputs)


def test_repl_reset_truncates_log_file(tmp_path: Path) -> None:
    log = tmp_path / "repl.jsonl"
    outputs: list[str] = []
    repl = ReplSession(
        _config(session_file=str(log)),
        input_fn=_iter_input(["hi", ":reset", ":q"]),
        output_fn=outputs.append,
    )
    _wire_mock(repl, MockLLMProvider(steps=[MockStep(step=1, content="answer")]))
    repl.session = Session.open(log)
    repl.run()
    assert log.exists()
    assert log.stat().st_size == 0
    # same session id retained
    assert repl.session.session_id


def test_repl_eval_command_after_turn(tmp_path: Path) -> None:
    outputs: list[str] = []
    repl = ReplSession(
        _config(),
        input_fn=_iter_input(["hi", "eval", ":q"]),
        output_fn=outputs.append,
    )
    _wire_mock(repl, MockLLMProvider(steps=[MockStep(step=1, content="answer")]))
    repl.run()
    joined = "\n".join(outputs)
    assert "Eval Report: repl" in joined
    assert "event_integrity" in joined


def test_repl_eval_with_suite_case(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """
name: repl-suite
evaluators:
  - stop_reason
cases:
  - id: turn-001
    expected:
      stop_reason: completed
""".strip()
        + "\n",
        encoding="utf-8",
    )
    outputs: list[str] = []
    repl = ReplSession(
        _config(eval_file=str(suite_path)),
        input_fn=_iter_input(["hi", "eval --case turn-001", ":q"]),
        output_fn=outputs.append,
    )
    _wire_mock(repl, MockLLMProvider(steps=[MockStep(step=1, content="answer")]))
    repl.run()
    joined = "\n".join(outputs)
    assert "Eval Report: repl-suite" in joined
    assert "Passed:" in joined


def test_repl_eval_case_requires_eval_file(tmp_path: Path) -> None:
    outputs: list[str] = []
    repl = ReplSession(
        _config(),
        input_fn=_iter_input(["hi", "eval --case turn-001", ":q"]),
        output_fn=outputs.append,
    )
    _wire_mock(repl, MockLLMProvider(steps=[MockStep(step=1, content="answer")]))
    repl.run()
    assert any("eval --case requires --eval-file" in o for o in outputs)


def test_repl_colon_eval_alias(tmp_path: Path) -> None:
    outputs: list[str] = []
    repl = ReplSession(
        _config(),
        input_fn=_iter_input(["hi", ":eval", ":q"]),
        output_fn=outputs.append,
    )
    _wire_mock(repl, MockLLMProvider(steps=[MockStep(step=1, content="answer")]))
    repl.run()
    assert any("Eval Report:" in o for o in outputs)

