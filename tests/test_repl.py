from __future__ import annotations

from pathlib import Path

import pytest

from liteness.repl import ReplConfig, ReplSession


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

