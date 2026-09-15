from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from liteness.cli import _build_runtime, _format_error_message, _summarize_session
from liteness.loop import TurnResult
from liteness.session import SessionEvent
from liteness.types import StopReason
from liteness.harness import HarnessRuntime
from liteness.plugins.base import PluginConfigError
from liteness.session import Session


def _args(**overrides) -> argparse.Namespace:
    base = dict(preset=None, project_id="default")
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_runtime_returns_none_when_no_preset(tmp_path: Path) -> None:
    session = Session()
    assert _build_runtime(_args(), session) is None


def test_build_runtime_returns_runtime_for_known_preset(tmp_path: Path) -> None:
    session = Session()
    runtime = _build_runtime(_args(preset="coder"), session)
    assert isinstance(runtime, HarnessRuntime)
    assert "read_file" in runtime.ctx.tools.names()


def test_build_runtime_raises_for_unknown_preset(tmp_path: Path) -> None:
    session = Session()
    with pytest.raises(PluginConfigError):
        _build_runtime(_args(preset="nope"), session)


def test_repl_subcommand_is_registered(capsys) -> None:
    from liteness.cli import main
    
    # Running top-level help shows the list of subcommands and their short help.
    with pytest.raises(SystemExit):
        main(["--help"])
    captured = capsys.readouterr()
    assert "repl" in captured.out
    assert "Interactive REPL" in captured.out


def test_format_error_message_extracts_nested_api_json() -> None:
    raw = (
        '{"error":{"message":"You have insufficient credits.",'
        '"type":"invalid_request_error","code":"BAD_REQUEST"}}'
    )
    assert _format_error_message(raw) == "You have insufficient credits."


def test_summarize_session_includes_error_reason() -> None:
    session = Session()
    session._turn = 1
    session.events.extend(
        [
            SessionEvent(
                id="1",
                type="step/start",
                timestamp="2026-01-01T00:00:00+00:00",
                session_id=session.session_id,
                turn=1,
                step=1,
                payload={"step": 1},
            ),
            SessionEvent(
                id="2",
                type="error",
                timestamp="2026-01-01T00:00:01+00:00",
                session_id=session.session_id,
                turn=1,
                step=1,
                payload={"code": "LLM_ERROR", "message": "API key invalid"},
            ),
        ]
    )
    result = TurnResult(
        session_id=session.session_id,
        turn=1,
        status="error",
        stop_reason=StopReason.LLM_ERROR,
    )
    summary = _summarize_session(session, result)
    assert "Error (llm_error)" in summary
    assert "Reason: API key invalid" in summary


def test_run_subcommand_returns_status_and_prints_summary(capsys) -> None:
    from liteness.cli import main

    rc = main(["run", "--readme", "README.md", "summarize the project"])

    captured = capsys.readouterr()
    assert isinstance(rc, int)
    assert rc in (0, 1)
    assert "Turn 1" in captured.out

