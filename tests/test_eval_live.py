"""Day 9 live eval tests — agent execution + evaluation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from liteness.cli import main
from liteness.eval import LiveEvalConfig, compare_baseline, run_suite_live
from liteness.eval.live import execute_live_case, weather_mock_llm
from liteness.eval.models import EvalCase, EvalSuite, ExpectedBehavior
from liteness.eval.runner import EvalRunner
from liteness.loop import AgentLoop
from liteness.tools import ToolDefinition, ToolRegistry, ToolResult

EXAMPLES = Path(__file__).parent.parent / "examples" / "evals"
BASELINES = Path(__file__).parent / "fixtures" / "evals" / "baselines"


def _weather_registry() -> ToolRegistry:
    def weather_handler(call_id: str, args: dict) -> ToolResult:
        return ToolResult(
            call_id=call_id,
            name="weather",
            content=f"Sunny in {args.get('city', 'unknown')}",
        )

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="weather",
            description="get weather",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
            handler=weather_handler,
        )
    )
    return registry


@pytest.mark.eval_live
def test_eval_run_live_readme_mock(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# lite-ness\n\nA tiny harness.\n", encoding="utf-8")

    config = LiveEvalConfig(
        sessions_dir=tmp_path / "sessions",
        readme=str(readme),
        provider="mock",
    )
    report = run_suite_live(EXAMPLES / "live-readme.yaml", live_config=config)

    assert report.summary["total"] == 1
    assert report.summary["passed"] == 1
    assert report.cases[0].id == "readme-001"
    session_path = Path(report.cases[0].session_path)
    assert session_path.exists()
    assert session_path.read_text(encoding="utf-8").count("turn/end") >= 1


@pytest.mark.eval_live
def test_eval_run_live_weather_injected_mock(tmp_path: Path) -> None:
    suite = EvalSuite(
        name="live-weather",
        evaluators=["stop_reason", "tool_lifecycle", "event_integrity"],
        cases=[
            EvalCase(
                id="weather-live-001",
                input="What's the weather in Delhi?",
                expected=ExpectedBehavior(
                    required_tools=["weather"],
                    stop_reason="completed",
                ),
            )
        ],
    )
    config = LiveEvalConfig(
        sessions_dir=tmp_path / "sessions",
        llm=weather_mock_llm(),
        tools=_weather_registry(),
    )
    report = EvalRunner(live_config=config).run(suite, live=True)

    assert report.summary["passed"] == 1
    assert any(
        result.evaluator == "tool_lifecycle" and result.passed
        for result in report.cases[0].results
    )


@pytest.mark.eval_live
def test_eval_run_invokes_agent_loop(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("hello", encoding="utf-8")
    config = LiveEvalConfig(
        sessions_dir=tmp_path / "sessions",
        readme=str(readme),
    )

    original = AgentLoop.run_turn
    calls: list[str] = []

    def tracking(self, session, user_message):  # noqa: ANN001
        calls.append(user_message)
        return original(self, session, user_message)

    with patch.object(AgentLoop, "run_turn", tracking):
        report = run_suite_live(EXAMPLES / "live-readme.yaml", live_config=config)

    assert len(calls) == 1
    assert report.summary["passed"] == 1


@pytest.mark.eval_live
def test_eval_run_live_matches_baseline(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# Demo\n", encoding="utf-8")
    config = LiveEvalConfig(
        sessions_dir=tmp_path / "sessions",
        readme=str(readme),
    )
    report = run_suite_live(EXAMPLES / "live-readme.yaml", live_config=config)
    compare_baseline(report, BASELINES / "live-readme.json", fail_on_regression=True)


@pytest.mark.eval_live
def test_eval_run_cli(tmp_path: Path, capsys) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("demo", encoding="utf-8")
    exit_code = main(
        [
            "eval",
            "run",
            str(EXAMPLES / "live-readme.yaml"),
            "--readme",
            str(readme),
            "--sessions-dir",
            str(tmp_path / "sessions"),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Eval Report: live-readme" in captured.out


@pytest.mark.eval_live
def test_execute_live_case_writes_session(tmp_path: Path) -> None:
    case = EvalCase(id="weather-live-001", input="weather?")
    suite = EvalSuite(name="live", cases=[case])
    config = LiveEvalConfig(
        sessions_dir=tmp_path / "sessions",
        llm=weather_mock_llm(),
        tools=_weather_registry(),
    )
    path = execute_live_case(case, suite, config)
    assert path.exists()
    assert "tool/call" in path.read_text(encoding="utf-8")
