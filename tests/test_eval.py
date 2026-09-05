"""Day 9 tests — offline evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from liteness.cli import main
from liteness.eval import (
    EvalRunner,
    build_eval_run,
    compare_baseline,
    format_eval_report,
    load_suite,
    run_session_file,
)
from liteness.eval.baseline import BaselineMismatchError
from liteness.eval.errors import EvalLoadError
from liteness.eval.models import EvalCase, EvalSuite, ExpectedBehavior
from liteness.loop import AgentLoop

FIXTURES = Path(__file__).parent / "fixtures" / "evals"
EXAMPLES = Path(__file__).parent.parent / "examples" / "evals"
BASELINES = FIXTURES / "baselines"


@pytest.mark.eval
def test_eval_session_offline_produces_report() -> None:
    session_path = FIXTURES / "weather-001.jsonl"
    suite = load_suite(EXAMPLES / "basic.yaml")
    report = run_session_file(session_path, eval_file=EXAMPLES / "basic.yaml", case_id="weather-001")

    assert report.schema_version == 1
    assert report.suite == "basic"
    assert report.summary["total"] == 1
    assert report.summary["passed"] == 1
    assert len(report.cases) == 1
    assert report.cases[0].id == "weather-001"
    assert report.cases[0].passed is True
    evaluator_names = {result.evaluator for result in report.cases[0].results}
    assert {"stop_reason", "tool_lifecycle", "event_integrity"}.issubset(evaluator_names)


@pytest.mark.eval
def test_eval_session_never_invokes_agent_loop() -> None:
    session_path = FIXTURES / "weather-001.jsonl"
    with patch.object(AgentLoop, "__init__", side_effect=AssertionError("AgentLoop must not be used")):
        with patch.object(AgentLoop, "run_turn", side_effect=AssertionError("AgentLoop must not be used")):
            report = run_session_file(
                session_path,
                eval_file=EXAMPLES / "basic.yaml",
                case_id="weather-001",
            )
    assert report.summary["passed"] == 1


@pytest.mark.eval
def test_eval_suite_yaml_aggregate_report() -> None:
    suite = load_suite(EXAMPLES / "basic.yaml")
    runner = EvalRunner()
    report = runner.run(
        suite,
        session_overrides={"weather-001": FIXTURES / "weather-001.jsonl"},
    )
    assert report.summary["total"] == 1
    assert report.summary["pass_rate"] == 1.0
    assert report.failures == []


@pytest.mark.eval
def test_eval_cli_summary_output(capsys) -> None:
    exit_code = main(
        [
            "eval",
            "session",
            str(FIXTURES / "weather-001.jsonl"),
            "--eval-file",
            str(EXAMPLES / "basic.yaml"),
            "--case",
            "weather-001",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Eval Report: basic" in captured.out
    assert "Passed:" in captured.out


@pytest.mark.eval
def test_eval_baseline_regression_detects_break(tmp_path: Path) -> None:
    session_path = FIXTURES / "weather-001.jsonl"
    report = run_session_file(
        session_path,
        eval_file=EXAMPLES / "basic.yaml",
        case_id="weather-001",
    )

    baseline = {
        "baseline_id": "basic-v1",
        "metrics": {"task_success": 1.0},
        "cases": {
            "weather-001": {"metrics": {"task_success": 1.0}},
        },
    }
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    compare_baseline(report, baseline_path, fail_on_regression=True)

    broken = {
        "baseline_id": "basic-v1",
        "metrics": {"task_success": 1.0},
        "cases": {
            "weather-001": {"metrics": {"task_success": 1.0}},
        },
    }
    broken_path = tmp_path / "broken.json"
    broken_path.write_text(json.dumps(broken), encoding="utf-8")

    # Force a failing report by expecting wrong stop_reason
    failing_suite = EvalSuite(
        name="broken",
        cases=[
            EvalCase(
                id="weather-001",
                session_file=str(session_path),
                expected=ExpectedBehavior(
                    stop_reason="budget_exceeded",
                    required_tools=["weather"],
                ),
            )
        ],
        evaluators=["stop_reason", "tool_lifecycle"],
    )
    failing_report = EvalRunner().run(
        failing_suite,
        session_overrides={"weather-001": session_path},
    )
    with pytest.raises(BaselineMismatchError):
        compare_baseline(failing_report, broken_path, fail_on_regression=True)


@pytest.mark.eval
def test_evaluation_result_json_schema_stable() -> None:
    session_path = FIXTURES / "weather-001.jsonl"
    report = run_session_file(
        session_path,
        eval_file=EXAMPLES / "basic.yaml",
        case_id="weather-001",
    )
    payload = report.to_dict()
    assert payload["schema_version"] == 1
    assert set(payload.keys()) >= {
        "schema_version",
        "suite",
        "timestamp",
        "summary",
        "evaluators_summary",
        "cases",
        "failures",
    }
    case = payload["cases"][0]
    result = case["results"][0]
    assert set(result.keys()) == {
        "evaluator",
        "evaluator_version",
        "case_id",
        "passed",
        "score",
        "reason",
        "metadata",
    }


@pytest.mark.eval
def test_corrupt_jsonl_raises_eval_load_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n", encoding="utf-8")
    case = EvalCase(id="bad", session_file=str(bad))
    with pytest.raises(EvalLoadError):
        build_eval_run(case, bad)


@pytest.mark.eval
def test_format_eval_report_verbose() -> None:
    session_path = FIXTURES / "weather-001.jsonl"
    report = run_session_file(
        session_path,
        eval_file=EXAMPLES / "basic.yaml",
        case_id="weather-001",
    )
    text = format_eval_report(report, verbose=True)
    assert "weather-001" in text
    assert "stop_reason" in text


@pytest.mark.eval
def test_committed_baseline_offline_passes() -> None:
    session_path = FIXTURES / "weather-001.jsonl"
    report = run_session_file(
        session_path,
        eval_file=EXAMPLES / "basic.yaml",
        case_id="weather-001",
    )
    compare_baseline(report, BASELINES / "basic.json", fail_on_regression=True)


@pytest.mark.eval
def test_eval_compare_cli(tmp_path: Path, capsys) -> None:
    session_path = FIXTURES / "weather-001.jsonl"
    report = run_session_file(
        session_path,
        eval_file=EXAMPLES / "basic.yaml",
        case_id="weather-001",
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report.to_dict()), encoding="utf-8")

    exit_code = main(
        [
            "eval",
            "compare",
            str(report_path),
            "--baseline",
            str(BASELINES / "basic.json"),
            "--fail-on-regression",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Baseline compare: PASS" in captured.out
