"""Baseline comparison for eval regression."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from liteness.eval.errors import BaselineMismatchError, EvalLoadError
from liteness.eval.models import EvalReport


def load_report(path: Path | str) -> EvalReport:
    """Load an EvalReport from a JSON file."""
    report_path = Path(path)
    if not report_path.exists():
        raise EvalLoadError(f"eval report not found: {report_path}")
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvalLoadError(f"invalid eval report JSON: {report_path}") from exc
    return EvalReport.from_dict(data)


def compare_baseline(
    report: EvalReport,
    baseline_path: Path | str,
    *,
    fail_on_regression: bool = True,
    epsilon: float = 0.0,
) -> dict[str, Any]:
    """Compare an eval report against a committed baseline JSON file."""
    path = Path(baseline_path)
    if not path.exists():
        raise BaselineMismatchError(f"baseline not found: {path}")

    baseline = json.loads(path.read_text(encoding="utf-8"))
    deltas: dict[str, Any] = {"suite": {}, "cases": {}}
    issues: list[str] = []

    baseline_metrics = baseline.get("metrics", {})
    report_pass_rate = float(report.summary.get("pass_rate", 0.0))
    baseline_pass_rate = float(baseline_metrics.get("task_success", report_pass_rate))
    pass_delta = report_pass_rate - baseline_pass_rate
    deltas["suite"]["pass_rate"] = pass_delta
    if pass_delta < -epsilon:
        issues.append(
            f"suite pass_rate regressed: {report_pass_rate:.3f} < baseline {baseline_pass_rate:.3f}"
        )

    baseline_cases = baseline.get("cases", {})
    report_by_id = {case.id: case for case in report.cases}
    for case_id, case_baseline in baseline_cases.items():
        report_case = report_by_id.get(case_id)
        if report_case is None:
            issues.append(f"missing case in report: {case_id}")
            continue
        expected_success = float(case_baseline.get("metrics", {}).get("task_success", 1.0))
        actual_success = 1.0 if report_case.passed else 0.0
        case_delta = actual_success - expected_success
        deltas["cases"][case_id] = {"task_success": case_delta}
        if case_delta < -epsilon:
            issues.append(f"case {case_id} regressed: success {actual_success} < baseline {expected_success}")

    result = {
        "matched": not issues,
        "issues": issues,
        "deltas": deltas,
    }
    if issues and fail_on_regression:
        raise BaselineMismatchError("; ".join(issues))
    return result
