"""Eval report formatting and aggregation."""

from __future__ import annotations

import subprocess
from collections import defaultdict

from liteness.eval.models import (
    EvalCase,
    EvalCaseResult,
    EvalFailure,
    EvalReport,
    EvalRun,
    EvalSuite,
    EvaluationResult,
    utc_now_iso,
)


def build_eval_report(
    suite: EvalSuite,
    case_results: list[EvalCaseResult],
    *,
    dataset: str | None = None,
) -> EvalReport:
    total = len(case_results)
    passed_cases = sum(1 for case in case_results if case.passed)
    failed_cases = sum(
        1
        for case in case_results
        if not case.passed and case.status != "skipped"
    )
    skipped_cases = sum(1 for case in case_results if case.status == "skipped")
    eligible = sum(1 for case in case_results if case.status != "skipped")
    pass_rate = (passed_cases / eligible) if eligible else 0.0

    evaluator_scores: dict[str, list[float]] = defaultdict(list)
    for case in case_results:
        for result in case.results:
            if result.passed is None:
                continue
            evaluator_scores[result.evaluator].append(result.score)

    evaluators_summary = {
        name: (sum(scores) / len(scores) if scores else 0.0)
        for name, scores in evaluator_scores.items()
    }

    failures: list[EvalFailure] = []
    for case in case_results:
        if case.passed:
            continue
        failed = next((result for result in case.results if result.passed is False), None)
        if failed is None:
            continue
        failures.append(
            EvalFailure(
                id=case.id,
                reason=failed.reason,
                session_path=case.session_path,
                evaluator=failed.evaluator,
            )
        )

    return EvalReport(
        schema_version=1,
        suite=suite.name,
        timestamp=utc_now_iso(),
        summary={
            "total": total,
            "passed": passed_cases,
            "failed": failed_cases,
            "skipped": skipped_cases,
            "pass_rate": pass_rate,
        },
        evaluators_summary=evaluators_summary,
        cases=case_results,
        failures=failures,
        dataset=dataset or suite.name,
        git_commit=_git_commit(),
    )


def aggregate_case_result(
    case: EvalCase,
    run: EvalRun,
    results: list[EvaluationResult],
) -> EvalCaseResult:
    ran = [result for result in results if result.passed is not None]
    skipped_all = len(ran) == 0 and len(results) > 0
    passed = len(ran) > 0 and all(result.passed for result in ran)
    status = run.status.value
    if skipped_all:
        status = "skipped"
    return EvalCaseResult(
        id=case.id,
        passed=passed,
        session_path=str(run.session_path),
        status=status,
        results=results,
    )


def format_eval_report(report: EvalReport, *, verbose: bool = False) -> str:
    summary = report.summary
    lines = [
        f"Eval Report: {report.suite}",
        f"Dataset: {report.dataset or report.suite}",
        "",
        "Summary",
        f"  Total:   {summary['total']}",
        f"  Passed:  {summary['passed']}",
        f"  Failed:  {summary['failed']}",
        f"  Skipped: {summary['skipped']}",
        f"  Pass rate: {summary['pass_rate']:.0%}",
        "",
    ]

    if report.evaluators_summary:
        lines.append("Evaluators")
        for name, rate in sorted(report.evaluators_summary.items()):
            lines.append(f"  {name}: {rate:.0%}")
        lines.append("")

    if report.failures:
        lines.append("Failures")
        for failure in report.failures:
            lines.append(f"  {failure.id}: {failure.reason} ({failure.evaluator})")
        lines.append("")

    if verbose:
        lines.append("Cases")
        for case in report.cases:
            status = "PASS" if case.passed else "FAIL"
            lines.append(f"  [{status}] {case.id} ({case.status})")
            for result in case.results:
                if result.passed is None:
                    label = "SKIP"
                else:
                    label = "PASS" if result.passed else "FAIL"
                lines.append(f"    - {result.evaluator}: {label} — {result.reason}")

    return "\n".join(lines).rstrip() + "\n"


def _git_commit() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return output or None
