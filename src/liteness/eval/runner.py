"""Eval runner — load sessions and invoke evaluators."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any

from liteness.eval.dataset import load_suite
from liteness.eval.errors import EvalCaseError, EvalLoadError
from liteness.eval.evaluators import BUILTIN_EVALUATORS
from liteness.eval.evaluators.base import Evaluator
from liteness.eval.live import LiveEvalConfig, execute_live_case
from liteness.eval.models import (
    EvalCase,
    EvalCaseResult,
    EvalReport,
    EvalRun,
    EvalRunStatus,
    EvalSuite,
    EvaluationResult,
)
from liteness.eval.report import aggregate_case_result, build_eval_report
from liteness.session import Session, find_orphan_tool_calls
from liteness.telemetry.projector import project_events
from liteness.types import StopReason

logger = logging.getLogger("eval.runner")


def build_eval_run_from_session(
    case: EvalCase,
    session: Session,
    *,
    session_path: Path | str | None = None,
    recover: bool = False,
) -> EvalRun:
    if recover:
        from liteness.session import recover_orphans

        recover_orphans(session)

    path = (
        Path(session_path)
        if session_path is not None
        else (session.log_path or Path(f"<memory:{session.session_id}>"))
    )
    status = _derive_run_status(session)
    trace = project_events(session.events)
    messages = session.derive_messages()
    stop_reason = _read_stop_reason(session)

    return EvalRun(
        case_id=case.id,
        session_path=path,
        session=session,
        status=status,
        trace=trace,
        messages=messages,
        stop_reason=stop_reason,
    )


def build_eval_run(
    case: EvalCase,
    session_path: Path | str,
    *,
    recover: bool = False,
) -> EvalRun:
    path = Path(session_path)
    try:
        session = Session.load_from_jsonl(path, recover=recover)
    except ValueError as exc:
        raise EvalLoadError(str(exc)) from exc

    return build_eval_run_from_session(case, session, session_path=path)


def _derive_run_status(session: Session) -> EvalRunStatus:
    if not session.events:
        return EvalRunStatus.UNKNOWN

    turns_started = {
        event.payload.get("turn", event.turn)
        for event in session.events
        if event.type == "turn/start"
    }
    turns_ended = {
        event.payload.get("turn", event.turn)
        for event in session.events
        if event.type == "turn/end"
    }
    if not turns_started:
        return EvalRunStatus.UNKNOWN
    if turns_started != turns_ended:
        return EvalRunStatus.INCOMPLETE
    if find_orphan_tool_calls(session.events):
        return EvalRunStatus.INCOMPLETE
    return EvalRunStatus.COMPLETE


def _read_stop_reason(session: Session) -> StopReason | None:
    for event in reversed(session.events):
        if event.type != "turn/end":
            continue
        raw = event.payload.get("stop_reason")
        if raw is None:
            return None
        try:
            return StopReason(raw)
        except ValueError:
            return None
    return None


class EvalRunner:
    def __init__(
        self,
        *,
        recover: bool = False,
        fail_fast: bool = False,
        live_config: LiveEvalConfig | None = None,
    ) -> None:
        self.recover = recover
        self.fail_fast = fail_fast
        self.live_config = live_config

    def run(
        self,
        suite: EvalSuite,
        *,
        case_id: str | None = None,
        session_overrides: dict[str, Path | str] | None = None,
        live: bool = False,
    ) -> EvalReport:
        cases = suite.cases
        if case_id is not None:
            cases = [case for case in cases if case.id == case_id]
            if not cases:
                raise EvalCaseError(f"case not found in suite: {case_id}")

        overrides = dict(session_overrides or {})
        if live:
            if self.live_config is None:
                raise EvalCaseError("live eval requires LiveEvalConfig")
            for case in cases:
                if case.id in overrides:
                    continue
                if case.input:
                    overrides[case.id] = execute_live_case(case, suite, self.live_config)
                elif case.session_file:
                    overrides[case.id] = case.session_file
                else:
                    raise EvalCaseError(
                        f"case {case.id!r} requires input (live) or session_file (offline)"
                    )

        case_results: list[EvalCaseResult] = []
        for case in cases:
            logger.info("evaluating case %s", case.id)
            session_path = _resolve_session_path(case, overrides)
            run = build_eval_run(case, session_path, recover=self.recover)
            evaluators = _resolve_evaluators(suite, case)
            results = [
                _safe_evaluate(evaluator, case, run, fail_fast=self.fail_fast)
                for evaluator in evaluators
            ]
            case_results.append(aggregate_case_result(case, run, results))

        return build_eval_report(suite, case_results, dataset=suite.name)

    def evaluate_session(
        self,
        suite: EvalSuite,
        case: EvalCase,
        session: Session,
        *,
        session_path: Path | str | None = None,
    ) -> EvalCaseResult:
        run = build_eval_run_from_session(
            case,
            session,
            session_path=session_path,
            recover=self.recover,
        )
        evaluators = _resolve_evaluators(suite, case)
        results = [
            _safe_evaluate(evaluator, case, run, fail_fast=self.fail_fast)
            for evaluator in evaluators
        ]
        return aggregate_case_result(case, run, results)


def _resolve_session_path(
    case: EvalCase,
    session_overrides: dict[str, Path | str] | None,
) -> Path:
    if session_overrides and case.id in session_overrides:
        return Path(session_overrides[case.id])
    if case.session_file:
        return Path(case.session_file)
    raise EvalCaseError(f"case {case.id!r} has no session_file for offline eval")


def _resolve_evaluators(suite: EvalSuite, case: EvalCase) -> list[Evaluator]:
    if case.evaluators is not None:
        names = list(case.evaluators)
    else:
        names = list(suite.evaluators)
    if not names:
        names = ["stop_reason", "tool_lifecycle", "event_integrity"]

    skip = set(case.skip_evaluators)
    config = {**suite.evaluator_config, **case.evaluator_config}
    evaluators: list[Evaluator] = []
    for name in names:
        if name in skip:
            continue
        cls = BUILTIN_EVALUATORS.get(name)
        if cls is None:
            raise EvalCaseError(f"unknown evaluator: {name}")
        evaluator_config = config.get(name, {})
        if isinstance(evaluator_config, dict) and evaluator_config:
            evaluators.append(cls(**evaluator_config))  # type: ignore[misc]
        else:
            evaluators.append(cls())
    return evaluators


def _safe_evaluate(
    evaluator: Evaluator,
    case: EvalCase,
    run: EvalRun,
    *,
    fail_fast: bool,
) -> EvaluationResult:
    try:
        return evaluator.evaluate(case, run)
    except Exception as exc:
        if fail_fast:
            raise
        return EvaluationResult(
            evaluator=evaluator.name,
            evaluator_version=getattr(evaluator, "version", "unknown"),
            case_id=case.id,
            passed=False,
            score=0.0,
            reason=f"evaluator error: {exc}",
            metadata={
                "error": str(exc),
                "traceback": traceback.format_exc(limit=5),
            },
        )


def run_suite_file(
    eval_file: Path | str,
    *,
    case_id: str | None = None,
    session_file: Path | str | None = None,
    recover: bool = False,
    fail_fast: bool = False,
    live_config: LiveEvalConfig | None = None,
) -> EvalReport:
    suite = load_suite(eval_file)
    overrides: dict[str, Path | str] | None = None
    if session_file is not None and case_id is not None:
        overrides = {case_id: session_file}
    runner = EvalRunner(recover=recover, fail_fast=fail_fast, live_config=live_config)
    return runner.run(
        suite,
        case_id=case_id,
        session_overrides=overrides,
        live=live_config is not None,
    )


def run_suite_live(
    eval_file: Path | str,
    *,
    case_id: str | None = None,
    live_config: LiveEvalConfig | None = None,
    recover: bool = False,
    fail_fast: bool = False,
) -> EvalReport:
    """Run live agent execution for input cases, then evaluate recorded sessions."""
    config = live_config or LiveEvalConfig()
    return run_suite_file(
        eval_file,
        case_id=case_id,
        recover=recover,
        fail_fast=fail_fast,
        live_config=config,
    )


def run_session_file(
    session_file: Path | str,
    *,
    eval_file: Path | str | None = None,
    case_id: str | None = None,
    recover: bool = False,
    fail_fast: bool = False,
) -> EvalReport:
    if eval_file is None:
        if case_id is None:
            case_id = Path(session_file).stem
        suite = EvalSuite(
            name=Path(session_file).stem,
            cases=[EvalCase(id=case_id, session_file=str(session_file))],
            evaluators=["stop_reason", "tool_lifecycle", "event_integrity"],
        )
        runner = EvalRunner(recover=recover, fail_fast=fail_fast)
        return runner.run(suite, case_id=case_id)

    suite = load_suite(eval_file)
    target_case_id = case_id
    if target_case_id is None:
        if len(suite.cases) == 1:
            target_case_id = suite.cases[0].id
        else:
            raise EvalCaseError("--case is required when suite has multiple cases")
    overrides = {target_case_id: session_file}
    runner = EvalRunner(recover=recover, fail_fast=fail_fast)
    return runner.run(suite, case_id=target_case_id, session_overrides=overrides)
