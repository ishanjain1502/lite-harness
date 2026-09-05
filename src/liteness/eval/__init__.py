"""Offline evaluation of recorded agent sessions."""

from liteness.eval.baseline import compare_baseline
from liteness.eval.dataset import load_suite
from liteness.eval.errors import BaselineMismatchError, EvalCaseError, EvalError, EvalLoadError
from liteness.eval.evaluators import BUILTIN_EVALUATORS
from liteness.eval.models import (
    EvalCase,
    EvalReport,
    EvalRun,
    EvaluationResult,
)
from liteness.eval.runner import (
    EvalRunner,
    build_eval_run,
    build_eval_run_from_session,
    run_session_file,
    run_suite_file,
    run_suite_live,
)
from liteness.eval.live import LiveEvalConfig
from liteness.eval.report import format_eval_report

__all__ = [
    "BUILTIN_EVALUATORS",
    "BaselineMismatchError",
    "EvalCase",
    "EvalCaseError",
    "EvalError",
    "EvalLoadError",
    "EvalReport",
    "EvalRun",
    "EvalRunner",
    "EvaluationResult",
    "build_eval_run",
    "build_eval_run_from_session",
    "compare_baseline",
    "format_eval_report",
    "load_report",
    "load_suite",
    "LiveEvalConfig",
    "run_session_file",
    "run_suite_file",
    "run_suite_live",
]
