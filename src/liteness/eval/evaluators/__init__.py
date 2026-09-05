"""Evaluator base class and registry."""

from __future__ import annotations

from liteness.eval.evaluators.base import Evaluator
from liteness.eval.evaluators.execution import (
    BudgetEnforcementEvaluator,
    CancellationEvaluator,
    RetryBehaviorEvaluator,
    TimeoutBehaviorEvaluator,
)
from liteness.eval.evaluators.stop_reason import StopReasonEvaluator
from liteness.eval.evaluators.tools import EventIntegrityEvaluator, FinalAnswerEvaluator, ToolLifecycleEvaluator

BUILTIN_EVALUATORS: dict[str, type[Evaluator]] = {
    "stop_reason": StopReasonEvaluator,
    "tool_lifecycle": ToolLifecycleEvaluator,
    "event_integrity": EventIntegrityEvaluator,
    "final_answer": FinalAnswerEvaluator,
    "retry_behavior": RetryBehaviorEvaluator,
    "timeout_behavior": TimeoutBehaviorEvaluator,
    "cancellation": CancellationEvaluator,
    "budget_enforcement": BudgetEnforcementEvaluator,
}

__all__ = ["BUILTIN_EVALUATORS", "Evaluator"]
