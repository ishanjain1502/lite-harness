"""Harness execution evaluators."""

from __future__ import annotations

from liteness.eval.evaluators.base import Evaluator
from liteness.eval.models import EvalCase, EvalRun, EvaluationResult


class RetryBehaviorEvaluator(Evaluator):
    name = "retry_behavior"
    version = "1.0.0"

    def evaluate(self, case: EvalCase, run: EvalRun) -> EvaluationResult:
        retries = [event for event in run.session.events if event.type in ("llm/retry", "tool/retry")]
        return EvaluationResult(
            evaluator=self.name,
            evaluator_version=self.version,
            case_id=case.id,
            passed=True,
            score=1.0,
            reason=f"recorded {len(retries)} retry event(s)",
            metadata={"evidence": {"retry_count": len(retries), "retry_events": len(retries)}},
        )


class TimeoutBehaviorEvaluator(Evaluator):
    name = "timeout_behavior"
    version = "1.0.0"

    def evaluate(self, case: EvalCase, run: EvalRun) -> EvaluationResult:
        timeout_events = [
            event
            for event in run.session.events
            if event.type == "tool/result"
            and event.payload.get("error_code") in ("TOOL_TIMEOUT", "TIMEOUT")
        ]
        if timeout_events:
            return EvaluationResult(
                evaluator=self.name,
                evaluator_version=self.version,
                case_id=case.id,
                passed=True,
                score=1.0,
                reason=f"recorded {len(timeout_events)} timeout tool result(s)",
                metadata={"evidence": {"timeout_count": len(timeout_events)}},
            )
        return EvaluationResult(
            evaluator=self.name,
            evaluator_version=self.version,
            case_id=case.id,
            passed=None,
            score=0.0,
            reason="no timeout events recorded",
            metadata={"skipped": True},
        )


class CancellationEvaluator(Evaluator):
    name = "cancellation"
    version = "1.0.0"

    def evaluate(self, case: EvalCase, run: EvalRun) -> EvaluationResult:
        cancel_events = [event for event in run.session.events if event.type == "cancel/requested"]
        if cancel_events:
            return EvaluationResult(
                evaluator=self.name,
                evaluator_version=self.version,
                case_id=case.id,
                passed=True,
                score=1.0,
                reason=f"recorded {len(cancel_events)} cancellation event(s)",
                metadata={"evidence": {"cancel_count": len(cancel_events)}},
            )
        return EvaluationResult(
            evaluator=self.name,
            evaluator_version=self.version,
            case_id=case.id,
            passed=None,
            score=0.0,
            reason="no cancellation events recorded",
            metadata={"skipped": True},
        )


class BudgetEnforcementEvaluator(Evaluator):
    name = "budget_enforcement"
    version = "1.0.0"

    def evaluate(self, case: EvalCase, run: EvalRun) -> EvaluationResult:
        step_events = [event for event in run.session.events if event.type == "step/end"]
        step_count = len(step_events)
        max_steps = case.expected.max_steps
        if max_steps is None:
            return EvaluationResult(
                evaluator=self.name,
                evaluator_version=self.version,
                case_id=case.id,
                passed=None,
                score=0.0,
                reason="no expected.max_steps configured",
                metadata={"skipped": True, "evidence": {"steps": step_count}},
            )

        passed = step_count <= max_steps
        return EvaluationResult(
            evaluator=self.name,
            evaluator_version=self.version,
            case_id=case.id,
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=(
                f"steps {step_count} within max {max_steps}"
                if passed
                else f"steps {step_count} exceeds max {max_steps}"
            ),
            metadata={"evidence": {"steps": step_count, "max_steps": max_steps}},
        )
