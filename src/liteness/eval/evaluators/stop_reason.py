"""Stop reason evaluator."""

from __future__ import annotations

from liteness.eval.evaluators.base import Evaluator
from liteness.eval.models import EvalCase, EvalRun, EvaluationResult


class StopReasonEvaluator(Evaluator):
    name = "stop_reason"
    version = "1.0.0"

    def evaluate(self, case: EvalCase, run: EvalRun) -> EvaluationResult:
        expected = case.expected.stop_reason
        if expected is None:
            return EvaluationResult(
                evaluator=self.name,
                evaluator_version=self.version,
                case_id=case.id,
                passed=None,
                score=0.0,
                reason="no expected.stop_reason configured",
                metadata={"skipped": True},
            )

        actual = run.stop_reason.value if run.stop_reason is not None else None
        passed = actual == expected
        return EvaluationResult(
            evaluator=self.name,
            evaluator_version=self.version,
            case_id=case.id,
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=(
                f"stop_reason matches {expected!r}"
                if passed
                else f"expected stop_reason {expected}, got {actual}"
            ),
            metadata={
                "evidence": {"expected": expected, "actual": actual},
            },
        )
