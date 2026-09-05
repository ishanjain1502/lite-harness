"""Evaluator protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod

from liteness.eval.models import EvalCase, EvalRun, EvaluationResult


class Evaluator(ABC):
    name: str
    version: str = "1.0.0"

    @abstractmethod
    def evaluate(self, case: EvalCase, run: EvalRun) -> EvaluationResult:
        """Evaluate one recorded run against case expectations."""
