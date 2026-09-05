"""Tool and outcome evaluators."""

from __future__ import annotations

import re

from liteness.eval.evaluators.base import Evaluator
from liteness.eval.models import EvalCase, EvalRun, EvaluationResult
from liteness.session import find_orphan_tool_calls


class ToolLifecycleEvaluator(Evaluator):
    name = "tool_lifecycle"
    version = "1.0.0"

    def evaluate(self, case: EvalCase, run: EvalRun) -> EvaluationResult:
        tool_names = [
            event.payload["name"]
            for event in run.session.events
            if event.type == "tool/call"
        ]
        config = {**case.evaluator_config}
        enforce_order = bool(config.get("enforce_order", False))

        required = case.expected.required_tools
        forbidden = case.expected.forbidden_tools
        evidence: dict[str, object] = {
            "tools_used": tool_names,
            "required_tools": required,
            "forbidden_tools": forbidden,
        }

        if enforce_order and required:
            index = 0
            for name in tool_names:
                if index < len(required) and name == required[index]:
                    index += 1
            if index != len(required):
                return EvaluationResult(
                    evaluator=self.name,
                    evaluator_version=self.version,
                    case_id=case.id,
                    passed=False,
                    score=0.0,
                    reason=f"required tool order not satisfied: {required}",
                    metadata={"evidence": evidence, "code": "WRONG_TOOL_ORDER"},
                )

        used_set = set(tool_names)
        for name in required:
            if name not in used_set:
                return EvaluationResult(
                    evaluator=self.name,
                    evaluator_version=self.version,
                    case_id=case.id,
                    passed=False,
                    score=0.0,
                    reason=f"missing required tool: {name}",
                    metadata={"evidence": evidence, "code": "MISSING_TOOL"},
                )

        for name in forbidden:
            if name in used_set:
                return EvaluationResult(
                    evaluator=self.name,
                    evaluator_version=self.version,
                    case_id=case.id,
                    passed=False,
                    score=0.0,
                    reason=f"forbidden tool used: {name}",
                    metadata={"evidence": evidence, "code": "FORBIDDEN_TOOL"},
                )

        if not case.expected.allow_tool_errors:
            for event in run.session.events:
                if event.type != "tool/result":
                    continue
                if event.payload.get("is_error"):
                    code = event.payload.get("error_code", "UNKNOWN")
                    return EvaluationResult(
                        evaluator=self.name,
                        evaluator_version=self.version,
                        case_id=case.id,
                        passed=False,
                        score=0.0,
                        reason=(
                            f"tool error at turn {event.turn} step {event.step}: {code}"
                        ),
                        metadata={
                            "evidence": {
                                "call_id": event.payload.get("call_id"),
                                "name": event.payload.get("name"),
                                "error_code": code,
                            },
                            "code": "TOOL_ERROR",
                        },
                    )

        if not required and not forbidden and case.expected.allow_tool_errors:
            return EvaluationResult(
                evaluator=self.name,
                evaluator_version=self.version,
                case_id=case.id,
                passed=None,
                score=0.0,
                reason="no tool expectations configured",
                metadata={"skipped": True},
            )

        return EvaluationResult(
            evaluator=self.name,
            evaluator_version=self.version,
            case_id=case.id,
            passed=True,
            score=1.0,
            reason="tool lifecycle expectations satisfied",
            metadata={"evidence": evidence},
        )


class EventIntegrityEvaluator(Evaluator):
    name = "event_integrity"
    version = "1.0.0"

    def evaluate(self, case: EvalCase, run: EvalRun) -> EvaluationResult:
        orphans = find_orphan_tool_calls(run.session.events)
        evidence = [
            {
                "call_id": orphan.call_id,
                "name": orphan.name,
                "turn": orphan.turn,
                "step": orphan.step,
            }
            for orphan in orphans
        ]
        incomplete = run.status.value != "complete"
        issues: list[str] = []
        if orphans:
            issues.append(f"{len(orphans)} orphan tool call(s)")
        if incomplete:
            issues.append(f"session status is {run.status.value}")

        if issues:
            return EvaluationResult(
                evaluator=self.name,
                evaluator_version=self.version,
                case_id=case.id,
                passed=False if incomplete else None,
                score=0.0 if incomplete else 0.0,
                reason="; ".join(issues),
                metadata={"evidence": evidence, "orphan_count": len(orphans)},
            )

        return EvaluationResult(
            evaluator=self.name,
            evaluator_version=self.version,
            case_id=case.id,
            passed=True,
            score=1.0,
            reason="session event integrity ok",
            metadata={"evidence": evidence},
        )


class FinalAnswerEvaluator(Evaluator):
    name = "final_answer"
    version = "1.0.0"

    def evaluate(self, case: EvalCase, run: EvalRun) -> EvaluationResult:
        expected_exact = case.expected.answer
        expected_contains = case.expected.answer_contains
        if expected_exact is None and expected_contains is None:
            return EvaluationResult(
                evaluator=self.name,
                evaluator_version=self.version,
                case_id=case.id,
                passed=None,
                score=0.0,
                reason="no expected answer configured",
                metadata={"skipped": True},
            )

        if run.stop_reason is None or run.status.value != "complete":
            return EvaluationResult(
                evaluator=self.name,
                evaluator_version=self.version,
                case_id=case.id,
                passed=False,
                score=0.0,
                reason="cannot evaluate final answer on incomplete session",
                metadata={"code": "INCOMPLETE_SESSION"},
            )

        actual = _final_assistant_text(run)
        if expected_exact is not None:
            normalized_expected = " ".join(expected_exact.split())
            normalized_actual = " ".join(actual.split())
            passed = normalized_actual == normalized_expected
            return EvaluationResult(
                evaluator=self.name,
                evaluator_version=self.version,
                case_id=case.id,
                passed=passed,
                score=1.0 if passed else 0.0,
                reason=(
                    "final answer matches"
                    if passed
                    else f"expected answer {expected_exact!r}, got {actual!r}"
                ),
                metadata={"evidence": {"actual": actual, "expected": expected_exact}},
            )

        assert expected_contains is not None
        passed = expected_contains in actual
        return EvaluationResult(
            evaluator=self.name,
            evaluator_version=self.version,
            case_id=case.id,
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=(
                "final answer contains expected substring"
                if passed
                else f"expected answer to contain {expected_contains!r}, got {actual!r}"
            ),
            metadata={"evidence": {"actual": actual, "expected_contains": expected_contains}},
        )


def _final_assistant_text(run: EvalRun) -> str:
    last_turn = 0
    for event in run.session.events:
        if event.type in ("turn/start", "turn/end"):
            last_turn = max(last_turn, event.payload.get("turn", event.turn))

    final_content = ""
    for event in reversed(run.session.events):
        if event.type != "assistant/message" or event.turn != last_turn:
            continue
        step = event.step
        has_tool_calls_in_step = any(
            other.type == "tool/call" and other.turn == event.turn and other.step == step
            for other in run.session.events
        )
        if has_tool_calls_in_step:
            continue
        final_content = event.payload.get("content", "")
        break
    return final_content
