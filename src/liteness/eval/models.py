"""Eval data models.

Evaluators consume SessionEvents via EvalRun; they never emit session events.
Telemetry records what happened; eval judges whether it met expectations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from liteness.eval.errors import EvalCaseError, EvalLoadError
from liteness.session import Message, Session
from liteness.telemetry.projector import TraceReport
from liteness.types import StopReason


class EvalRunStatus(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    CORRUPTED = "corrupted"
    UNKNOWN = "unknown"


@dataclass
class ExpectedBehavior:
    required_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    answer: str | None = None
    answer_contains: str | None = None
    allow_tool_errors: bool = False
    max_steps: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ExpectedBehavior:
        if not data:
            return cls()
        return cls(
            required_tools=list(data.get("required_tools") or []),
            forbidden_tools=list(data.get("forbidden_tools") or []),
            stop_reason=data.get("stop_reason"),
            answer=data.get("answer"),
            answer_contains=data.get("answer_contains"),
            allow_tool_errors=bool(data.get("allow_tool_errors", False)),
            max_steps=data.get("max_steps"),
        )


@dataclass
class EvalCase:
    id: str
    input: str | None = None
    session_file: str | None = None
    expected: ExpectedBehavior = field(default_factory=ExpectedBehavior)
    metadata: dict[str, Any] = field(default_factory=dict)
    evaluators: list[str] | None = None
    skip_evaluators: list[str] = field(default_factory=list)
    evaluator_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalCase:
        if "id" not in data:
            raise EvalCaseError("eval case missing required field: id")
        has_input = bool(data.get("input"))
        has_session = bool(data.get("session_file"))
        has_expected = bool(data.get("expected"))
        if not has_input and not has_session and not has_expected:
            raise EvalCaseError(
                f"eval case {data['id']!r} requires input, session_file, or expected"
            )
        return cls(
            id=data["id"],
            input=data.get("input"),
            session_file=data.get("session_file"),
            expected=ExpectedBehavior.from_dict(data.get("expected")),
            metadata=dict(data.get("metadata") or {}),
            evaluators=data.get("evaluators"),
            skip_evaluators=list(data.get("skip_evaluators") or []),
            evaluator_config=dict(data.get("evaluator_config") or {}),
        )


@dataclass
class EvalSuite:
    name: str
    cases: list[EvalCase]
    evaluators: list[str] = field(default_factory=list)
    preset: str | None = None
    evaluator_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base_dir: Path | None = None) -> EvalSuite:
        if "name" not in data:
            raise EvalLoadError("eval suite missing required field: name")
        cases_raw = data.get("cases")
        if not cases_raw:
            raise EvalLoadError("eval suite missing required field: cases")
        cases = [EvalCase.from_dict(case) for case in cases_raw]
        if base_dir is not None:
            for case in cases:
                if case.session_file and not Path(case.session_file).is_absolute():
                    case.session_file = str((base_dir / case.session_file).resolve())
        return cls(
            name=data["name"],
            cases=cases,
            evaluators=list(data.get("evaluators") or []),
            preset=data.get("preset"),
            evaluator_config=dict(data.get("evaluator_config") or {}),
        )


@dataclass
class EvalRun:
    case_id: str
    session_path: Path
    session: Session
    status: EvalRunStatus
    trace: TraceReport
    messages: list[Message]
    stop_reason: StopReason | None


@dataclass
class EvaluationResult:
    evaluator: str
    evaluator_version: str
    case_id: str
    passed: bool | None
    score: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator": self.evaluator,
            "evaluator_version": self.evaluator_version,
            "case_id": self.case_id,
            "passed": self.passed,
            "score": self.score,
            "reason": self.reason,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationResult:
        return cls(
            evaluator=data["evaluator"],
            evaluator_version=data["evaluator_version"],
            case_id=data["case_id"],
            passed=data["passed"],
            score=float(data["score"]),
            reason=data["reason"],
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class EvalCaseResult:
    id: str
    passed: bool
    session_path: str
    status: str
    results: list[EvaluationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "passed": self.passed,
            "session_path": self.session_path,
            "status": self.status,
            "results": [result.to_dict() for result in self.results],
        }


@dataclass
class EvalFailure:
    id: str
    reason: str
    session_path: str
    evaluator: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "reason": self.reason,
            "session_path": self.session_path,
            "evaluator": self.evaluator,
        }


@dataclass
class EvalReport:
    schema_version: int
    suite: str
    timestamp: str
    summary: dict[str, Any]
    evaluators_summary: dict[str, float]
    cases: list[EvalCaseResult]
    failures: list[EvalFailure]
    dataset: str | None = None
    git_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "suite": self.suite,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "evaluators_summary": self.evaluators_summary,
            "cases": [case.to_dict() for case in self.cases],
            "failures": [failure.to_dict() for failure in self.failures],
        }
        if self.dataset is not None:
            payload["dataset"] = self.dataset
        if self.git_commit is not None:
            payload["git_commit"] = self.git_commit
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalReport:
        return cls(
            schema_version=int(data["schema_version"]),
            suite=data["suite"],
            timestamp=data["timestamp"],
            summary=dict(data["summary"]),
            evaluators_summary={
                key: float(value) for key, value in data.get("evaluators_summary", {}).items()
            },
            cases=[
                EvalCaseResult(
                    id=case["id"],
                    passed=bool(case["passed"]),
                    session_path=case["session_path"],
                    status=case["status"],
                    results=[EvaluationResult.from_dict(result) for result in case["results"]],
                )
                for case in data["cases"]
            ],
            failures=[
                EvalFailure(
                    id=failure["id"],
                    reason=failure["reason"],
                    session_path=failure["session_path"],
                    evaluator=failure["evaluator"],
                )
                for failure in data.get("failures", [])
            ],
            dataset=data.get("dataset"),
            git_commit=data.get("git_commit"),
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
