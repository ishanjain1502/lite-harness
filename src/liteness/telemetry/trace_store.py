"""Derived span model — built from session events, not persisted to JSONL."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

SpanKind = Literal["turn", "llm", "tool", "retry-attempt"]
SpanStatus = Literal["success", "error", "cancelled", "timeout", "in_progress"]


@dataclass
class Span:
    span_id: str
    parent_span_id: str | None
    kind: SpanKind
    name: str
    session_id: str
    turn: int
    step: int = 0
    call_id: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_ms: float | None = None
    status: SpanStatus = "in_progress"
    error_code: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Trace:
    trace_id: str
    session_id: str
    spans: list[Span] = field(default_factory=list)
    outcome: str | None = None


class TraceStore:
    """In-memory store of derived spans for one or more traces."""

    def __init__(self) -> None:
        self._traces: dict[str, Trace] = {}

    def get_or_create(self, session_id: str) -> Trace:
        if session_id not in self._traces:
            self._traces[session_id] = Trace(trace_id=session_id, session_id=session_id)
        return self._traces[session_id]

    def add_span(self, span: Span) -> None:
        trace = self.get_or_create(span.session_id)
        trace.spans.append(span)

    def replace_spans(self, session_id: str, spans: list[Span], *, outcome: str | None = None) -> None:
        trace = self.get_or_create(session_id)
        trace.spans = spans
        if outcome is not None:
            trace.outcome = outcome

    def get_trace(self, session_id: str) -> Trace | None:
        return self._traces.get(session_id)

    def all_traces(self) -> list[Trace]:
        return list(self._traces.values())

    def clear(self) -> None:
        self._traces.clear()
