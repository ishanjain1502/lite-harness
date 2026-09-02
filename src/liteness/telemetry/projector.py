"""Project session events into traces, metrics, and human-readable reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from liteness.session import SessionEvent
from liteness.telemetry.metrics_store import MetricsSnapshot, MetricsStore
from liteness.telemetry.trace_store import Span, Trace, TraceStore


class Recovery(str, Enum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_RETRIED = "NOT_RETRIED"
    RETRIED_N = "RETRIED_N"
    ORPHAN_SYNTHESIZED = "ORPHAN_SYNTHESIZED"
    AGENT_RECOVERED = "AGENT_RECOVERED"


def _ms_between(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start).total_seconds() * 1000.0


def _span_status_from_tool(payload: dict[str, Any]) -> str:
    if payload.get("is_error"):
        code = payload.get("error_code", "")
        if code == "TOOL_TIMEOUT":
            return "timeout"
        if code == "CANCELLED":
            return "cancelled"
        return "error"
    return "success"


@dataclass
class TraceReport:
    trace: Trace
    metrics: MetricsSnapshot
    recovery: Recovery = Recovery.NOT_APPLICABLE
    retry_count: int = 0
    notes: list[str] = field(default_factory=list)


class EventProjector:
    """Incremental projector: feed SessionEvent objects, build trace + metrics."""

    def __init__(self) -> None:
        self.trace_store = TraceStore()
        self.metrics_store = MetricsStore()
        self._open_turns: dict[tuple[str, int], datetime] = {}
        self._open_steps: dict[tuple[str, int, int], datetime] = {}
        self._open_tools: dict[str, datetime] = {}
        self._llm_retries: dict[tuple[str, int, int], int] = {}
        self._tool_retries: dict[str, int] = {}
        self._session_id: str | None = None

    def process(self, event: SessionEvent) -> None:
        self._session_id = event.session_id
        handler = _HANDLERS.get(event.type)
        if handler is not None:
            handler(self, event)

    def build_report(self, session_id: str | None = None) -> TraceReport:
        sid = session_id or self._session_id
        if sid is None:
            empty = Trace(trace_id="", session_id="")
            return TraceReport(trace=empty, metrics=self.metrics_store.snapshot)

        trace = self.trace_store.get_trace(sid)
        if trace is None:
            trace = Trace(trace_id=sid, session_id=sid)

        recovery = Recovery.NOT_APPLICABLE
        retry_count = sum(self._llm_retries.values()) + sum(self._tool_retries.values())
        notes: list[str] = []
        if any(n for n in notes if "INV-18" in n):
            pass
        for span in trace.spans:
            if span.status == "timeout":
                notes.append(
                    "INV-18: TOOL_TIMEOUT means the harness stopped waiting; "
                    "the worker may still be running."
                )
                break

        return TraceReport(
            trace=trace,
            metrics=self.metrics_store.snapshot,
            recovery=recovery,
            retry_count=retry_count,
            notes=notes,
        )

    def _on_turn_start(self, event: SessionEvent) -> None:
        key = (event.session_id, event.turn)
        self._open_turns[key] = event.timestamp
        parent = f"turn-{event.turn}"
        self.trace_store.add_span(
            Span(
                span_id=parent,
                parent_span_id=None,
                kind="turn",
                name=f"Turn {event.turn}",
                session_id=event.session_id,
                turn=event.turn,
                started_at=event.timestamp,
            )
        )

    def _on_turn_end(self, event: SessionEvent) -> None:
        key = (event.session_id, event.turn)
        started = self._open_turns.pop(key, None)
        duration = _ms_between(started, event.timestamp)
        status = payload_status(event.payload.get("stop_reason", "completed"))
        trace = self.trace_store.get_or_create(event.session_id)
        for span in trace.spans:
            if span.kind == "turn" and span.turn == event.turn:
                span.ended_at = event.timestamp
                span.duration_ms = duration
                span.status = status
                span.metadata = {
                    "stop_reason": event.payload.get("stop_reason"),
                    "total_tokens": event.payload.get("total_tokens"),
                    "total_cost_usd": event.payload.get("total_cost_usd"),
                }
                break
        success = event.payload.get("status") == "completed"
        self.metrics_store.record_session_outcome(success=success)
        trace.outcome = event.payload.get("stop_reason")

    def _on_step_start(self, event: SessionEvent) -> None:
        key = (event.session_id, event.turn, event.step)
        self._open_steps[key] = event.timestamp

    def _on_llm_response(self, event: SessionEvent) -> None:
        key = (event.session_id, event.turn, event.step)
        started = self._open_steps.get(key)
        duration = _ms_between(started, event.timestamp)
        parent = f"turn-{event.turn}"
        span_id = f"llm-{event.turn}-{event.step}"
        payload = event.payload
        self.trace_store.add_span(
            Span(
                span_id=span_id,
                parent_span_id=parent,
                kind="llm",
                name="llm",
                session_id=event.session_id,
                turn=event.turn,
                step=event.step,
                started_at=started,
                ended_at=event.timestamp,
                duration_ms=duration,
                status="success",
                metadata={
                    "input_tokens": payload.get("input_tokens", 0),
                    "output_tokens": payload.get("output_tokens", 0),
                    "cost_usd": payload.get("cost_usd", 0.0),
                },
            )
        )
        self.metrics_store.record_llm(
            duration_ms=duration or 0.0,
            input_tokens=int(payload.get("input_tokens", 0)),
            output_tokens=int(payload.get("output_tokens", 0)),
            cost_usd=float(payload.get("cost_usd", 0.0)),
        )

    def _on_tool_call(self, event: SessionEvent) -> None:
        call_id = event.payload["call_id"]
        self._open_tools[call_id] = event.timestamp

    def _on_tool_result(self, event: SessionEvent) -> None:
        payload = event.payload
        call_id = payload["call_id"]
        started = self._open_tools.pop(call_id, None)
        duration = _ms_between(started, event.timestamp)
        status = _span_status_from_tool(payload)
        parent = f"turn-{event.turn}"
        self.trace_store.add_span(
            Span(
                span_id=f"tool-{call_id}",
                parent_span_id=parent,
                kind="tool",
                name=payload.get("name", "tool"),
                session_id=event.session_id,
                turn=event.turn,
                step=event.step,
                call_id=call_id,
                started_at=started,
                ended_at=event.timestamp,
                duration_ms=duration,
                status=status,  # type: ignore[arg-type]
                error_code=payload.get("error_code"),
            )
        )
        self.metrics_store.record_tool(
            duration_ms=duration or 0.0,
            failed=bool(payload.get("is_error")),
        )

    def _on_llm_retry(self, event: SessionEvent) -> None:
        key = (event.session_id, event.turn, event.step)
        attempt = event.payload.get("attempt", 1)
        self._llm_retries[key] = self._llm_retries.get(key, 0) + 1
        self.metrics_store.record_llm_retry()
        parent = f"llm-{event.turn}-{event.step}"
        self.trace_store.add_span(
            Span(
                span_id=f"retry-llm-{event.turn}-{event.step}-{attempt}",
                parent_span_id=parent,
                kind="retry-attempt",
                name=f"retry #{attempt}",
                session_id=event.session_id,
                turn=event.turn,
                step=event.step,
                started_at=event.timestamp,
                ended_at=event.timestamp,
                duration_ms=0.0,
                status="error",
                metadata={"reason_code": event.payload.get("reason_code")},
            )
        )

    def _on_tool_retry(self, event: SessionEvent) -> None:
        call_id = event.payload.get("call_id", "unknown")
        attempt = event.payload.get("attempt", 1)
        self._tool_retries[call_id] = self._tool_retries.get(call_id, 0) + 1
        self.metrics_store.record_tool_retry()
        parent = f"tool-{call_id}"
        self.trace_store.add_span(
            Span(
                span_id=f"retry-tool-{call_id}-{attempt}",
                parent_span_id=parent,
                kind="retry-attempt",
                name=f"retry #{attempt}",
                session_id=event.session_id,
                turn=event.turn,
                step=event.step,
                call_id=call_id,
                started_at=event.timestamp,
                ended_at=event.timestamp,
                duration_ms=0.0,
                status="error",
                metadata={"reason_code": event.payload.get("reason_code")},
            )
        )


def payload_status(stop_reason: str) -> str:
    if stop_reason == "completed":
        return "success"
    if stop_reason == "user_cancelled":
        return "cancelled"
    return "error"


_HANDLERS: dict[str, Any] = {
    "turn/start": EventProjector._on_turn_start,
    "turn/end": EventProjector._on_turn_end,
    "step/start": EventProjector._on_step_start,
    "llm/response": EventProjector._on_llm_response,
    "tool/call": EventProjector._on_tool_call,
    "tool/result": EventProjector._on_tool_result,
    "llm/retry": EventProjector._on_llm_retry,
    "tool/retry": EventProjector._on_tool_retry,
}


def project_events(events: list[SessionEvent]) -> TraceReport:
    """Replay a session log through the projector (offline / CLI report)."""
    projector = EventProjector()
    for event in events:
        projector.process(event)
    session_id = events[0].session_id if events else None
    return projector.build_report(session_id)


def format_trace_report(report: TraceReport) -> str:
    """Render a human-readable tree + metrics block."""
    trace = report.trace
    metrics = report.metrics
    lines: list[str] = []
    lines.append(f"Trace: {trace.trace_id or trace.session_id}")
    if trace.outcome:
        lines.append(f"Outcome: {trace.outcome}")
    lines.append("")

    turns = [s for s in trace.spans if s.kind == "turn"]
    for turn_span in sorted(turns, key=lambda s: s.turn):
        dur = f" ({turn_span.duration_ms:.0f}ms)" if turn_span.duration_ms else ""
        lines.append(f"├── {turn_span.name}{dur} [{turn_span.status}]")
        children = [
            s
            for s in trace.spans
            if s.parent_span_id == turn_span.span_id and s.kind != "retry-attempt"
        ]
        for child in sorted(children, key=lambda s: (s.step, s.kind)):
            cdur = f" ({child.duration_ms:.0f}ms)" if child.duration_ms else ""
            mark = "✓" if child.status == "success" else "✗"
            extra = f" {child.error_code}" if child.error_code else ""
            lines.append(f"│   ├── {child.kind} {child.name}{cdur} {mark}{extra}")
            retries = [
                s
                for s in trace.spans
                if s.parent_span_id == child.span_id and s.kind == "retry-attempt"
            ]
            for retry in retries:
                lines.append(f"│   │   └── {retry.name} ({retry.metadata.get('reason_code', '')})")

    lines.append("")
    lines.append("Metrics:")
    lines.append(f"  Sessions: {metrics.sessions_total} ({metrics.sessions_success} ok, {metrics.sessions_failed} failed)")
    lines.append(f"  LLM requests: {metrics.llm_requests} (retries: {metrics.llm_retries})")
    lines.append(f"  LLM latency p50/p95: {metrics.llm_p50_ms:.0f}ms / {metrics.llm_p95_ms:.0f}ms")
    lines.append(f"  Tool calls: {metrics.tool_calls} (failures: {metrics.tool_failures}, retries: {metrics.tool_retries})")
    lines.append(f"  Tool latency p50/p95: {metrics.tool_p50_ms:.0f}ms / {metrics.tool_p95_ms:.0f}ms")
    lines.append(f"  Tool success rate: {metrics.tool_success_rate:.0%}")
    lines.append(f"  Tokens: {metrics.input_tokens} in / {metrics.output_tokens} out")
    lines.append(f"  Cost: ${metrics.total_cost_usd:.4f} (avg ${metrics.avg_cost_usd:.4f}/session)")

    if report.retry_count:
        lines.append(f"  Recovery: {Recovery.RETRIED_N.value} (n={report.retry_count})")
    if report.notes:
        lines.append("")
        for note in report.notes:
            lines.append(f"  Note: {note}")

    return "\n".join(lines)
