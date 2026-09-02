"""Telemetry plugin — TraceStore + MetricsStore behind the plugin boundary."""

from __future__ import annotations

from typing import Any

from liteness.context import Context
from liteness.session import SessionEvent
from liteness.telemetry.projector import EventProjector, TraceReport, format_trace_report


class TelemetryPlugin:
    name = "telemetry"

    def __init__(self) -> None:
        self._projector = EventProjector()

    @property
    def projector(self) -> EventProjector:
        return self._projector

    def install(self, ctx: Context, config: dict[str, Any]) -> None:
        plugin = self

        def on_session_event(
            event_type: str,
            payload: dict[str, Any],
            *,
            session_event: SessionEvent | None = None,
        ) -> None:
            if session_event is not None:
                plugin._projector.process(session_event)
                return
            # Legacy test path: minimal event without full SessionEvent
            ctx.services.setdefault("telemetry_events", []).append(
                {"event": event_type, "payload": payload}
            )

        ctx.on("session/event", on_session_event)
        ctx.services["telemetry"] = self

    def uninstall(self, ctx: Context) -> None:
        ctx.services.pop("telemetry", None)

    def report(self, session_id: str | None = None) -> TraceReport:
        return self._projector.build_report(session_id)

    def format_report(self, session_id: str | None = None) -> str:
        return format_trace_report(self.report(session_id))
