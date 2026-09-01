"""Telemetry plugin — observe session events without model-facing tools."""

from __future__ import annotations

from typing import Any

from liteness.context import Context


class TelemetryPlugin:
    name = "telemetry"

    def install(self, ctx: Context, config: dict[str, Any]) -> None:
        def on_session_event(event_name: str, payload: dict[str, Any]) -> None:
            ctx.services.setdefault("telemetry_events", []).append(
                {"event": event_name, "payload": payload}
            )

        ctx.on("session/event", on_session_event)

    def uninstall(self, ctx: Context) -> None:
        pass
