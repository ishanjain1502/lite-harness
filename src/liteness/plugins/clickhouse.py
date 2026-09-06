"""ClickHouse observability plugin — exports session events to ClickHouse."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from liteness.context import Context

if TYPE_CHECKING:
    from liteness.clickhouse.exporter import ClickHouseExporter


class ClickHousePlugin:
    name = "clickhouse"

    def __init__(self) -> None:
        self.exporter: ClickHouseExporter | None = None
        self._session_id = ""

    def install(self, ctx: Context, config: dict[str, Any]) -> None:
        from liteness.clickhouse.client import (
            HttpClickHouseClient,
            clickhouse_credentials_from_config,
        )
        from liteness.clickhouse.exporter import ClickHouseExporter
        from liteness.clickhouse.projector import ClickHouseProjector

        mode = str(config.get("mode") or "redacted")
        url = (
            config.get("url")
            or os.environ.get("LITENESS_CLICKHOUSE_URL")
            or "http://localhost:8123"
        )
        database = str(config.get("database") or "liteness")
        spill_path = Path(config.get("spill_path") or ".liteness/clickhouse-spill.jsonl")
        user, password = clickhouse_credentials_from_config(config)
        client = config.get("client")
        if client is None:
            client = HttpClickHouseClient(
                url=url,
                database=database,
                user=user,
                password=password,
            )
        projector = ClickHouseProjector(mode=mode)
        try:
            client.ensure_schema()
        except Exception:
            logging.getLogger("liteness.clickhouse").warning(
                "ClickHouse schema ensure failed; continuing with spill",
                exc_info=True,
            )
        self.exporter = ClickHouseExporter(
            client,
            spill_path=spill_path,
            projector=projector,
            flush_interval_s=float(config.get("flush_interval_s", 0.2)),
        )

        plugin = self

        def on_session_event(event_type, payload, *, session_event=None):
            if session_event is None or plugin.exporter is None:
                return
            plugin._session_id = session_event.session_id
            row = projector.project_event(session_event)
            if row is not None:
                plugin.exporter.emit_session(row)

        ctx.on("session/event", on_session_event)
        ctx.services["clickhouse"] = self

    def uninstall(self, ctx: Context) -> None:
        if self.exporter is not None:
            self.exporter.shutdown(session_id=self._session_id)
            self.exporter = None
        ctx.services.pop("clickhouse", None)
