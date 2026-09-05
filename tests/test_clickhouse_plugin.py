from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from liteness.context import Context
from liteness.clickhouse.client import FakeClickHouseClient
from liteness.plugins import get_plugin, list_plugins
from liteness.plugins.clickhouse import ClickHousePlugin
from liteness.session import SessionEvent


def test_clickhouse_is_registered() -> None:
    assert "clickhouse" in list_plugins()
    plugin = get_plugin("clickhouse")
    assert isinstance(plugin, ClickHousePlugin)


def test_plugin_exports_session_event(tmp_path: Path) -> None:
    ctx = Context()
    plugin = ClickHousePlugin()
    client = FakeClickHouseClient()
    plugin.install(
        ctx,
        {
            "mode": "redacted",
            "spill_path": str(tmp_path / "spill.jsonl"),
            "client": client,
            "flush_interval_s": 0.01,
        },
    )
    event = SessionEvent(
        id="e9",
        type="user/message",
        timestamp=datetime(2026, 9, 5, tzinfo=timezone.utc),
        session_id="s9",
        turn=1,
        step=0,
        payload={"content": "hi"},
    )
    ctx.emit("session/event", event.type, event.payload, session_event=event)
    plugin.uninstall(ctx)
    event_ids = {
        row["event_id"]
        for table, rows in client.calls
        if table == "session_events"
        for row in rows
    }
    assert "e9" in event_ids


def test_plugin_ensures_schema_before_exporter_drains_spill(tmp_path: Path) -> None:
    class SchemaAwareClient(FakeClickHouseClient):
        def __init__(self) -> None:
            super().__init__()
            self.schema_ready = False

        def ensure_schema(self) -> None:
            self.schema_ready = True

        def insert_rows(self, table, rows) -> None:
            if not self.schema_ready:
                raise RuntimeError("schema missing")
            super().insert_rows(table, rows)

    spill = tmp_path / "spill.jsonl"
    spill.write_text(
        json.dumps(
            {
                "table": "session_events",
                "row": {"event_id": "spilled-before-install"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = SchemaAwareClient()
    plugin = ClickHousePlugin()
    ctx = Context()

    plugin.install(
        ctx,
        {"client": client, "spill_path": str(spill), "flush_interval_s": 0.01},
    )
    plugin.uninstall(ctx)

    event_ids = {
        row["event_id"]
        for table, rows in client.calls
        if table == "session_events"
        for row in rows
    }
    assert "spilled-before-install" in event_ids
    assert not spill.exists()
