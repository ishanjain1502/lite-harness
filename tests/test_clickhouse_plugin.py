from __future__ import annotations

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
    tables = [table for table, _ in client.calls]
    assert "session_events" in tables
