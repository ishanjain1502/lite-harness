from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from liteness.clickhouse.client import FakeClickHouseClient
from liteness.clickhouse.exporter import ClickHouseExporter
from liteness.clickhouse.projector import ClickHouseProjector
from liteness.session import SessionEvent


def _event() -> SessionEvent:
    return SessionEvent(
        id="e1",
        type="turn/start",
        timestamp=datetime(2026, 9, 5, tzinfo=timezone.utc),
        session_id="s1",
        turn=1,
        step=0,
        payload={"turn": 1},
    )


def test_emit_does_not_block_on_http(tmp_path: Path) -> None:
    client = FakeClickHouseClient(insert_delay_s=0.4)
    exporter = ClickHouseExporter(
        client,
        spill_path=tmp_path / "spill.jsonl",
        projector=ClickHouseProjector(),
        flush_interval_s=0.05,
        batch_size=10,
    )
    row = ClickHouseProjector().project_event(_event())
    assert row is not None
    started = time.monotonic()
    exporter.emit_session(row)
    elapsed = time.monotonic() - started
    assert elapsed < 0.15
    exporter.shutdown(session_id="s1")


def test_clickhouse_down_spills(tmp_path: Path) -> None:
    client = FakeClickHouseClient(fail_with=RuntimeError("down"))
    spill = tmp_path / "spill.jsonl"
    exporter = ClickHouseExporter(
        client,
        spill_path=spill,
        projector=ClickHouseProjector(),
        max_retries=1,
        flush_interval_s=0.01,
    )
    row = ClickHouseProjector().project_event(_event())
    assert row is not None
    exporter.emit_session(row)
    exporter.shutdown(session_id="s1", timeout_s=2.0)
    assert spill.exists()
    line = json.loads(spill.read_text(encoding="utf-8").splitlines()[0])
    assert line["table"] == "session_events"
    assert line["row"]["event_id"] == "e1"


def test_dedupe_event_id(tmp_path: Path) -> None:
    projector = ClickHouseProjector()
    event = _event()
    first = projector.project_event(event)
    second = projector.project_event(event)
    assert first is not None and second is not None
    assert first.event_id == second.event_id == "e1"
