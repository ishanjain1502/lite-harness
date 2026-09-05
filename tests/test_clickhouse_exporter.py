from __future__ import annotations

import json
import threading
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


def test_shutdown_inserts_all_queued_session_events(tmp_path: Path) -> None:
    client = FakeClickHouseClient()
    spill = tmp_path / "spill.jsonl"
    exporter = ClickHouseExporter(
        client,
        spill_path=spill,
        projector=ClickHouseProjector(),
        flush_interval_s=10.0,
        batch_size=100,
    )
    event_ids = {f"e{index}" for index in range(10)}
    for event_id in event_ids:
        exporter.emit("session_events", {"event_id": event_id})

    exporter.shutdown(session_id="s1")

    inserted_event_ids = {
        row.get("event_id")
        for table, rows in client.calls
        if table == "session_events"
        for row in rows
    }
    assert event_ids <= inserted_event_ids
    assert not spill.exists() or not spill.read_text(encoding="utf-8").strip()


def test_dedupe_event_id(tmp_path: Path) -> None:
    projector = ClickHouseProjector()
    event = _event()
    first = projector.project_event(event)
    second = projector.project_event(event)
    assert first is not None and second is not None
    assert first.event_id == second.event_id == "e1"


def test_shutdown_honors_timeout(tmp_path: Path) -> None:
    client = FakeClickHouseClient(insert_delay_s=2.0)
    exporter = ClickHouseExporter(
        client,
        spill_path=tmp_path / "spill.jsonl",
        projector=ClickHouseProjector(),
        flush_interval_s=0.05,
        batch_size=10,
    )
    row = ClickHouseProjector().project_event(_event())
    assert row is not None
    exporter.emit_session(row)
    started = time.monotonic()
    exporter.shutdown(session_id="s1", timeout_s=0.3)
    elapsed = time.monotonic() - started
    assert elapsed < 1.0


def test_emit_never_raises_when_spill_fails(tmp_path: Path) -> None:
    spill_dir = tmp_path / "spill_is_dir"
    spill_dir.mkdir()
    client = FakeClickHouseClient()
    exporter = ClickHouseExporter(
        client,
        spill_path=spill_dir,
        projector=ClickHouseProjector(),
        queue_maxsize=1,
        flush_interval_s=10.0,
        batch_size=100,
    )
    row = ClickHouseProjector().project_event(_event())
    assert row is not None
    exporter.emit_session(row)
    exporter.emit_session(row)
    exporter.shutdown()


def test_emit_never_raises_on_unserializable_row(tmp_path: Path, monkeypatch) -> None:
    insert_started = threading.Event()
    release_insert = threading.Event()

    class BlockingClient(FakeClickHouseClient):
        def insert_rows(self, table, rows):
            insert_started.set()
            release_insert.wait(timeout=2.0)
            super().insert_rows(table, rows)

    client = BlockingClient()
    exporter = ClickHouseExporter(
        client,
        spill_path=tmp_path / "spill.jsonl",
        projector=ClickHouseProjector(),
        queue_maxsize=1,
        flush_interval_s=0.01,
        batch_size=1,
    )
    original_dumps = json.dumps

    def _fail_spill_dumps(obj, *args, **kwargs):
        if isinstance(obj, dict) and "table" in obj and "row" in obj:
            raise TypeError("not serializable")
        return original_dumps(obj, *args, **kwargs)

    monkeypatch.setattr(
        "liteness.clickhouse.exporter.json.dumps", _fail_spill_dumps
    )
    exporter.emit("session_events", {"first": "blocks worker"})
    assert insert_started.wait(timeout=1.0)
    exporter.emit("session_events", {"second": "fills queue"})
    exporter.emit("session_events", {"bad": "value"})
    release_insert.set()
    exporter.shutdown(timeout_s=1.0)


def test_successful_insert_drains_existing_spill(tmp_path: Path) -> None:
    spill = tmp_path / "spill.jsonl"
    spill.write_text(
        json.dumps(
            {
                "table": "session_events",
                "row": {"event_id": "spilled-e1"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = FakeClickHouseClient(fail_with=RuntimeError("initially down"))
    exporter = ClickHouseExporter(
        client,
        spill_path=spill,
        projector=ClickHouseProjector(),
        max_retries=1,
        flush_interval_s=0.01,
    )
    client.fail_with = None

    exporter.emit("session_events", {"event_id": "live-e1"})
    exporter.shutdown(session_id="s1")

    inserted_event_ids = {
        row.get("event_id")
        for table, rows in client.calls
        if table == "session_events"
        for row in rows
    }
    assert {"live-e1", "spilled-e1"} <= inserted_event_ids
    assert not spill.exists()


def test_flush_window_not_reset_by_steady_stream(tmp_path: Path) -> None:
    client = FakeClickHouseClient()
    exporter = ClickHouseExporter(
        client,
        spill_path=tmp_path / "spill.jsonl",
        projector=ClickHouseProjector(),
        flush_interval_s=0.2,
        batch_size=1000,
    )
    row = ClickHouseProjector().project_event(_event())
    assert row is not None
    exporter.emit_session(row)
    time.sleep(0.1)
    exporter.emit_session(row)
    time.sleep(0.15)
    assert len(client.calls) >= 1
    exporter.shutdown()
