"""Non-blocking ClickHouse exporter with queue, spill, and retry."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

from liteness.clickhouse.client import ClickHouseClient
from liteness.clickhouse.projector import (
    ClickHouseProjector,
    EvalResultRow,
    SessionEventRow,
)

logger = logging.getLogger("liteness.clickhouse")

_SENTINEL = object()
_SESSION_EVENTS = "session_events"
_EVAL_RESULTS = "eval_results"


class ClickHouseExporter:
    def __init__(
        self,
        client: ClickHouseClient,
        *,
        spill_path: Path,
        projector: ClickHouseProjector,
        batch_size: int = 100,
        flush_interval_s: float = 0.2,
        max_retries: int = 3,
        queue_maxsize: int = 1000,
    ) -> None:
        self.client = client
        self.projector = projector
        self.spill_path = spill_path
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s
        self.max_retries = max_retries
        self._queue: queue.Queue[tuple[str, dict[str, Any]] | object] = queue.Queue(
            maxsize=queue_maxsize
        )
        self._stopping = False
        self._degraded = False
        self._degraded_emitted = False
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self._drain_spill()

    def emit(self, table: str, row: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait((table, row))
        except queue.Full:
            self._spill(table, row)

    def emit_session(self, row: SessionEventRow) -> None:
        self.emit(_SESSION_EVENTS, row.to_insert_dict())

    def emit_eval(self, row: EvalResultRow) -> None:
        self.emit(_EVAL_RESULTS, row.to_insert_dict())

    def shutdown(self, session_id: str = "", timeout_s: float = 3.0) -> None:
        self._stopping = True
        try:
            self._queue.put_nowait(_SENTINEL)
        except queue.Full:
            pass
        self._worker.join(timeout=timeout_s)

        shutdown_row = self.projector.project_ops(
            "ops/shutdown",
            session_id,
            {"session_id": session_id},
        )
        shutdown_dict = shutdown_row.to_insert_dict()
        try:
            self.client.insert_rows(_SESSION_EVENTS, [shutdown_dict])
        except Exception:
            self._spill(_SESSION_EVENTS, shutdown_dict)

        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is _SENTINEL:
                continue
            table, row = item
            self._spill(table, row)

        self._degraded_emitted = False

    def _worker_loop(self) -> None:
        while True:
            try:
                first = self._queue.get(timeout=self.flush_interval_s)
            except queue.Empty:
                continue
            if first is _SENTINEL:
                break

            batch: list[tuple[str, dict[str, Any]]] = [first]
            while len(batch) < self.batch_size:
                try:
                    item = self._queue.get(timeout=self.flush_interval_s)
                except queue.Empty:
                    break
                if item is _SENTINEL:
                    self._flush_collected(batch)
                    return
                batch.append(item)

            self._flush_collected(batch)

    def _flush_collected(self, items: list[tuple[str, dict[str, Any]]]) -> None:
        by_table: dict[str, list[dict[str, Any]]] = {}
        for table, row in items:
            by_table.setdefault(table, []).append(row)
        for table, rows in by_table.items():
            self._flush_batch(table, rows)

    def _flush_batch(self, table: str, rows: list[dict[str, Any]]) -> None:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                self.client.insert_rows(table, rows)
                return
            except Exception as exc:
                last_exc = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(min(2**attempt, 5))
        assert last_exc is not None
        for row in rows:
            self._spill(table, row)
        was_degraded = self._degraded
        self._degraded = True
        if not self._degraded_emitted:
            self._degraded_emitted = True
            ops_row = self.projector.project_ops(
                "ops/persistence_degraded",
                "",
                {"code": "PERSISTENCE_DEGRADED", "message": str(last_exc)},
            )
            if was_degraded:
                try:
                    self.client.insert_rows(
                        _SESSION_EVENTS, [ops_row.to_insert_dict()]
                    )
                except Exception:
                    self._spill(_SESSION_EVENTS, ops_row.to_insert_dict())
            else:
                self.emit_session(ops_row)

    def _spill(self, table: str, row: dict[str, Any]) -> None:
        line = json.dumps({"table": table, "row": row}, ensure_ascii=False)
        try:
            self.spill_path.parent.mkdir(parents=True, exist_ok=True)
            with self.spill_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            logger.exception("failed to spill ClickHouse row to %s", self.spill_path)

    def _drain_spill(self) -> None:
        if not self.spill_path.exists():
            return
        try:
            lines = self.spill_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        if not lines:
            return

        by_table: dict[str, list[dict[str, Any]]] = {}
        for line in lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                return
            by_table.setdefault(entry["table"], []).append(entry["row"])

        try:
            for table, rows in by_table.items():
                self.client.insert_rows(table, rows)
        except Exception:
            return

        try:
            self.spill_path.unlink()
        except OSError:
            try:
                self.spill_path.write_text("", encoding="utf-8")
            except OSError:
                pass
