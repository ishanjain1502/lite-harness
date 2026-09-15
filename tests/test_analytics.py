from __future__ import annotations

import json
import shutil
import threading
from datetime import datetime, timezone
from http.client import HTTPConnection
from pathlib import Path

import pytest

from liteness.analytics.jsonl_source import (
    load_all_summaries,
    summarize_session,
    timeseries_from_dir,
)
from liteness.analytics.store import AnalyticsStore
from liteness.analytics.time_range import parse_since
from liteness.clickhouse.client import FakeClickHouseClient
from liteness.analytics.serve import AnalyticsHTTPServer


FIXTURE = Path("tests/fixtures/evals/weather-001.jsonl")


def test_parse_since_values() -> None:
    assert parse_since("all") is None
    assert parse_since("7d") is not None
    with pytest.raises(ValueError):
        parse_since("bad")


def test_summarize_jsonl_fixture(tmp_path: Path) -> None:
    target = tmp_path / "weather.jsonl"
    shutil.copy(FIXTURE, target)
    summary = summarize_session(target)
    assert summary is not None
    assert summary.session_id == "3c5a58e79de1"
    assert summary.turn_count == 1
    assert summary.total_tokens == 36
    assert summary.source == "jsonl"


def test_store_merges_clickhouse_and_jsonl_without_double_count(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    only_jsonl = sessions_dir / "local.jsonl"
    shutil.copy(FIXTURE, only_jsonl)

    fake = FakeClickHouseClient(
        query_results={
            "uniqExact(session_id)": [
                {
                    "session_count": 1,
                    "turn_count": 2,
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_cost_usd": 0.01,
                    "llm_requests": 2,
                    "tool_calls": 1,
                    "tool_failures": 0,
                }
            ],
            "DISTINCT session_id": [{"session_id": "ch-only"}],
            "GROUP BY session_id": [
                {
                    "session_id": "ch-only",
                    "first_seen": "2026-09-01T00:00:00+00:00",
                    "last_seen": "2026-09-02T00:00:00+00:00",
                    "turn_count": 2,
                    "total_tokens": 150,
                    "total_cost_usd": 0.01,
                    "stop_reason": "completed",
                }
            ],
            "GROUP BY bucket": [],
            "GROUP BY stop_reason": [],
            "tool_name": [],
        }
    )
    store = AnalyticsStore(clickhouse_client=fake, sessions_dir=sessions_dir)
    overview = store.overview()
    assert overview.session_count == 2
    assert overview.turn_count == 3
    assert overview.total_tokens == 186

    sessions = store.list_sessions(limit=10)
    assert len(sessions) == 2
    sources = {item.session_id: item.source for item in sessions}
    assert sources["3c5a58e79de1"] == "jsonl"
    assert sources["ch-only"] == "clickhouse"


def test_timeseries_from_jsonl(tmp_path: Path) -> None:
    target = tmp_path / "weather.jsonl"
    shutil.copy(FIXTURE, target)
    buckets = timeseries_from_dir(tmp_path, since=None, bucket="day")
    assert len(buckets) == 1
    assert buckets[0].total_tokens == 36


def test_analytics_api_serves_overview(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    shutil.copy(FIXTURE, sessions_dir / "weather.jsonl")

    store = AnalyticsStore(clickhouse_client=None, sessions_dir=sessions_dir)
    server = AnalyticsHTTPServer(("127.0.0.1", 0), store)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection(host, port, timeout=2)
        conn.request("GET", "/api/overview?since=all")
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
        assert response.status == 200
        assert body["session_count"] == 1
        assert body["total_tokens"] == 36
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_fake_clickhouse_query_json() -> None:
    client = FakeClickHouseClient(
        query_results={"SELECT 1": [{"value": 1}]},
    )
    rows = client.query_json("SELECT 1")
    assert rows == [{"value": 1}]
    assert client.queries == ["SELECT 1"]
