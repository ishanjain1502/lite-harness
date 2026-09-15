"""ClickHouse HTTP client and test doubles."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Protocol
from urllib.parse import quote

from liteness.plugins.base import PluginConfigError


def clickhouse_credentials_from_config(
    config: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    """Resolve optional HTTP basic-auth credentials from config or environment."""
    config = config or {}
    user = config.get("user") or os.environ.get("LITENESS_CLICKHOUSE_USER")
    password = config.get("password") or os.environ.get("LITENESS_CLICKHOUSE_PASSWORD")
    if user is not None and not str(user):
        user = None
    if password is not None and not str(password):
        password = None
    return user, password

CREATE_DATABASE = "CREATE DATABASE IF NOT EXISTS {database}"

CREATE_SESSION_EVENTS = """\
CREATE TABLE IF NOT EXISTS {database}.session_events
(
    event_id    String,
    channel     LowCardinality(String),
    event_type  LowCardinality(String),
    timestamp   DateTime64(3, 'UTC'),
    ingested_at DateTime64(3, 'UTC') DEFAULT now64(3),
    session_id  String,
    turn        UInt32,
    step        UInt32,
    severity    LowCardinality(String),
    tool_name   String DEFAULT '',
    call_id     String DEFAULT '',
    error_code  String DEFAULT '',
    is_error    UInt8 DEFAULT 0,
    stop_reason LowCardinality(String) DEFAULT '',
    model       String DEFAULT '',
    payload     String
)
ENGINE = ReplacingMergeTree(ingested_at)
ORDER BY (session_id, event_id)"""

CREATE_EVAL_RESULTS = """\
CREATE TABLE IF NOT EXISTS {database}.eval_results
(
    result_id     String,
    ingested_at   DateTime64(3, 'UTC') DEFAULT now64(3),
    timestamp     DateTime64(3, 'UTC'),
    suite         String,
    dataset       String DEFAULT '',
    case_id       String,
    session_id    String,
    evaluator     String,
    passed        Nullable(UInt8),
    score         Float64,
    reason        String,
    status        LowCardinality(String),
    session_path  String,
    preset        String DEFAULT '',
    provider      String DEFAULT '',
    model         String DEFAULT '',
    git_commit    String DEFAULT ''
)
ENGINE = ReplacingMergeTree(ingested_at)
ORDER BY (suite, case_id, evaluator, session_id, result_id)"""


class ClickHouseClient(Protocol):
    def ensure_schema(self) -> None: ...

    def insert_rows(self, table: str, rows: list[dict[str, Any]]) -> None: ...

    def query_json(self, sql: str) -> list[dict[str, Any]]: ...


class FakeClickHouseClient:
    def __init__(
        self,
        *,
        fail_with: Exception | None = None,
        insert_delay_s: float = 0.0,
        query_results: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.calls: list[tuple[str, list[dict]]] = []
        self.queries: list[str] = []
        self.fail_with = fail_with
        self.insert_delay_s = insert_delay_s
        self.query_results = query_results or {}

    def ensure_schema(self) -> None:
        pass

    def insert_rows(self, table: str, rows: list[dict[str, Any]]) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        if self.insert_delay_s:
            time.sleep(self.insert_delay_s)
        self.calls.append((table, list(rows)))

    def query_json(self, sql: str) -> list[dict[str, Any]]:
        self.queries.append(sql)
        if self.fail_with is not None:
            raise self.fail_with
        for pattern, rows in self.query_results.items():
            if pattern in sql:
                return list(rows)
        return []


class HttpClickHouseClient:
    def __init__(
        self,
        url: str,
        database: str,
        *,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        try:
            import httpx  # noqa: PLC0415
        except ImportError as exc:
            raise PluginConfigError(
                "httpx is required for ClickHouse export; "
                "install with: pip install liteness[clickhouse]"
            ) from exc

        self._httpx = httpx
        self.url = url.rstrip("/")
        self.database = database
        self.user = user
        self.password = password

    def _auth(self) -> tuple[str, str] | None:
        if self.user and self.password:
            return (self.user, self.password)
        return None

    def ensure_schema(self) -> None:
        statements = [
            CREATE_DATABASE.format(database=self.database),
            CREATE_SESSION_EVENTS.format(database=self.database),
            CREATE_EVAL_RESULTS.format(database=self.database),
        ]
        for statement in statements:
            self._post_query(statement)

    def insert_rows(self, table: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        query = f"INSERT INTO {table} FORMAT JSONEachRow"
        body = "\n".join(json.dumps(row) for row in rows)
        self._post_query(query, body=body)

    def query_json(self, sql: str) -> list[dict[str, Any]]:
        query = sql.strip().rstrip(";")
        if "FORMAT" not in query.upper():
            query = f"{query} FORMAT JSONEachRow"
        body = self._post_query(query, expect_body=True)
        if not body.strip():
            return []
        rows: list[dict[str, Any]] = []
        for line in body.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    def _post_query(self, query: str, *, body: str = "", expect_body: bool = False) -> str:
        params = (
            f"database={quote(self.database)}"
            f"&query={quote(query)}"
            "&date_time_input_format=best_effort"
        )
        url = f"{self.url}/?{params}"
        response = self._httpx.post(
            url, content=body, timeout=5.0, auth=self._auth()
        )
        if response.status_code < 200 or response.status_code >= 300:
            raise RuntimeError(
                f"ClickHouse HTTP {response.status_code}: {response.text}"
            )
        if expect_body:
            return response.text
        return ""
