"""Load session analytics from ClickHouse."""

from __future__ import annotations

from datetime import datetime, timezone

from liteness.analytics.models import (
    OverviewStats,
    SessionSummary,
    StopReasonCount,
    TimeBucket,
    ToolErrorRate,
    TurnSummary,
)
from liteness.analytics.time_range import clickhouse_since_clause
from liteness.clickhouse.client import ClickHouseClient


def _parse_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _table(client: ClickHouseClient) -> str:
    database = getattr(client, "database", "liteness")
    return f"{database}.session_events"


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def list_sessions(
    client: ClickHouseClient,
    *,
    since: datetime | None,
    limit: int,
    offset: int,
) -> list[SessionSummary]:
    since_clause = clickhouse_since_clause(since)
    sql = f"""
SELECT
    session_id,
    min(timestamp) AS first_seen,
    max(timestamp) AS last_seen,
    countIf(event_type = 'turn/end') AS turn_count,
    sumIf(JSONExtractUInt(payload, 'total_tokens'), event_type = 'turn/end') AS total_tokens,
    sumIf(JSONExtractFloat(payload, 'total_cost_usd'), event_type = 'turn/end') AS total_cost_usd,
    argMaxIf(stop_reason, timestamp, event_type = 'turn/end') AS stop_reason
FROM {_table(client)} FINAL
WHERE channel = 'ledger'{since_clause}
GROUP BY session_id
ORDER BY last_seen DESC
LIMIT {int(limit)} OFFSET {int(offset)}
"""
    rows = client.query_json(sql)
    summaries: list[SessionSummary] = []
    for row in rows:
        summaries.append(
            SessionSummary(
                session_id=str(row["session_id"]),
                source="clickhouse",
                first_seen=_parse_datetime(row["first_seen"]),
                last_seen=_parse_datetime(row["last_seen"]),
                turn_count=int(row.get("turn_count") or 0),
                total_tokens=int(row.get("total_tokens") or 0),
                total_cost_usd=float(row.get("total_cost_usd") or 0.0),
                stop_reason=str(row.get("stop_reason") or "") or None,
            )
        )
    return summaries


def all_session_ids(client: ClickHouseClient, *, since: datetime | None) -> set[str]:
    since_clause = clickhouse_since_clause(since)
    sql = f"""
SELECT DISTINCT session_id
FROM {_table(client)} FINAL
WHERE channel = 'ledger'{since_clause}
"""
    return {str(row["session_id"]) for row in client.query_json(sql)}


def overview(client: ClickHouseClient, *, since: datetime | None) -> OverviewStats:
    since_clause = clickhouse_since_clause(since)
    sql = f"""
SELECT
    uniqExact(session_id) AS session_count,
    countIf(event_type = 'turn/end') AS turn_count,
    sumIf(JSONExtractUInt(payload, 'input_tokens'), event_type = 'llm/response') AS input_tokens,
    sumIf(JSONExtractUInt(payload, 'output_tokens'), event_type = 'llm/response') AS output_tokens,
    sumIf(JSONExtractFloat(payload, 'cost_usd'), event_type = 'llm/response') AS total_cost_usd,
    countIf(event_type = 'llm/response') AS llm_requests,
    countIf(event_type = 'tool/result') AS tool_calls,
    countIf(event_type = 'tool/result' AND is_error = 1) AS tool_failures
FROM {_table(client)} FINAL
WHERE channel = 'ledger'{since_clause}
"""
    rows = client.query_json(sql)
    if not rows:
        return OverviewStats()
    row = rows[0]
    input_tokens = int(row.get("input_tokens") or 0)
    output_tokens = int(row.get("output_tokens") or 0)
    return OverviewStats(
        session_count=int(row.get("session_count") or 0),
        turn_count=int(row.get("turn_count") or 0),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        total_cost_usd=float(row.get("total_cost_usd") or 0.0),
        llm_requests=int(row.get("llm_requests") or 0),
        tool_calls=int(row.get("tool_calls") or 0),
        tool_failures=int(row.get("tool_failures") or 0),
    )


def timeseries(
    client: ClickHouseClient,
    *,
    since: datetime | None,
    bucket: str,
) -> list[TimeBucket]:
    since_clause = clickhouse_since_clause(since)
    bucket_expr = (
        "toStartOfHour(timestamp)" if bucket == "hour" else "toDate(timestamp)"
    )
    sql = f"""
SELECT
    {bucket_expr} AS bucket,
    countIf(event_type = 'turn/end') AS turn_count,
    sumIf(JSONExtractUInt(payload, 'total_tokens'), event_type = 'turn/end') AS total_tokens,
    sumIf(JSONExtractFloat(payload, 'total_cost_usd'), event_type = 'turn/end') AS total_cost_usd
FROM {_table(client)} FINAL
WHERE channel = 'ledger'{since_clause}
GROUP BY bucket
ORDER BY bucket
"""
    rows = client.query_json(sql)
    buckets: list[TimeBucket] = []
    for row in rows:
        bucket_value = row["bucket"]
        if isinstance(bucket_value, datetime):
            key = (
                bucket_value.strftime("%Y-%m-%dT%H:00")
                if bucket == "hour"
                else bucket_value.strftime("%Y-%m-%d")
            )
        else:
            key = str(bucket_value)
        buckets.append(
            TimeBucket(
                bucket=key,
                turn_count=int(row.get("turn_count") or 0),
                total_tokens=int(row.get("total_tokens") or 0),
                total_cost_usd=float(row.get("total_cost_usd") or 0.0),
            )
        )
    return buckets


def stop_reasons(client: ClickHouseClient, *, since: datetime | None) -> list[StopReasonCount]:
    since_clause = clickhouse_since_clause(since)
    sql = f"""
SELECT stop_reason, count() AS count
FROM {_table(client)} FINAL
WHERE event_type = 'turn/end'{since_clause}
GROUP BY stop_reason
ORDER BY count DESC
"""
    return [
        StopReasonCount(stop_reason=str(row["stop_reason"] or "unknown"), count=int(row["count"]))
        for row in client.query_json(sql)
    ]


def tool_error_rates(client: ClickHouseClient, *, since: datetime | None) -> list[ToolErrorRate]:
    since_clause = clickhouse_since_clause(since)
    sql = f"""
SELECT
    tool_name,
    count() AS calls,
    countIf(is_error = 1) AS failures
FROM {_table(client)} FINAL
WHERE event_type = 'tool/result' AND tool_name != ''{since_clause}
GROUP BY tool_name
ORDER BY calls DESC
"""
    return [
        ToolErrorRate(
            tool_name=str(row["tool_name"]),
            calls=int(row["calls"]),
            failures=int(row["failures"]),
        )
        for row in client.query_json(sql)
    ]


def turns_for_session(client: ClickHouseClient, session_id: str) -> list[TurnSummary]:
    sql = f"""
SELECT
    turn,
    timestamp,
    JSONExtractUInt(payload, 'total_tokens') AS tokens,
    JSONExtractFloat(payload, 'total_cost_usd') AS cost_usd,
    stop_reason
FROM {_table(client)} FINAL
WHERE session_id = {_sql_literal(session_id)} AND event_type = 'turn/end'
ORDER BY turn
"""
    turns: list[TurnSummary] = []
    for row in client.query_json(sql):
        turns.append(
            TurnSummary(
                turn=int(row["turn"]),
                tokens=int(row.get("tokens") or 0),
                cost_usd=float(row.get("cost_usd") or 0.0),
                stop_reason=str(row.get("stop_reason") or "") or None,
                timestamp=_parse_datetime(row["timestamp"]),
            )
        )
    return turns


def session_summary(
    client: ClickHouseClient,
    session_id: str,
) -> SessionSummary | None:
    sql = f"""
SELECT
    session_id,
    min(timestamp) AS first_seen,
    max(timestamp) AS last_seen,
    countIf(event_type = 'turn/end') AS turn_count,
    sumIf(JSONExtractUInt(payload, 'total_tokens'), event_type = 'turn/end') AS total_tokens,
    sumIf(JSONExtractFloat(payload, 'total_cost_usd'), event_type = 'turn/end') AS total_cost_usd,
    argMaxIf(stop_reason, timestamp, event_type = 'turn/end') AS stop_reason
FROM {_table(client)} FINAL
WHERE session_id = {_sql_literal(session_id)} AND channel = 'ledger'
GROUP BY session_id
"""
    rows = client.query_json(sql)
    if not rows:
        return None
    row = rows[0]
    return SessionSummary(
        session_id=str(row["session_id"]),
        source="clickhouse",
        first_seen=_parse_datetime(row["first_seen"]),
        last_seen=_parse_datetime(row["last_seen"]),
        turn_count=int(row.get("turn_count") or 0),
        total_tokens=int(row.get("total_tokens") or 0),
        total_cost_usd=float(row.get("total_cost_usd") or 0.0),
        stop_reason=str(row.get("stop_reason") or "") or None,
    )
