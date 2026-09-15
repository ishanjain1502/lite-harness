"""Merged analytics store — ClickHouse warehouse plus local JSONL sessions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from liteness.analytics import clickhouse_source, jsonl_source
from liteness.analytics.models import (
    OverviewStats,
    SessionDetail,
    SessionSummary,
    StopReasonCount,
    TimeBucket,
    ToolErrorRate,
    TurnSummary,
)
from liteness.clickhouse.client import ClickHouseClient


def _merge_overview(base: OverviewStats, extra: OverviewStats) -> OverviewStats:
    return OverviewStats(
        session_count=base.session_count + extra.session_count,
        turn_count=base.turn_count + extra.turn_count,
        input_tokens=base.input_tokens + extra.input_tokens,
        output_tokens=base.output_tokens + extra.output_tokens,
        total_tokens=base.total_tokens + extra.total_tokens,
        total_cost_usd=base.total_cost_usd + extra.total_cost_usd,
        llm_requests=base.llm_requests + extra.llm_requests,
        tool_calls=base.tool_calls + extra.tool_calls,
        tool_failures=base.tool_failures + extra.tool_failures,
    )


def _merge_timeseries(
    primary: list[TimeBucket],
    secondary: list[TimeBucket],
) -> list[TimeBucket]:
    merged: dict[str, TimeBucket] = {bucket.bucket: bucket for bucket in primary}
    for bucket in secondary:
        existing = merged.get(bucket.bucket)
        if existing is None:
            merged[bucket.bucket] = bucket
        else:
            existing.turn_count += bucket.turn_count
            existing.total_tokens += bucket.total_tokens
            existing.total_cost_usd += bucket.total_cost_usd
    return sorted(merged.values(), key=lambda item: item.bucket)


def _merge_stop_reasons(
    primary: list[StopReasonCount],
    secondary: list[StopReasonCount],
) -> list[StopReasonCount]:
    counts: dict[str, int] = {item.stop_reason: item.count for item in primary}
    for item in secondary:
        counts[item.stop_reason] = counts.get(item.stop_reason, 0) + item.count
    return [
        StopReasonCount(stop_reason=reason, count=count)
        for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _merge_tool_errors(
    primary: list[ToolErrorRate],
    secondary: list[ToolErrorRate],
) -> list[ToolErrorRate]:
    calls: dict[str, int] = {item.tool_name: item.calls for item in primary}
    failures: dict[str, int] = {item.tool_name: item.failures for item in primary}
    for item in secondary:
        calls[item.tool_name] = calls.get(item.tool_name, 0) + item.calls
        failures[item.tool_name] = failures.get(item.tool_name, 0) + item.failures
    return [
        ToolErrorRate(tool_name=name, calls=calls[name], failures=failures.get(name, 0))
        for name in sorted(calls, key=lambda key: (-calls[key], key))
    ]


class AnalyticsStore:
    def __init__(
        self,
        *,
        clickhouse_client: ClickHouseClient | None = None,
        sessions_dir: Path | str = ".sessions",
    ) -> None:
        self._clickhouse = clickhouse_client
        self._sessions_dir = Path(sessions_dir)

    def _jsonl_summaries(self, since: datetime | None) -> dict[str, SessionSummary]:
        return jsonl_source.load_all_summaries(self._sessions_dir, since=since)

    def _clickhouse_session_ids(self, since: datetime | None) -> set[str]:
        if self._clickhouse is None:
            return set()
        try:
            return clickhouse_source.all_session_ids(self._clickhouse, since=since)
        except Exception:
            return set()

    def list_sessions(
        self,
        *,
        since: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SessionSummary]:
        merged: dict[str, SessionSummary] = {}
        if self._clickhouse is not None:
            try:
                for summary in clickhouse_source.list_sessions(
                    self._clickhouse,
                    since=since,
                    limit=10_000,
                    offset=0,
                ):
                    merged[summary.session_id] = summary
            except Exception:
                pass

        ch_ids = set(merged)
        for session_id, summary in self._jsonl_summaries(since).items():
            if session_id in merged:
                existing = merged[session_id]
                existing.source = "both"
                if summary.session_path:
                    existing.session_path = summary.session_path
            else:
                merged[session_id] = summary

        ordered = sorted(merged.values(), key=lambda item: item.last_seen, reverse=True)
        return ordered[offset : offset + limit]

    def overview(self, *, since: datetime | None = None) -> OverviewStats:
        ch_overview = OverviewStats()
        ch_ids: set[str] = set()
        if self._clickhouse is not None:
            try:
                ch_overview = clickhouse_source.overview(self._clickhouse, since=since)
                ch_ids = clickhouse_source.all_session_ids(self._clickhouse, since=since)
            except Exception:
                pass

        jsonl_summaries = self._jsonl_summaries(since)
        jsonl_only = {
            session_id: summary
            for session_id, summary in jsonl_summaries.items()
            if session_id not in ch_ids
        }
        jsonl_overview = jsonl_source.overview_from_summaries(jsonl_only)
        return _merge_overview(ch_overview, jsonl_overview)

    def timeseries(
        self,
        *,
        since: datetime | None = None,
        bucket: str = "day",
    ) -> list[TimeBucket]:
        ch_buckets: list[TimeBucket] = []
        ch_ids: set[str] = set()
        if self._clickhouse is not None:
            try:
                ch_buckets = clickhouse_source.timeseries(
                    self._clickhouse, since=since, bucket=bucket
                )
                ch_ids = clickhouse_source.all_session_ids(self._clickhouse, since=since)
            except Exception:
                pass
        jsonl_buckets = jsonl_source.timeseries_from_dir(
            self._sessions_dir,
            since=since,
            bucket=bucket,
            exclude_session_ids=ch_ids,
        )
        return _merge_timeseries(ch_buckets, jsonl_buckets)

    def stop_reasons(self, *, since: datetime | None = None) -> list[StopReasonCount]:
        ch_rows: list[StopReasonCount] = []
        ch_ids: set[str] = set()
        if self._clickhouse is not None:
            try:
                ch_rows = clickhouse_source.stop_reasons(self._clickhouse, since=since)
                ch_ids = clickhouse_source.all_session_ids(self._clickhouse, since=since)
            except Exception:
                pass
        jsonl_rows = jsonl_source.stop_reasons_from_dir(
            self._sessions_dir,
            since=since,
            exclude_session_ids=ch_ids,
        )
        return _merge_stop_reasons(ch_rows, jsonl_rows)

    def tool_error_rates(self, *, since: datetime | None = None) -> list[ToolErrorRate]:
        ch_rows: list[ToolErrorRate] = []
        ch_ids: set[str] = set()
        if self._clickhouse is not None:
            try:
                ch_rows = clickhouse_source.tool_error_rates(self._clickhouse, since=since)
                ch_ids = clickhouse_source.all_session_ids(self._clickhouse, since=since)
            except Exception:
                pass
        jsonl_rows = jsonl_source.tool_errors_from_dir(
            self._sessions_dir,
            since=since,
            exclude_session_ids=ch_ids,
        )
        return _merge_tool_errors(ch_rows, jsonl_rows)

    def get_session(self, session_id: str) -> SessionDetail | None:
        summary: SessionSummary | None = None
        turns: list[TurnSummary] = []
        metrics: dict = {}

        if self._clickhouse is not None:
            try:
                summary = clickhouse_source.session_summary(self._clickhouse, session_id)
                if summary is not None:
                    turns = clickhouse_source.turns_for_session(self._clickhouse, session_id)
            except Exception:
                summary = None

        jsonl_path: Path | None = None
        for path in jsonl_source.discover_jsonl_files(self._sessions_dir):
            try:
                candidate = jsonl_source.summarize_session(path)
            except Exception:
                continue
            if candidate is not None and candidate.session_id == session_id:
                jsonl_path = path
                if summary is None:
                    summary = candidate
                else:
                    summary.source = "both"
                    summary.session_path = str(path)
                break

        if summary is None:
            return None

        if jsonl_path is not None:
            try:
                detail = jsonl_source.session_detail(jsonl_path)
                metrics = detail.metrics
                if not turns:
                    turns = detail.turns
            except Exception:
                pass

        return SessionDetail(summary=summary, turns=turns, metrics=metrics)
