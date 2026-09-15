"""Load session analytics from local JSONL logs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from liteness.analytics.models import (
    OverviewStats,
    SessionDetail,
    SessionSummary,
    StopReasonCount,
    TimeBucket,
    ToolErrorRate,
    TurnSummary,
)
from liteness.session import Session
from liteness.telemetry.projector import project_events


def discover_jsonl_files(sessions_dir: Path) -> list[Path]:
    if not sessions_dir.exists():
        return []
    return sorted(sessions_dir.rglob("*.jsonl"))


def _event_in_range(timestamp: datetime, since: datetime | None) -> bool:
    if since is None:
        return True
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=since.tzinfo)
    return timestamp >= since


def summarize_session(path: Path, *, since: datetime | None = None) -> SessionSummary | None:
    try:
        session = Session.load_from_jsonl(path, recover=False)
    except (OSError, ValueError, KeyError):
        return None
    if not session.events:
        return None

    first_seen = session.events[0].timestamp
    last_seen = session.events[-1].timestamp
    if since is not None and last_seen < since:
        return None

    turn_count = 0
    total_tokens = 0
    total_cost_usd = 0.0
    last_stop_reason: str | None = None
    for event in session.events:
        if not _event_in_range(event.timestamp, since):
            continue
        if event.type == "turn/end":
            turn_count += 1
            total_tokens += int(event.payload.get("total_tokens", 0))
            total_cost_usd += float(event.payload.get("total_cost_usd", 0.0))
            last_stop_reason = str(event.payload.get("stop_reason") or "") or None

    if turn_count == 0 and since is not None:
        return None

    return SessionSummary(
        session_id=session.session_id,
        source="jsonl",
        first_seen=first_seen,
        last_seen=last_seen,
        turn_count=turn_count or max(session._turn, 0),
        total_tokens=total_tokens,
        total_cost_usd=total_cost_usd,
        stop_reason=last_stop_reason,
        session_path=str(path),
    )


def load_all_summaries(
    sessions_dir: Path,
    *,
    since: datetime | None = None,
) -> dict[str, SessionSummary]:
    summaries: dict[str, SessionSummary] = {}
    for path in discover_jsonl_files(sessions_dir):
        summary = summarize_session(path, since=since)
        if summary is None:
            continue
        existing = summaries.get(summary.session_id)
        if existing is None or summary.last_seen > existing.last_seen:
            summaries[summary.session_id] = summary
    return summaries


def turns_for_session(path: Path) -> list[TurnSummary]:
    session = Session.load_from_jsonl(path, recover=False)
    turns: list[TurnSummary] = []
    for event in session.events:
        if event.type != "turn/end":
            continue
        turns.append(
            TurnSummary(
                turn=event.turn,
                tokens=int(event.payload.get("total_tokens", 0)),
                cost_usd=float(event.payload.get("total_cost_usd", 0.0)),
                stop_reason=str(event.payload.get("stop_reason") or "") or None,
                timestamp=event.timestamp,
            )
        )
    return sorted(turns, key=lambda item: item.turn)


def session_detail(path: Path) -> SessionDetail:
    summary = summarize_session(path)
    if summary is None:
        raise FileNotFoundError(path)
    session = Session.load_from_jsonl(path, recover=False)
    report = project_events(session.events)
    return SessionDetail(
        summary=summary,
        turns=turns_for_session(path),
        metrics={
            "llm_requests": report.metrics.llm_requests,
            "tool_calls": report.metrics.tool_calls,
            "tool_failures": report.metrics.tool_failures,
            "llm_p50_ms": report.metrics.llm_p50_ms,
            "llm_p95_ms": report.metrics.llm_p95_ms,
            "tool_success_rate": report.metrics.tool_success_rate,
        },
    )


def overview_from_summaries(summaries: dict[str, SessionSummary]) -> OverviewStats:
    overview = OverviewStats()
    overview.session_count = len(summaries)
    for summary in summaries.values():
        overview.turn_count += summary.turn_count
        overview.total_tokens += summary.total_tokens
        overview.total_cost_usd += summary.total_cost_usd
    return overview


def _bucket_key(timestamp: datetime, bucket: str) -> str:
    if bucket == "hour":
        return timestamp.strftime("%Y-%m-%dT%H:00")
    return timestamp.strftime("%Y-%m-%d")


def timeseries_from_dir(
    sessions_dir: Path,
    *,
    since: datetime | None,
    bucket: str,
    exclude_session_ids: set[str] | None = None,
) -> list[TimeBucket]:
    exclude = exclude_session_ids or set()
    buckets: dict[str, TimeBucket] = {}
    for path in discover_jsonl_files(sessions_dir):
        try:
            session = Session.load_from_jsonl(path, recover=False)
        except (OSError, ValueError, KeyError):
            continue
        if session.session_id in exclude:
            continue
        for event in session.events:
            if event.type != "turn/end":
                continue
            if not _event_in_range(event.timestamp, since):
                continue
            key = _bucket_key(event.timestamp, bucket)
            entry = buckets.setdefault(key, TimeBucket(bucket=key))
            entry.turn_count += 1
            entry.total_tokens += int(event.payload.get("total_tokens", 0))
            entry.total_cost_usd += float(event.payload.get("total_cost_usd", 0.0))
    return sorted(buckets.values(), key=lambda item: item.bucket)


def stop_reasons_from_dir(
    sessions_dir: Path,
    *,
    since: datetime | None,
    exclude_session_ids: set[str] | None = None,
) -> list[StopReasonCount]:
    exclude = exclude_session_ids or set()
    counts: dict[str, int] = {}
    for path in discover_jsonl_files(sessions_dir):
        try:
            session = Session.load_from_jsonl(path, recover=False)
        except (OSError, ValueError, KeyError):
            continue
        if session.session_id in exclude:
            continue
        for event in session.events:
            if event.type != "turn/end":
                continue
            if not _event_in_range(event.timestamp, since):
                continue
            reason = str(event.payload.get("stop_reason") or "unknown")
            counts[reason] = counts.get(reason, 0) + 1
    return [
        StopReasonCount(stop_reason=reason, count=count)
        for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def tool_errors_from_dir(
    sessions_dir: Path,
    *,
    since: datetime | None,
    exclude_session_ids: set[str] | None = None,
) -> list[ToolErrorRate]:
    exclude = exclude_session_ids or set()
    calls: dict[str, int] = {}
    failures: dict[str, int] = {}
    for path in discover_jsonl_files(sessions_dir):
        try:
            session = Session.load_from_jsonl(path, recover=False)
        except (OSError, ValueError, KeyError):
            continue
        if session.session_id in exclude:
            continue
        for event in session.events:
            if event.type != "tool/result":
                continue
            if not _event_in_range(event.timestamp, since):
                continue
            name = str(event.payload.get("name") or "unknown")
            calls[name] = calls.get(name, 0) + 1
            if event.payload.get("is_error"):
                failures[name] = failures.get(name, 0) + 1
    return [
        ToolErrorRate(
            tool_name=name,
            calls=calls[name],
            failures=failures.get(name, 0),
        )
        for name in sorted(calls, key=lambda key: (-calls[key], key))
    ]
