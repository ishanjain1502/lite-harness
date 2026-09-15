"""Parse relative time ranges for analytics filters."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

_SINCE_RE = re.compile(r"^(\d+)([hdwm])$", re.IGNORECASE)


def parse_since(value: str | None) -> datetime | None:
    """Return UTC cutoff for values like 24h, 7d, 30d, or None for all time."""
    if value is None or value in {"", "all"}:
        return None
    match = _SINCE_RE.match(value.strip())
    if match is None:
        raise ValueError(f"invalid since value: {value!r} (use 24h, 7d, 30d, or all)")
    amount = int(match.group(1))
    unit = match.group(2).lower()
    if unit == "h":
        delta = timedelta(hours=amount)
    elif unit == "d":
        delta = timedelta(days=amount)
    elif unit == "w":
        delta = timedelta(weeks=amount)
    else:
        delta = timedelta(days=amount * 30)
    return datetime.now(timezone.utc) - delta


def clickhouse_since_clause(since: datetime | None) -> str:
    if since is None:
        return ""
    formatted = since.strftime("%Y-%m-%d %H:%M:%S")
    return f" AND timestamp >= toDateTime64('{formatted}', 3, 'UTC')"
