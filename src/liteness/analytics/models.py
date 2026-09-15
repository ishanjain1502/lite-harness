"""Shared analytics data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

SessionSource = Literal["clickhouse", "jsonl", "both"]


@dataclass
class SessionSummary:
    session_id: str
    source: SessionSource
    first_seen: datetime
    last_seen: datetime
    turn_count: int
    total_tokens: int
    total_cost_usd: float
    stop_reason: str | None = None
    session_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "source": self.source,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "turn_count": self.turn_count,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "stop_reason": self.stop_reason,
            "session_path": self.session_path,
        }


@dataclass
class TurnSummary:
    turn: int
    tokens: int
    cost_usd: float
    stop_reason: str | None
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
            "stop_reason": self.stop_reason,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OverviewStats:
    session_count: int = 0
    turn_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    llm_requests: int = 0
    tool_calls: int = 0
    tool_failures: int = 0

    def to_dict(self) -> dict:
        return {
            "session_count": self.session_count,
            "turn_count": self.turn_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "llm_requests": self.llm_requests,
            "tool_calls": self.tool_calls,
            "tool_failures": self.tool_failures,
            "avg_tokens_per_turn": (
                self.total_tokens / self.turn_count if self.turn_count else 0.0
            ),
            "avg_cost_per_turn": (
                self.total_cost_usd / self.turn_count if self.turn_count else 0.0
            ),
        }


@dataclass
class TimeBucket:
    bucket: str
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    turn_count: int = 0

    def to_dict(self) -> dict:
        return {
            "bucket": self.bucket,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "turn_count": self.turn_count,
        }


@dataclass
class ToolErrorRate:
    tool_name: str
    calls: int
    failures: int

    @property
    def error_rate(self) -> float:
        if self.calls == 0:
            return 0.0
        return self.failures / self.calls

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "calls": self.calls,
            "failures": self.failures,
            "error_rate": self.error_rate,
        }


@dataclass
class StopReasonCount:
    stop_reason: str
    count: int

    def to_dict(self) -> dict:
        return {"stop_reason": self.stop_reason, "count": self.count}


@dataclass
class SessionDetail:
    summary: SessionSummary
    turns: list[TurnSummary] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary.to_dict(),
            "turns": [turn.to_dict() for turn in self.turns],
            "metrics": self.metrics,
        }
