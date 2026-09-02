"""In-memory aggregate metrics — counters, histograms, gauges."""

from __future__ import annotations

from dataclasses import dataclass, field


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


@dataclass
class MetricsSnapshot:
    sessions_total: int = 0
    sessions_success: int = 0
    sessions_failed: int = 0
    llm_requests: int = 0
    llm_retries: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    tool_retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0
    llm_latency_ms: list[float] = field(default_factory=list)
    tool_latency_ms: list[float] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def llm_p50_ms(self) -> float:
        return _percentile(self.llm_latency_ms, 50)

    @property
    def llm_p95_ms(self) -> float:
        return _percentile(self.llm_latency_ms, 95)

    @property
    def tool_p50_ms(self) -> float:
        return _percentile(self.tool_latency_ms, 50)

    @property
    def tool_p95_ms(self) -> float:
        return _percentile(self.tool_latency_ms, 95)

    @property
    def tool_success_rate(self) -> float:
        if self.tool_calls == 0:
            return 1.0
        return (self.tool_calls - self.tool_failures) / self.tool_calls

    @property
    def avg_cost_usd(self) -> float:
        if self.sessions_total == 0:
            return 0.0
        return self.total_cost_usd / self.sessions_total


class MetricsStore:
    def __init__(self) -> None:
        self._snapshot = MetricsSnapshot()

    @property
    def snapshot(self) -> MetricsSnapshot:
        return self._snapshot

    def record_session_outcome(self, *, success: bool) -> None:
        self._snapshot.sessions_total += 1
        if success:
            self._snapshot.sessions_success += 1
        else:
            self._snapshot.sessions_failed += 1

    def record_llm(
        self,
        *,
        duration_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        self._snapshot.llm_requests += 1
        self._snapshot.llm_latency_ms.append(duration_ms)
        self._snapshot.input_tokens += input_tokens
        self._snapshot.output_tokens += output_tokens
        self._snapshot.total_cost_usd += cost_usd

    def record_llm_retry(self) -> None:
        self._snapshot.llm_retries += 1

    def record_tool(self, *, duration_ms: float, failed: bool) -> None:
        self._snapshot.tool_calls += 1
        self._snapshot.tool_latency_ms.append(duration_ms)
        if failed:
            self._snapshot.tool_failures += 1

    def record_tool_retry(self) -> None:
        self._snapshot.tool_retries += 1

    def clear(self) -> None:
        self._snapshot = MetricsSnapshot()
