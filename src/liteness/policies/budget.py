"""Budget manager — session-scoped hard limits and soft warnings."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

EventEmitter = Callable[[str, dict], None]


class BudgetDimension(str, Enum):
    TOKENS = "tokens"
    COST = "cost_usd"
    WALL_CLOCK = "wall_clock_s"
    TOOL_CALLS = "tool_calls_per_turn"


@dataclass
class BudgetConfig:
    max_tokens_session: int | None = None
    max_cost_session: float | None = None
    max_wall_clock_session_s: float | None = None
    max_tool_calls_per_turn: int | None = None
    soft_token_warn_fraction: float | None = 0.8
    soft_cost_warn_fraction: float | None = 0.8


@dataclass
class BudgetState:
    tokens_used: int = 0
    cost_usd: float = 0.0
    tool_calls_this_turn: int = 0
    warned_tokens: bool = False
    warned_cost: bool = False
    exhausted: bool = False
    exhausted_dimension: BudgetDimension | None = None


class BudgetManager:
    def __init__(
        self,
        config: BudgetConfig | None = None,
        *,
        emit_event: EventEmitter | None = None,
    ) -> None:
        self.config = config or BudgetConfig()
        self.emit_event = emit_event
        self._state = BudgetState()
        self._started_at = time.monotonic()

    @property
    def state(self) -> BudgetState:
        return self._state

    def reset_turn(self) -> None:
        self._state.tool_calls_this_turn = 0

    def wall_clock_elapsed_s(self) -> float:
        return time.monotonic() - self._started_at

    def can_continue(self) -> bool:
        if self._state.exhausted:
            return False
        cfg = self.config
        if cfg.max_wall_clock_session_s is not None:
            if self.wall_clock_elapsed_s() >= cfg.max_wall_clock_session_s:
                self._exhaust(BudgetDimension.WALL_CLOCK, cfg.max_wall_clock_session_s, self.wall_clock_elapsed_s())
                return False
        if cfg.max_tokens_session is not None and self._state.tokens_used >= cfg.max_tokens_session:
            self._exhaust(BudgetDimension.TOKENS, cfg.max_tokens_session, self._state.tokens_used)
            return False
        if cfg.max_cost_session is not None and self._state.cost_usd >= cfg.max_cost_session:
            self._exhaust(BudgetDimension.COST, cfg.max_cost_session, self._state.cost_usd)
            return False
        return True

    def can_start_tool_calls(self, count: int = 1) -> bool:
        if not self.can_continue():
            return False
        limit = self.config.max_tool_calls_per_turn
        if limit is None:
            return True
        if self._state.tool_calls_this_turn + count > limit:
            self._exhaust(
                BudgetDimension.TOOL_CALLS,
                limit,
                self._state.tool_calls_this_turn + count,
            )
            return False
        return True

    def record_llm_usage(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float = 0.0,
    ) -> None:
        self._state.tokens_used += input_tokens + output_tokens
        self._state.cost_usd += cost_usd
        self._maybe_warn()

    def record_tool_calls(self, count: int) -> None:
        self._state.tool_calls_this_turn += count

    def _maybe_warn(self) -> None:
        cfg = self.config
        if (
            cfg.max_tokens_session is not None
            and cfg.soft_token_warn_fraction is not None
            and not self._state.warned_tokens
        ):
            threshold = cfg.max_tokens_session * cfg.soft_token_warn_fraction
            if self._state.tokens_used >= threshold:
                self._state.warned_tokens = True
                self._warn(BudgetDimension.TOKENS, threshold, self._state.tokens_used)
        if (
            cfg.max_cost_session is not None
            and cfg.soft_cost_warn_fraction is not None
            and not self._state.warned_cost
        ):
            threshold = cfg.max_cost_session * cfg.soft_cost_warn_fraction
            if self._state.cost_usd >= threshold:
                self._state.warned_cost = True
                self._warn(BudgetDimension.COST, threshold, self._state.cost_usd)

    def _warn(self, dimension: BudgetDimension, threshold: float, observed: float) -> None:
        if self.emit_event is not None:
            self.emit_event(
                "budget/warn",
                {"dimension": dimension.value, "threshold": threshold, "observed": observed},
            )

    def _exhaust(self, dimension: BudgetDimension, limit: float, observed: float) -> None:
        if self._state.exhausted:
            return
        self._state.exhausted = True
        self._state.exhausted_dimension = dimension
        if self.emit_event is not None:
            self.emit_event(
                "budget/exhausted",
                {"dimension": dimension.value, "limit": limit, "observed": observed},
            )
