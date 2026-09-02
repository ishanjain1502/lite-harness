"""AgentLoop — owns turn/step lifecycle and orchestrates all components."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Literal

from liteness.agent import (
    Agent,
    AgentContext,
    ErrorStop,
    FinalAnswerStop,
    InjectContextAction,
    PolicyHaltStop,
    ToolCall,
    ToolCallsAction,
)
from liteness.context_builder import ContextBuilder
from liteness.errors import ErrorCode
from liteness.llm import LLMProvider, LlmChunk, LlmRequest
from liteness.policies.budget import BudgetConfig, BudgetManager
from liteness.policies.retry import DomainRetryConfig, RetryPolicy, RetryPolicyConfig
from liteness.session import Session, SessionEvent
from liteness.tools import DispatchCall, ToolRegistry
from liteness.types import (
    CancelToken,
    CancelledError,
    LlmError,
    StopReason,
    stop_reason_to_status,
)
from liteness.usage import estimate_cost_usd, estimate_message_tokens, estimate_tokens


@dataclass
class TurnResult:
    session_id: str
    turn: int
    status: Literal["completed", "stopped", "error"]
    stop_reason: StopReason
    final_output: str | None = None
    steps_run: int = 0


@dataclass
class LoopConfig:
    max_steps_per_turn: int = 10
    max_turns_per_session: int | None = None
    max_context_chars: int | None = None
    allowed_tools: list[str] | None = None
    model: str = "mock"
    default_tool_timeout_s: float = 30.0
    llm_max_retries: int = 3
    llm_retry_delay_s: float = 3.0
    tool_max_retries: int = 2
    tool_retry_delay_s: float = 0.0
    budget: BudgetConfig | None = None


SessionEventSink = Callable[[SessionEvent], None]


class AgentLoop:
    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        agent: Agent | None = None,
        config: LoopConfig | None = None,
        context_builder: ContextBuilder | None = None,
        event_sink: SessionEventSink | None = None,
        retry_policy: RetryPolicy | None = None,
        budget_manager: BudgetManager | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.agent = agent or Agent()
        self.config = config or LoopConfig()
        self.context_builder = context_builder or ContextBuilder()
        self.event_sink = event_sink
        self._retry_policy = retry_policy
        self._budget_manager = budget_manager
        self._turn_tokens = 0
        self._turn_cost = 0.0
        self._turn_retries = 0
        self._cancel_emitted = False

    def run_turn(
        self,
        session: Session,
        user_message: str,
        cancel: CancelToken | None = None,
    ) -> TurnResult:
        retry_policy = self._retry_policy or RetryPolicy(
            config=RetryPolicyConfig(
                llm=DomainRetryConfig(
                    max_attempts=self.config.llm_max_retries,
                    backoff_s=self.config.llm_retry_delay_s,
                ),
                tool=DomainRetryConfig(
                    max_attempts=self.config.tool_max_retries,
                    backoff_s=self.config.tool_retry_delay_s,
                ),
            ),
            emit_event=lambda event_type, payload: self._append(
                session, event_type, payload, turn=session._turn, step=session._step
            ),
        )
        budget = self._budget_manager
        if budget is None:
            budget = BudgetManager(
                config=self.config.budget,
                emit_event=lambda event_type, payload: self._append(
                    session, event_type, payload, turn=session._turn, step=session._step
                ),
            )
        elif budget.emit_event is None:
            budget.emit_event = lambda event_type, payload: self._append(
                session, event_type, payload, turn=session._turn, step=session._step
            )
        self._turn_tokens = 0
        self._turn_cost = 0.0
        self._turn_retries = 0
        self._cancel_emitted = False

        if (
            self.config.max_turns_per_session is not None
            and session._turn >= self.config.max_turns_per_session
        ):
            return self._finish_without_turn(session, StopReason.TURN_LIMIT, budget)

        if cancel is not None and cancel.cancelled:
            self._emit_cancel(session, turn=session._turn, step=0)
            return self._finish_without_turn(session, StopReason.USER_CANCELLED, budget)

        turn = session._turn + 1
        session.set_turn(turn)
        budget.reset_turn()
        self._append(session, "turn/start", {"turn": turn}, turn=turn, step=0)

        self._append(
            session,
            "user/message",
            {"content": user_message},
            turn=turn,
            step=0,
        )

        allowed = (
            self.config.allowed_tools
            if self.config.allowed_tools is not None
            else self.tools.names()
        )
        schemas = [s for s in self.tools.schemas() if s.name in allowed]
        final_output: str | None = None
        stop_reason = StopReason.COMPLETED
        steps_run = 0

        try:
            for step in range(1, self.config.max_steps_per_turn + 1):
                if not budget.can_continue():
                    stop_reason = StopReason.BUDGET_EXCEEDED
                    break

                if cancel is not None and cancel.cancelled:
                    self._emit_cancel(session, turn=turn, step=step)
                    stop_reason = StopReason.USER_CANCELLED
                    break

                session.set_step(step)
                steps_run = step
                self._append(session, "step/start", {"step": step}, turn=turn, step=step)

                built = self.context_builder.build(
                    session,
                    tools=schemas,
                    memory_snapshot=None,
                )
                messages = built.messages
                if self._context_over_limit(messages):
                    stop_reason = StopReason.CONTEXT_LIMIT
                    self._append(
                        session,
                        "policy/stop",
                        {"reason": "context_limit"},
                        turn=turn,
                        step=step,
                    )
                    self._append(session, "step/end", {"step": step}, turn=turn, step=step)
                    break

                request = LlmRequest(
                    session_id=session.session_id,
                    turn=turn,
                    step=step,
                    messages=messages,
                    tools=schemas,
                    model=self.config.model,
                )

                self._append(
                    session,
                    "llm/request",
                    {"model": self.config.model},
                    turn=turn,
                    step=step,
                )

                llm_started = time.monotonic()
                try:
                    chunks = self._stream_with_retry(
                        session, request, cancel, retry_policy, turn=turn, step=step
                    )
                except CancelledError:
                    self._emit_cancel(session, turn=turn, step=step)
                    stop_reason = StopReason.USER_CANCELLED
                    self._append(session, "step/end", {"step": step}, turn=turn, step=step)
                    break
                except LlmError as exc:
                    if exc.retryable and retry_policy.should_retry_llm(
                        exc.code, attempt=retry_policy.config.llm.max_attempts
                    ):
                        stop_reason = StopReason.RETRY_EXHAUSTED
                    else:
                        stop_reason = StopReason.LLM_ERROR
                    self._append(
                        session,
                        "error",
                        {"code": exc.code, "message": exc.message},
                        turn=turn,
                        step=step,
                    )
                    self._append(session, "step/end", {"step": step}, turn=turn, step=step)
                    break
                except Exception as exc:  # noqa: BLE001 — last-resort LLM failure
                    stop_reason = StopReason.LLM_ERROR
                    self._append(
                        session,
                        "error",
                        {"code": "LLM_ERROR", "message": str(exc)},
                        turn=turn,
                        step=step,
                    )
                    self._append(session, "step/end", {"step": step}, turn=turn, step=step)
                    break

                for chunk in chunks:
                    if chunk.content_delta or chunk.tool_call_deltas:
                        self._append(
                            session,
                            "assistant/chunk",
                            {
                                "content_delta": chunk.content_delta,
                                "tool_call_deltas": chunk.tool_call_deltas,
                            },
                            turn=turn,
                            step=step,
                        )

                assembled = self.llm._assemble(chunks)
                input_tokens = estimate_message_tokens(messages)
                output_tokens = estimate_tokens(assembled.content)
                cost_usd = estimate_cost_usd(input_tokens, output_tokens)
                duration_ms = (time.monotonic() - llm_started) * 1000.0
                budget.record_llm_usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                )
                self._turn_tokens += input_tokens + output_tokens
                self._turn_cost += cost_usd

                self._append(
                    session,
                    "llm/response",
                    {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost_usd": cost_usd,
                        "duration_ms": duration_ms,
                    },
                    turn=turn,
                    step=step,
                )
                self._append(
                    session,
                    "assistant/message",
                    {
                        "content": assembled.content,
                        "tool_calls": [
                            {
                                "call_id": tc.call_id,
                                "name": tc.name,
                                "arguments": tc.arguments,
                            }
                            for tc in assembled.tool_calls
                        ],
                    },
                    turn=turn,
                    step=step,
                )

                if not budget.can_continue():
                    stop_reason = StopReason.BUDGET_EXCEEDED
                    self._append(session, "step/end", {"step": step}, turn=turn, step=step)
                    break

                ctx = AgentContext(
                    session_id=session.session_id,
                    turn=turn,
                    step=step,
                    user_goal=user_message,
                    messages=messages,
                    tool_schemas=schemas,
                    allowed_tools=allowed,
                    llm_content=assembled.content,
                    tool_call_drafts=assembled.tool_calls,
                    max_steps=self.config.max_steps_per_turn,
                    steps_remaining=self.config.max_steps_per_turn - step,
                )
                decision = self.agent.decide(ctx)

                if isinstance(decision, FinalAnswerStop):
                    final_output = decision.content
                    stop_reason = StopReason.COMPLETED
                    self._append(
                        session,
                        "agent/decision",
                        {"kind": "final_answer", "rationale": decision.rationale},
                        turn=turn,
                        step=step,
                    )
                    self._append(session, "step/end", {"step": step}, turn=turn, step=step)
                    break

                if isinstance(decision, ErrorStop):
                    stop_reason = StopReason.AGENT_ERROR
                    self._append(
                        session,
                        "error",
                        {"code": decision.code, "message": decision.message},
                        turn=turn,
                        step=step,
                    )
                    self._append(session, "step/end", {"step": step}, turn=turn, step=step)
                    break

                if isinstance(decision, PolicyHaltStop):
                    stop_reason = self._policy_stop_reason(decision.reason)
                    self._append(
                        session,
                        "policy/stop",
                        {"reason": decision.reason},
                        turn=turn,
                        step=step,
                    )
                    self._append(session, "step/end", {"step": step}, turn=turn, step=step)
                    break

                if isinstance(decision, InjectContextAction):
                    self._append(
                        session,
                        "inject/context",
                        {"content": decision.content},
                        turn=turn,
                        step=step,
                    )
                    self._append(
                        session,
                        "agent/decision",
                        {"kind": "inject_context", "rationale": decision.rationale},
                        turn=turn,
                        step=step,
                    )
                    self._append(session, "step/end", {"step": step}, turn=turn, step=step)
                    continue

                if isinstance(decision, ToolCallsAction):
                    calls = decision.calls or []
                    if not budget.can_start_tool_calls(len(calls)):
                        stop_reason = StopReason.BUDGET_EXCEEDED
                        self._append(session, "step/end", {"step": step}, turn=turn, step=step)
                        break

                    self._append(
                        session,
                        "agent/decision",
                        {
                            "kind": "tool_calls",
                            "calls": [
                                {
                                    "call_id": c.call_id,
                                    "name": c.name,
                                    "arguments": c.arguments,
                                }
                                for c in calls
                            ],
                            "rationale": decision.rationale,
                        },
                        turn=turn,
                        step=step,
                    )
                    self._execute_tool_batch(
                        session,
                        calls,
                        turn=turn,
                        step=step,
                        cancel=cancel,
                        retry_policy=retry_policy,
                        budget=budget,
                    )
                    budget.record_tool_calls(len(calls))
                    self._append(session, "step/end", {"step": step}, turn=turn, step=step)
                    continue
            else:
                stop_reason = StopReason.STEP_LIMIT
                self._append(
                    session,
                    "policy/stop",
                    {"reason": "max_steps_per_turn"},
                    turn=turn,
                    step=steps_run,
                )
        except CancelledError:
            self._emit_cancel(session, turn=turn, step=session._step)
            stop_reason = StopReason.USER_CANCELLED

        status = stop_reason_to_status(stop_reason)
        self._append(
            session,
            "turn/end",
            {
                "turn": turn,
                "status": status,
                "stop_reason": stop_reason.value,
                "total_tokens": self._turn_tokens,
                "total_cost_usd": self._turn_cost,
                "retry_count": self._turn_retries,
            },
        )
        return TurnResult(
            session_id=session.session_id,
            turn=turn,
            status=status,
            stop_reason=stop_reason,
            final_output=final_output,
            steps_run=steps_run,
        )

    def _append(
        self,
        session: Session,
        event_type: str,
        payload: dict,
        *,
        turn: int | None = None,
        step: int | None = None,
        fsync: bool | None = None,
    ) -> SessionEvent:
        event = session.append(event_type, payload, turn=turn, step=step, fsync=fsync)
        if self.event_sink is not None:
            self.event_sink(event)
        return event

    def _emit_cancel(self, session: Session, *, turn: int, step: int) -> None:
        if self._cancel_emitted:
            return
        self._cancel_emitted = True
        self._append(
            session,
            "cancel/requested",
            {"source": "cancel_token"},
            turn=turn,
            step=step,
        )

    def _finish_without_turn(
        self,
        session: Session,
        reason: StopReason,
        budget: BudgetManager,
    ) -> TurnResult:
        status = stop_reason_to_status(reason)
        turn = session._turn
        self._append(
            session,
            "turn/end",
            {
                "turn": turn,
                "status": status,
                "stop_reason": reason.value,
                "total_tokens": self._turn_tokens,
                "total_cost_usd": self._turn_cost,
                "retry_count": self._turn_retries,
            },
        )
        return TurnResult(
            session_id=session.session_id,
            turn=turn,
            status=status,
            stop_reason=reason,
            final_output=None,
            steps_run=0,
        )

    def _context_over_limit(self, messages: list) -> bool:
        if self.config.max_context_chars is None:
            return False
        total = sum(len(m.content) for m in messages)
        return total > self.config.max_context_chars

    def _policy_stop_reason(self, reason: str) -> StopReason:
        if reason == "max_steps_per_turn":
            return StopReason.STEP_LIMIT
        return StopReason.POLICY_DENIED

    def _stream_with_retry(
        self,
        session: Session,
        request: LlmRequest,
        cancel: CancelToken | None,
        retry_policy: RetryPolicy,
        *,
        turn: int,
        step: int,
    ) -> list[LlmChunk]:
        last_error: LlmError | None = None
        cfg = retry_policy.config.llm

        for attempt in range(cfg.max_attempts):
            if cancel is not None and cancel.cancelled:
                raise CancelledError()

            try:
                return list(self.llm.stream(request))
            except CancelledError:
                raise
            except LlmError as exc:
                last_error = exc
                if (
                    not exc.retryable
                    or not retry_policy.should_retry_llm(exc.code, attempt=attempt + 1)
                ):
                    raise
                if attempt == cfg.max_attempts - 1:
                    raise
            except Exception as exc:
                wrapped = LlmError("LLM_ERROR", str(exc), retryable=False)
                last_error = wrapped
                raise wrapped from exc

            self._turn_retries += 1
            retry_policy.schedule_retry(
                "llm",
                attempt=attempt + 1,
                reason_code=last_error.code if last_error else ErrorCode.LLM_ERROR.value,
                call_id=None,
            )

        if last_error is not None:
            raise last_error
        raise LlmError("LLM_ERROR", "stream failed with no error detail")

    def _execute_tool_batch(
        self,
        session: Session,
        calls: list[ToolCall],
        *,
        turn: int,
        step: int,
        cancel: CancelToken | None,
        retry_policy: RetryPolicy,
        budget: BudgetManager,
    ) -> None:
        for call in calls:
            self._append(
                session,
                "tool/call",
                {
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": call.arguments,
                },
                turn=turn,
                step=step,
                fsync=True,
            )

        session.fsync_log()

        dispatch_calls = [
            DispatchCall(
                call_id=call.call_id,
                name=call.name,
                arguments=call.arguments,
            )
            for call in calls
        ]
        results = self.tools.dispatch(
            dispatch_calls,
            default_timeout_s=self.config.default_tool_timeout_s,
            cancel=cancel,
        )
        for call, result in zip(calls, results, strict=True):
            final = self._maybe_retry_tool(
                session,
                call,
                result,
                turn=turn,
                step=step,
                cancel=cancel,
                retry_policy=retry_policy,
            )
            self._append(
                session,
                "tool/result",
                final.to_event_payload(),
                turn=turn,
                step=step,
            )

    def _maybe_retry_tool(
        self,
        session: Session,
        call: ToolCall,
        result: ToolResult,
        *,
        turn: int,
        step: int,
        cancel: CancelToken | None,
        retry_policy: RetryPolicy,
    ) -> ToolResult:
        tool_def = self.tools.get(call.name)
        dispatch = DispatchCall(
            call_id=call.call_id,
            name=call.name,
            arguments=call.arguments,
        )
        attempt = 1
        current = result
        while current.is_error and retry_policy.should_retry_tool(
            current.error_code, tool=tool_def, attempt=attempt + 1
        ):
            self._turn_retries += 1
            retry_policy.schedule_retry(
                "tool",
                attempt=attempt,
                reason_code=current.error_code or "UNKNOWN_ERROR",
                call_id=call.call_id,
            )
            attempt += 1
            current = self.tools.dispatch(
                [dispatch],
                default_timeout_s=self.config.default_tool_timeout_s,
                cancel=cancel,
            )[0]
        return current
