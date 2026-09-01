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
from liteness.llm import LLMProvider, LlmChunk, LlmRequest
from liteness.session import Session
from liteness.tools import DispatchCall, ToolRegistry
from liteness.types import (
    CancelToken,
    CancelledError,
    LlmError,
    StopReason,
    stop_reason_to_status,
)


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


class AgentLoop:
    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        agent: Agent | None = None,
        config: LoopConfig | None = None,
        context_builder: ContextBuilder | None = None,
        event_sink: Callable[[str, dict], None] | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.agent = agent or Agent()
        self.config = config or LoopConfig()
        self.context_builder = context_builder or ContextBuilder()
        self.event_sink = event_sink

    def run_turn(
        self,
        session: Session,
        user_message: str,
        cancel: CancelToken | None = None,
    ) -> TurnResult:
        if (
            self.config.max_turns_per_session is not None
            and session._turn >= self.config.max_turns_per_session
        ):
            return self._finish_without_turn(session, StopReason.TURN_LIMIT)

        if cancel is not None and cancel.cancelled:
            return self._finish_without_turn(session, StopReason.USER_CANCELLED)

        turn = session._turn + 1
        session.set_turn(turn)
        session.append("turn/start", {"turn": turn})

        session.append(
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
                if cancel is not None and cancel.cancelled:
                    stop_reason = StopReason.USER_CANCELLED
                    break

                session.set_step(step)
                steps_run = step
                session.append("step/start", {"step": step}, turn=turn, step=step)

                built = self.context_builder.build(
                    session,
                    tools=schemas,
                    memory_snapshot=None,
                )
                messages = built.messages
                if self._context_over_limit(messages):
                    stop_reason = StopReason.CONTEXT_LIMIT
                    session.append(
                        "policy/stop",
                        {"reason": "context_limit"},
                        turn=turn,
                        step=step,
                    )
                    session.append("step/end", {"step": step}, turn=turn, step=step)
                    break

                request = LlmRequest(
                    session_id=session.session_id,
                    turn=turn,
                    step=step,
                    messages=messages,
                    tools=schemas,
                    model=self.config.model,
                )

                try:
                    chunks = self._stream_with_retry(request, cancel)
                except CancelledError:
                    stop_reason = StopReason.USER_CANCELLED
                    session.append("step/end", {"step": step}, turn=turn, step=step)
                    break
                except LlmError as exc:
                    stop_reason = StopReason.LLM_ERROR
                    session.append(
                        "error",
                        {"code": exc.code, "message": exc.message},
                        turn=turn,
                        step=step,
                    )
                    session.append("step/end", {"step": step}, turn=turn, step=step)
                    break
                except Exception as exc:  # noqa: BLE001 — last-resort LLM failure
                    stop_reason = StopReason.LLM_ERROR
                    session.append(
                        "error",
                        {"code": "LLM_ERROR", "message": str(exc)},
                        turn=turn,
                        step=step,
                    )
                    session.append("step/end", {"step": step}, turn=turn, step=step)
                    break

                for chunk in chunks:
                    if chunk.content_delta or chunk.tool_call_deltas:
                        session.append(
                            "assistant/chunk",
                            {
                                "content_delta": chunk.content_delta,
                                "tool_call_deltas": chunk.tool_call_deltas,
                            },
                            turn=turn,
                            step=step,
                        )

                assembled = self.llm._assemble(chunks)
                session.append(
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
                    session.append(
                        "agent/decision",
                        {"kind": "final_answer", "rationale": decision.rationale},
                        turn=turn,
                        step=step,
                    )
                    session.append("step/end", {"step": step}, turn=turn, step=step)
                    break

                if isinstance(decision, ErrorStop):
                    stop_reason = StopReason.AGENT_ERROR
                    session.append(
                        "error",
                        {"code": decision.code, "message": decision.message},
                        turn=turn,
                        step=step,
                    )
                    session.append("step/end", {"step": step}, turn=turn, step=step)
                    break

                if isinstance(decision, PolicyHaltStop):
                    stop_reason = self._policy_stop_reason(decision.reason)
                    session.append(
                        "policy/stop",
                        {"reason": decision.reason},
                        turn=turn,
                        step=step,
                    )
                    session.append("step/end", {"step": step}, turn=turn, step=step)
                    break

                if isinstance(decision, InjectContextAction):
                    session.append(
                        "inject/context",
                        {"content": decision.content},
                        turn=turn,
                        step=step,
                    )
                    session.append(
                        "agent/decision",
                        {"kind": "inject_context", "rationale": decision.rationale},
                        turn=turn,
                        step=step,
                    )
                    session.append("step/end", {"step": step}, turn=turn, step=step)
                    continue

                if isinstance(decision, ToolCallsAction):
                    session.append(
                        "agent/decision",
                        {
                            "kind": "tool_calls",
                            "calls": [
                                {
                                    "call_id": c.call_id,
                                    "name": c.name,
                                    "arguments": c.arguments,
                                }
                                for c in (decision.calls or [])
                            ],
                            "rationale": decision.rationale,
                        },
                        turn=turn,
                        step=step,
                    )
                    self._execute_tool_batch(
                        session,
                        decision.calls or [],
                        turn=turn,
                        step=step,
                        cancel=cancel,
                    )
                    session.append("step/end", {"step": step}, turn=turn, step=step)
                    continue
            else:
                stop_reason = StopReason.STEP_LIMIT
                session.append(
                    "policy/stop",
                    {"reason": "max_steps_per_turn"},
                    turn=turn,
                    step=steps_run,
                )
        except CancelledError:
            stop_reason = StopReason.USER_CANCELLED

        status = stop_reason_to_status(stop_reason)
        session.append(
            "turn/end",
            {"turn": turn, "status": status, "stop_reason": stop_reason.value},
        )
        return TurnResult(
            session_id=session.session_id,
            turn=turn,
            status=status,
            stop_reason=stop_reason,
            final_output=final_output,
            steps_run=steps_run,
        )

    def _finish_without_turn(self, session: Session, reason: StopReason) -> TurnResult:
        status = stop_reason_to_status(reason)
        turn = session._turn
        session.append(
            "turn/end",
            {"turn": turn, "status": status, "stop_reason": reason.value},
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
        request: LlmRequest,
        cancel: CancelToken | None,
    ) -> list[LlmChunk]:
        last_error: LlmError | None = None
        attempts = max(1, self.config.llm_max_retries)

        for attempt in range(attempts):
            if cancel is not None and cancel.cancelled:
                raise CancelledError()

            try:
                return list(self.llm.stream(request))
            except CancelledError:
                raise
            except LlmError as exc:
                last_error = exc
                if not exc.retryable or attempt == attempts - 1:
                    raise
            except Exception as exc:
                wrapped = LlmError("LLM_ERROR", str(exc), retryable=False)
                last_error = wrapped
                raise wrapped from exc

            if attempt < attempts - 1:
                time.sleep(self.config.llm_retry_delay_s)

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
    ) -> None:
        for call in calls:
            session.append(
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
        for result in results:
            session.append(
                "tool/result",
                result.to_event_payload(),
                turn=turn,
                step=step,
            )