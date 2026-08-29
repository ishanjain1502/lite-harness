"""AgentLoop — owns turn/step lifecycle and orchestrates all components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from liteness.agent import (
    Agent,
    AgentContext,
    ErrorStop,
    FinalAnswerStop,
    InjectContextAction,
    PolicyHaltStop,
    ToolCallsAction,
)
from liteness.llm import LLMProvider, LlmRequest
from liteness.session import Session
from liteness.tools import ToolRegistry


@dataclass
class TurnResult:
    session_id: str
    turn: int
    status: Literal["completed", "stopped", "error"]
    final_output: str | None = None
    steps_run: int = 0


@dataclass
class LoopConfig:
    max_steps_per_turn: int = 10
    allowed_tools: list[str] | None = None
    model: str = "mock"


class AgentLoop:
    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        agent: Agent | None = None,
        config: LoopConfig | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.agent = agent or Agent()
        self.config = config or LoopConfig()

    def run_turn(self, session: Session, user_message: str) -> TurnResult:
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
        status: Literal["completed", "stopped", "error"] = "completed"
        steps_run = 0

        for step in range(1, self.config.max_steps_per_turn + 1):
            session.set_step(step)
            steps_run = step
            session.append("step/start", {"step": step}, turn=turn, step=step)

            messages = session.derive_messages()
            request = LlmRequest(
                session_id=session.session_id,
                turn=turn,
                step=step,
                messages=messages,
                tools=schemas,
                model=self.config.model,
            )

            chunks = list(self.llm.stream(request))
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
                session.append(
                    "agent/decision",
                    {"kind": "final_answer", "rationale": decision.rationale},
                    turn=turn,
                    step=step,
                )
                session.append("step/end", {"step": step}, turn=turn, step=step)
                break

            if isinstance(decision, ErrorStop):
                status = "error"
                session.append(
                    "error",
                    {"code": decision.code, "message": decision.message},
                    turn=turn,
                    step=step,
                )
                session.append("step/end", {"step": step}, turn=turn, step=step)
                break

            if isinstance(decision, PolicyHaltStop):
                status = "stopped"
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
                for call in decision.calls or []:
                    session.append(
                        "tool/call",
                        {
                            "call_id": call.call_id,
                            "name": call.name,
                            "arguments": call.arguments,
                        },
                        turn=turn,
                        step=step,
                    )
                    result = self.tools.execute(
                        call.call_id, call.name, call.arguments
                    )
                    session.append(
                        "tool/result",
                        result.to_event_payload(),
                        turn=turn,
                        step=step,
                    )
                session.append("step/end", {"step": step}, turn=turn, step=step)
                continue
        else:
            status = "stopped"
            session.append(
                "policy/stop",
                {"reason": "max_steps_per_turn"},
                turn=turn,
                step=steps_run,
            )

        session.append("turn/end", {"turn": turn, "status": status})
        return TurnResult(
            session_id=session.session_id,
            turn=turn,
            status=status,
            final_output=final_output,
            steps_run=steps_run,
        )
