"""Assemble model-visible context for each step."""

from __future__ import annotations

from dataclasses import dataclass

from liteness.llm import ToolSchema
from liteness.session import Message, Session


@dataclass
class BuiltContext:
    messages: list[Message]
    tools: list[ToolSchema]
    memory_snapshot: list[dict] | None = None
    token_estimate: int = 0


class ContextBuilder:
    def build(
        self,
        session: Session,
        *,
        tools: list[ToolSchema],
        memory_snapshot: list[dict] | None = None,
    ) -> BuiltContext:
        messages = session.derive_messages()
        token_estimate = sum(len(m.content) for m in messages)
        return BuiltContext(
            messages=messages,
            tools=tools,
            memory_snapshot=memory_snapshot,
            token_estimate=token_estimate,
        )
