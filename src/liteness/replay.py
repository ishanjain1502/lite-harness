"""Replay a persisted session log without calling a live LLM."""

from __future__ import annotations

import json
from typing import Iterator

from liteness.llm import LLMProvider, LlmChunk, LlmRequest
from liteness.session import Session, SessionEvent


class ReplayLLMProvider(LLMProvider):
    """Replays assistant/message payloads from a session log by turn+step."""

    def __init__(self, events: list[SessionEvent]) -> None:
        self._by_step: dict[tuple[int, int], dict] = {}
        for event in events:
            if event.type == "assistant/message":
                self._by_step[(event.turn, event.step)] = event.payload

    @classmethod
    def from_session(cls, session: Session) -> ReplayLLMProvider:
        return cls(session.events)

    @classmethod
    def from_jsonl(cls, path: str) -> ReplayLLMProvider:
        return cls(Session.load_from_jsonl(path, recover=False).events)

    def stream(self, request: LlmRequest) -> Iterator[LlmChunk]:
        payload = self._by_step.get((request.turn, request.step))
        if payload is None:
            yield LlmChunk(
                content_delta="[replay] no recorded assistant/message for this step",
                done=True,
            )
            return

        content = payload.get("content", "")
        if content:
            yield LlmChunk(content_delta=content)

        for index, tool_call in enumerate(payload.get("tool_calls", [])):
            yield LlmChunk(
                tool_call_deltas=[
                    {
                        "index": index,
                        "id": tool_call.get("call_id", ""),
                        "name": tool_call.get("name", ""),
                        "arguments": json.dumps(tool_call.get("arguments", {})),
                    }
                ]
            )

        yield LlmChunk(done=True)
