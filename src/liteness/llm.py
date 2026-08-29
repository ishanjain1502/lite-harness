"""LLM transport: streaming, tool-call drafts, mock and OpenAI-compatible providers."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

from liteness.session import Message


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCallDraft:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LlmRequest:
    session_id: str
    turn: int
    step: int
    messages: list[Message]
    tools: list[ToolSchema]
    model: str = "mock"
    base_url: str | None = None
    api_key: str | None = None


@dataclass
class LlmChunk:
    content_delta: str = ""
    tool_call_deltas: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False


@dataclass
class LlmStreamResult:
    content: str
    tool_calls: list[ToolCallDraft]
    chunks: list[LlmChunk]


class LLMProvider(ABC):
    @abstractmethod
    def stream(self, request: LlmRequest) -> Iterator[LlmChunk]:
        """Yield incremental chunks; final chunk has done=True."""

    def generate(self, request: LlmRequest) -> LlmStreamResult:
        """Non-streaming convenience — consumes stream internally."""
        chunks = list(self.stream(request))
        return self._assemble(chunks)

    def _assemble(self, chunks: list[LlmChunk]) -> LlmStreamResult:
        content_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}

        for chunk in chunks:
            if chunk.content_delta:
                content_parts.append(chunk.content_delta)
            for delta in chunk.tool_call_deltas:
                idx = delta.get("index", 0)
                entry = tool_calls_by_index.setdefault(
                    idx,
                    {"id": "", "name": "", "arguments": ""},
                )
                if "id" in delta and delta["id"]:
                    entry["id"] = delta["id"]
                if "name" in delta and delta["name"]:
                    entry["name"] = delta["name"]
                if "arguments" in delta and delta["arguments"]:
                    entry["arguments"] += delta["arguments"]

        tool_calls: list[ToolCallDraft] = []
        for entry in tool_calls_by_index.values():
            if not entry["name"]:
                continue
            args_raw = entry["arguments"] or "{}"
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCallDraft(
                    call_id=entry["id"] or f"call_{len(tool_calls)}",
                    name=entry["name"],
                    arguments=args,
                )
            )

        return LlmStreamResult(
            content="".join(content_parts),
            tool_calls=tool_calls,
            chunks=chunks,
        )


@dataclass
class MockStep:
    """One scripted model response for a given step number."""

    step: int
    content: str = ""
    tool_calls: list[ToolCallDraft] = field(default_factory=list)

    def stream_chunks(self) -> list[LlmChunk]:
        chunks: list[LlmChunk] = []
        if self.content:
            mid = max(1, len(self.content) // 2)
            chunks.append(LlmChunk(content_delta=self.content[:mid]))
            chunks.append(LlmChunk(content_delta=self.content[mid:]))
        for i, call in enumerate(self.tool_calls):
            chunks.append(
                LlmChunk(
                    tool_call_deltas=[
                        {
                            "index": i,
                            "id": call.call_id,
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        }
                    ]
                )
            )
        chunks.append(LlmChunk(done=True))
        return chunks


class MockLLMProvider(LLMProvider):
    """Deterministic replay LLM for tests and offline demos."""

    def __init__(self, steps: list[MockStep] | None = None) -> None:
        self._steps = {s.step: s for s in (steps or [])}
        self._call_count = 0

    def add_step(self, step: MockStep) -> None:
        self._steps[step.step] = step

    def stream(self, request: LlmRequest) -> Iterator[LlmChunk]:
        self._call_count += 1
        step_num = request.step
        scripted = self._steps.get(step_num)
        if scripted is None:
            yield LlmChunk(content_delta="I don't know how to help.", done=True)
            return
        for chunk in scripted.stream_chunks():
            yield chunk


class OpenAILLMProvider(LLMProvider):
    """OpenAI-compatible chat completions with streaming."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model

    def stream(self, request: LlmRequest) -> Iterator[LlmChunk]:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI provider requires httpx: pip install 'lite-ness[openai]'"
            ) from exc

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        body = {
            "model": request.model or self.model,
            "messages": [m.to_dict() for m in request.messages],
            "stream": True,
        }
        if request.tools:
            body["tools"] = [t.to_openai_tool() for t in request.tools]

        with httpx.Client(timeout=120.0) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        yield LlmChunk(done=True)
                        return
                    payload = json.loads(data)
                    choice = payload["choices"][0]
                    delta = choice.get("delta", {})
                    content = delta.get("content") or ""
                    tool_deltas: list[dict[str, Any]] = []
                    if "tool_calls" in delta:
                        for tc in delta["tool_calls"]:
                            tool_deltas.append(
                                {
                                    "index": tc.get("index", 0),
                                    "id": tc.get("id"),
                                    "name": (tc.get("function") or {}).get("name"),
                                    "arguments": (tc.get("function") or {}).get(
                                        "arguments", ""
                                    ),
                                }
                            )
                    yield LlmChunk(
                        content_delta=content,
                        tool_call_deltas=tool_deltas,
                        done=choice.get("finish_reason") is not None,
                    )
