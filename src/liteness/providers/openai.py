"""OpenAI-compatible chat completions provider."""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

from liteness.llm import LLMProvider, LlmChunk, LlmRequest, LlmUsage
from liteness.session import Message
from liteness.types import LlmError


def _llm_error_from_status(status_code: int, message: str) -> LlmError:
    retryable = status_code in (429, 502, 503, 504)
    if status_code == 429:
        code = "LLM_RATE_LIMITED"
    elif status_code >= 500:
        code = "LLM_UNAVAILABLE"
    else:
        code = "LLM_ERROR"
    return LlmError(code, message, retryable=retryable)


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
        self._last_usage: LlmUsage | None = None

    @property
    def last_usage(self) -> LlmUsage | None:
        return self._last_usage

    def _messages_to_api(self, messages: list[Message]) -> list[dict[str, Any]]:
        return [m.to_dict() for m in messages]

    def stream(self, request: LlmRequest) -> Iterator[LlmChunk]:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI provider requires httpx: pip install 'lite-ness[openai]'"
            ) from exc

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        body: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": self._messages_to_api(request.messages),
            "stream": True,
        }
        if request.tools:
            body["tools"] = [t.to_openai_tool() for t in request.tools]

        self._last_usage = None
        try:
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
                    if response.status_code >= 400:
                        response.read()
                        raise _llm_error_from_status(
                            response.status_code,
                            response.text or f"HTTP {response.status_code}",
                        )
                    for line in response.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            yield LlmChunk(done=True)
                            return
                        payload = json.loads(data)
                        usage = payload.get("usage")
                        if usage:
                            input_tokens = usage.get("prompt_tokens", 0)
                            output_tokens = usage.get("completion_tokens", 0)
                            self._last_usage = LlmUsage(
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                total_tokens=usage.get(
                                    "total_tokens", input_tokens + output_tokens
                                ),
                            )
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
        except LlmError:
            raise
        except httpx.TimeoutException as exc:
            raise LlmError("LLM_TIMEOUT", str(exc), retryable=True) from exc
        except httpx.HTTPStatusError as exc:
            raise _llm_error_from_status(
                exc.response.status_code,
                exc.response.text or str(exc),
            ) from exc
        except httpx.HTTPError as exc:
            raise LlmError("LLM_UNAVAILABLE", str(exc), retryable=True) from exc
