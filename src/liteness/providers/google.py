"""Google Gemini provider via google-genai SDK."""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

from liteness.llm import LLMProvider, LlmChunk, LlmRequest, LlmUsage, ToolSchema
from liteness.session import Message
from liteness.types import LlmError


def messages_to_gemini_contents(messages: list[Message]) -> tuple[list[Any], str | None]:
    """Convert harness Message list to Gemini Content objects and system instruction."""
    from google.genai import types

    system_parts: list[str] = []
    contents: list[Any] = []
    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
        elif msg.role == "user":
            contents.append(
                types.Content(role="user", parts=[types.Part(text=msg.content)])
            )
        elif msg.role == "assistant":
            contents.append(
                types.Content(role="model", parts=[types.Part(text=msg.content)])
            )
        elif msg.role == "tool":
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=msg.name or "tool",
                                response={"output": msg.content},
                                id=msg.tool_call_id,
                            )
                        )
                    ],
                )
            )
    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return contents, system_instruction


def tools_to_gemini(tools: list[ToolSchema]) -> list[Any]:
    """Convert ToolSchema list to Gemini Tool objects."""
    if not tools:
        return []
    from google.genai import types

    declarations = [
        types.FunctionDeclaration(
            name=t.name,
            description=t.description,
            parameters=t.parameters,
        )
        for t in tools
    ]
    return [types.Tool(function_declarations=declarations)]


def _llm_error_from_google(exc: Exception) -> LlmError:
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    message = str(exc)
    if status_code == 429 or "429" in message:
        return LlmError("LLM_RATE_LIMITED", message, retryable=True)
    if status_code in (503, 502, 504) or any(
        s in message for s in ("503", "502", "504", "UNAVAILABLE")
    ):
        return LlmError("LLM_UNAVAILABLE", message, retryable=True)
    if status_code in (401, 403, 400):
        return LlmError("LLM_ERROR", message, retryable=False)
    if "timeout" in message.lower() or "deadline" in message.lower():
        return LlmError("LLM_TIMEOUT", message, retryable=True)
    return LlmError("LLM_ERROR", message, retryable=False)


class GoogleLLMProvider(LLMProvider):
    """Native Gemini adapter using google-genai."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or ""
        )
        self.model = model
        self._last_usage: LlmUsage | None = None
        self._client: Any = None

    @property
    def last_usage(self) -> LlmUsage | None:
        return self._last_usage

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "Google provider requires google-genai: pip install 'lite-ness[google]'"
            ) from exc
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set")
        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def stream(self, request: LlmRequest) -> Iterator[LlmChunk]:
        from google.genai import types

        client = self._get_client()
        model = request.model or self.model
        contents, system_instruction = messages_to_gemini_contents(request.messages)
        gemini_tools = tools_to_gemini(request.tools)

        config_kwargs: dict[str, Any] = {}
        if gemini_tools:
            config_kwargs["tools"] = gemini_tools
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        self._last_usage = None
        tool_index = 0
        try:
            stream = client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )
            for chunk in stream:
                usage_meta = getattr(chunk, "usage_metadata", None)
                if usage_meta is not None:
                    input_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
                    output_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0
                    total = getattr(usage_meta, "total_token_count", None)
                    self._last_usage = LlmUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total if total is not None else input_tokens + output_tokens,
                    )

                text = getattr(chunk, "text", None)
                if text:
                    yield LlmChunk(content_delta=text)

                candidates = getattr(chunk, "candidates", None) or []
                for candidate in candidates:
                    content = getattr(candidate, "content", None)
                    if content is None:
                        continue
                    parts = getattr(content, "parts", None) or []
                    for part in parts:
                        function_call = getattr(part, "function_call", None)
                        if function_call is None:
                            continue
                        name = getattr(function_call, "name", "") or ""
                        args = getattr(function_call, "args", None) or {}
                        call_id = getattr(function_call, "id", None) or f"call_{name}"
                        yield LlmChunk(
                            tool_call_deltas=[
                                {
                                    "index": tool_index,
                                    "id": call_id,
                                    "name": name,
                                    "arguments": json.dumps(dict(args)),
                                }
                            ]
                        )
                        tool_index += 1

            yield LlmChunk(done=True)
        except LlmError:
            raise
        except RuntimeError:
            raise
        except Exception as exc:
            raise _llm_error_from_google(exc) from exc
