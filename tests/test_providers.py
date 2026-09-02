"""Tests for multi-provider LLM factory and adapters (no live API keys)."""

from __future__ import annotations

import json
import os
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from liteness.llm import LlmRequest, ToolSchema
from liteness.providers import default_model, list_providers, resolve_provider
from liteness.providers.commandcode import (
    COMMANDCODE_BASE_URL,
    CommandCodeLLMProvider,
    messages_to_openai_api,
)
from liteness.providers.google import (
    GoogleLLMProvider,
    messages_to_gemini_contents,
    tools_to_gemini,
)
from liteness.providers.openai import OpenAILLMProvider
from liteness.session import Message
from liteness.types import LlmError


def test_list_providers() -> None:
    assert list_providers() == ["commandcode", "google", "openai"]


def test_resolve_provider_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        resolve_provider("anthropic")


def test_default_models() -> None:
    assert default_model("openai") == "gpt-4o-mini"
    assert default_model("google") == "gemini-2.0-flash"
    assert default_model("commandcode") == "deepseek/deepseek-v4-flash"


def test_resolve_openai_uses_default_model() -> None:
    provider = resolve_provider("openai")
    assert isinstance(provider, OpenAILLMProvider)
    assert provider.model == "gpt-4o-mini"


def test_resolve_google_custom_model() -> None:
    provider = resolve_provider("google", model="gemini-1.5-pro")
    assert isinstance(provider, GoogleLLMProvider)
    assert provider.model == "gemini-1.5-pro"


def test_commandcode_config() -> None:
    with patch.dict(os.environ, {"COMMANDCODE_API_KEY": "cc-test-key"}, clear=False):
        provider = CommandCodeLLMProvider()
    assert provider.base_url == COMMANDCODE_BASE_URL
    assert provider.api_key == "cc-test-key"
    assert provider.model == "gpt-4o-mini"


def test_commandcode_messages_include_tool_calls_before_tool_results() -> None:
    messages = [
        Message(role="user", content="List files"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "call_id": "call_abc",
                    "name": "read_file",
                    "arguments": {"path": "."},
                }
            ],
        ),
        Message(
            role="tool",
            content="not a file: .",
            tool_call_id="call_abc",
            name="read_file",
        ),
    ]
    api_messages = messages_to_openai_api(messages)
    assert api_messages[1]["role"] == "assistant"
    assert api_messages[1]["tool_calls"][0]["id"] == "call_abc"
    assert api_messages[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert json.loads(api_messages[1]["tool_calls"][0]["function"]["arguments"]) == {
        "path": "."
    }
    assert api_messages[2]["role"] == "tool"
    assert api_messages[2]["tool_call_id"] == "call_abc"


def test_openai_provider_omits_tool_calls_from_assistant_messages() -> None:
    provider = OpenAILLMProvider(api_key="k")
    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "call_id": "call_abc",
                    "name": "read_file",
                    "arguments": {"path": "."},
                }
            ],
        ),
    ]
    api_messages = provider._messages_to_api(messages)
    assert "tool_calls" not in api_messages[0]


def test_google_missing_sdk_raises_on_client() -> None:
    provider = GoogleLLMProvider(api_key="test-key")

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "google" or name.startswith("google."):
            raise ImportError("No module named 'google'")
        return orig_import(name, *args, **kwargs)

    import builtins

    orig_import = builtins.__import__
    with patch.object(builtins, "__import__", side_effect=_fake_import):
        with pytest.raises(RuntimeError, match="google-genai"):
            provider._get_client()


def test_messages_to_gemini_contents() -> None:
    pytest.importorskip("google.genai")
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there"),
        Message(
            role="tool",
            content='{"result": "ok"}',
            tool_call_id="call_1",
            name="read_file",
        ),
    ]
    contents = messages_to_gemini_contents(messages)
    assert len(contents) == 3
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "Hello"
    assert contents[1].role == "model"
    assert contents[2].parts[0].function_response.name == "read_file"
    assert contents[2].parts[0].function_response.id == "call_1"


def test_tools_to_gemini() -> None:
    pytest.importorskip("google.genai")
    tools = [
        ToolSchema(
            name="read_file",
            description="Read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )
    ]
    gemini_tools = tools_to_gemini(tools)
    assert len(gemini_tools) == 1
    decl = gemini_tools[0].function_declarations[0]
    assert decl.name == "read_file"
    assert decl.description == "Read a file"


class _FakeStreamResponse:
    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code
        self.text = ""

    def iter_lines(self) -> Iterator[str]:
        yield from self._lines

    def read(self) -> None:
        pass

    def __enter__(self) -> _FakeStreamResponse:
        return self

    def __exit__(self, *args: object) -> None:
        pass


def test_openai_stream_parses_content_and_tool_calls() -> None:
    lines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"read_file","arguments":"{\\"path\\":\\"a.txt\\"}"}}]}}]}',
        "data: [DONE]",
    ]
    fake_response = _FakeStreamResponse(lines)
    fake_client = MagicMock()
    fake_client.stream.return_value = fake_response
    fake_client.__enter__ = MagicMock(return_value=fake_client)
    fake_client.__exit__ = MagicMock(return_value=False)

    provider = OpenAILLMProvider(api_key="test-key")
    request = LlmRequest(
        session_id="s1",
        turn=1,
        step=1,
        messages=[Message(role="user", content="read")],
        tools=[],
    )

    with patch("httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value = fake_client
        chunks = list(provider.stream(request))

    assert any(c.content_delta == "Hello" for c in chunks)
    tool_chunks = [c for c in chunks if c.tool_call_deltas]
    assert tool_chunks
    assert tool_chunks[0].tool_call_deltas[0]["name"] == "read_file"
    assert chunks[-1].done is True


def test_openai_http_429_raises_retryable_llm_error() -> None:
    fake_response = _FakeStreamResponse([], status_code=429)
    fake_response.text = "rate limited"
    fake_client = MagicMock()
    fake_client.stream.return_value = fake_response

    provider = OpenAILLMProvider(api_key="test-key")
    request = LlmRequest(
        session_id="s1",
        turn=1,
        step=1,
        messages=[Message(role="user", content="hi")],
        tools=[],
    )

    with patch("httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value = fake_client
        with pytest.raises(LlmError) as exc_info:
            list(provider.stream(request))

    assert exc_info.value.code == "LLM_RATE_LIMITED"
    assert exc_info.value.retryable is True


def test_commandcode_stream_includes_tool_calls_in_request_body() -> None:
    lines = [
        'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]
    fake_response = _FakeStreamResponse(lines)
    fake_client = MagicMock()
    fake_client.stream.return_value = fake_response

    provider = CommandCodeLLMProvider(api_key="cc-key")
    request = LlmRequest(
        session_id="s1",
        turn=1,
        step=2,
        messages=[
            Message(role="user", content="List files"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "call_id": "call_abc",
                        "name": "read_file",
                        "arguments": {"path": "."},
                    }
                ],
            ),
            Message(
                role="tool",
                content="not a file: .",
                tool_call_id="call_abc",
                name="read_file",
            ),
        ],
        tools=[],
    )

    with patch("httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value = fake_client
        list(provider.stream(request))
        body = fake_client.stream.call_args[1]["json"]
        assert body["messages"][1]["tool_calls"][0]["id"] == "call_abc"


def test_commandcode_stream_uses_provider_base_url() -> None:
    lines = [
        'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]
    fake_response = _FakeStreamResponse(lines)
    fake_client = MagicMock()
    fake_client.stream.return_value = fake_response

    provider = CommandCodeLLMProvider(api_key="cc-key")
    request = LlmRequest(
        session_id="s1",
        turn=1,
        step=1,
        messages=[Message(role="user", content="hi")],
        tools=[],
    )

    with patch("httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value = fake_client
        list(provider.stream(request))
        call_kwargs = fake_client.stream.call_args
        assert call_kwargs[0][1] == f"{COMMANDCODE_BASE_URL}/chat/completions"
        headers = call_kwargs[1]["headers"]
        assert headers["Authorization"] == "Bearer cc-key"


def test_google_stream_mock_client() -> None:
    pytest.importorskip("google.genai")

    function_call = MagicMock()
    function_call.name = "read_file"
    function_call.args = {"path": "README.md"}
    function_call.id = "fc_1"

    part = MagicMock()
    part.function_call = function_call
    part.text = None

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    text_chunk = MagicMock()
    text_chunk.text = "I'll read the file."
    text_chunk.candidates = []
    text_chunk.usage_metadata = None

    tool_chunk = MagicMock()
    tool_chunk.text = None
    tool_chunk.candidates = [candidate]
    tool_chunk.usage_metadata = None

    usage_chunk = MagicMock()
    usage_chunk.text = None
    usage_chunk.candidates = []
    usage_meta = MagicMock()
    usage_meta.prompt_token_count = 10
    usage_meta.candidates_token_count = 5
    usage_meta.total_token_count = 15
    usage_chunk.usage_metadata = usage_meta

    mock_client = MagicMock()
    mock_client.models.generate_content_stream.return_value = iter(
        [text_chunk, tool_chunk, usage_chunk]
    )

    provider = GoogleLLMProvider(api_key="g-key")
    provider._client = mock_client

    request = LlmRequest(
        session_id="s1",
        turn=1,
        step=1,
        messages=[Message(role="user", content="read readme")],
        tools=[
            ToolSchema(
                name="read_file",
                description="Read",
                parameters={"type": "object", "properties": {}},
            )
        ],
        model="gemini-2.0-flash",
    )

    chunks = list(provider.stream(request))
    assembled = provider._assemble(chunks)

    assert "I'll read the file." in assembled.content
    assert len(assembled.tool_calls) == 1
    assert assembled.tool_calls[0].name == "read_file"
    assert assembled.tool_calls[0].arguments == {"path": "README.md"}
    assert provider.last_usage is not None
    assert provider.last_usage.total_tokens == 15
    assert chunks[-1].done is True

    call_kwargs = mock_client.models.generate_content_stream.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.0-flash"
    assert call_kwargs["config"].tools is not None


def test_openai_missing_httpx_raises() -> None:
    provider = OpenAILLMProvider(api_key="k")
    request = LlmRequest(
        session_id="s1",
        turn=1,
        step=1,
        messages=[],
        tools=[],
    )

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "httpx":
            raise ImportError("No module named 'httpx'")
        return orig_import(name, *args, **kwargs)

    import builtins

    orig_import = builtins.__import__
    with patch.object(builtins, "__import__", side_effect=_fake_import):
        with pytest.raises(RuntimeError, match="httpx"):
            list(provider.stream(request))
