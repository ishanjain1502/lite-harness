"""Command Code Provider API — OpenAI-compatible chat completions preset."""

from __future__ import annotations

import json
import os
from typing import Any

from liteness.providers.openai import OpenAILLMProvider
from liteness.session import Message

COMMANDCODE_BASE_URL = "https://api.commandcode.ai/provider/v1"


def messages_to_openai_api(messages: list[Message]) -> list[dict[str, Any]]:
    """Format messages for strict OpenAI-compatible providers (e.g. Command Code).

    Command Code rejects tool results unless the immediately preceding assistant
    message includes the matching tool_calls block.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc["call_id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("arguments", {})),
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
        else:
            out.append(msg.to_dict())
    return out


class CommandCodeLLMProvider(OpenAILLMProvider):
    """Thin wrapper over OpenAILLMProvider for Command Code's Provider API.

    Only OpenAI/open-source model IDs routed to /chat/completions are supported.
    Anthropic-shaped models on Command Code use /messages and are not supported here.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        super().__init__(
            api_key=api_key or os.environ.get("COMMANDCODE_API_KEY", ""),
            base_url=COMMANDCODE_BASE_URL,
            model=model,
        )

    def _messages_to_api(self, messages: list[Message]) -> list[dict[str, Any]]:
        return messages_to_openai_api(messages)
