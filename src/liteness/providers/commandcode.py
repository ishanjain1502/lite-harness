"""Command Code Provider API — OpenAI-compatible chat completions preset."""

from __future__ import annotations

import os

from liteness.providers.openai import OpenAILLMProvider

COMMANDCODE_BASE_URL = "https://api.commandcode.ai/provider/v1"


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
