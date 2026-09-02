"""LLM provider factory — resolve by name without plugin install() lifecycle."""

from __future__ import annotations

from liteness.llm import LLMProvider
from liteness.providers.commandcode import CommandCodeLLMProvider
from liteness.providers.google import GoogleLLMProvider
from liteness.providers.openai import OpenAILLMProvider

_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "google": "gemini-2.0-flash",
    "commandcode": "deepseek/deepseek-v4-flash",
}

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "openai": OpenAILLMProvider,
    "google": GoogleLLMProvider,
    "commandcode": CommandCodeLLMProvider,
}


def default_model(name: str) -> str:
    """Return the default model id for a provider name."""
    return _DEFAULT_MODELS.get(name.lower(), "mock")


def list_providers() -> list[str]:
    """Return registered provider names."""
    return sorted(_PROVIDERS.keys())


def resolve_provider(name: str, *, model: str | None = None) -> LLMProvider:
    """Instantiate an LLM provider by name."""
    key = name.lower()
    if key not in _PROVIDERS:
        known = ", ".join(list_providers())
        raise ValueError(f"Unknown provider {name!r}; choose from: {known}")
    cls = _PROVIDERS[key]
    resolved_model = model or _DEFAULT_MODELS[key]
    return cls(model=resolved_model)  # type: ignore[call-arg]
