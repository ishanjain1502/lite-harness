"""Token and cost estimation for providers without usage metadata."""

from __future__ import annotations

# Rough heuristic for mock / missing usage fields.
_CHARS_PER_TOKEN = 4
_COST_PER_1K_TOKENS_USD = 0.002


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_message_tokens(messages: list) -> int:
    total = 0
    for message in messages:
        total += estimate_tokens(getattr(message, "content", "") or "")
    return total


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    total = input_tokens + output_tokens
    return (total / 1000.0) * _COST_PER_1K_TOKENS_USD
