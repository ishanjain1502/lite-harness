"""Session LLM helpers for video plugin tools."""

from __future__ import annotations

from liteness.context import Context
from liteness.llm import LLMProvider, LlmRequest
from liteness.session import Message


def inject_session_llm(ctx: Context, llm: LLMProvider, model: str) -> None:
    """Expose the harness LLM to plugins that need one-shot completions."""
    ctx.services["llm"] = llm
    ctx.services["llm_model"] = model


def session_llm_complete(ctx: Context, *, system: str, user: str) -> str:
    """Run a single text completion using the session's configured provider."""
    llm = ctx.services.get("llm")
    if llm is None:
        raise RuntimeError(
            "session LLM not available — video tools use the same provider as "
            "liteness run --provider"
        )
    model = ctx.services.get("llm_model") or "mock"
    result = llm.generate(
        LlmRequest(
            session_id="video-plugin",
            turn=0,
            step=0,
            messages=[
                Message(role="system", content=system),
                Message(role="user", content=user),
            ],
            tools=[],
            model=model,
        )
    )
    return result.content
