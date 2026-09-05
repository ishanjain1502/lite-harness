"""Preset loading tests."""

from __future__ import annotations

from liteness.context_builder import ContextBuilder
from liteness.presets import load_preset
from liteness.session import Message, Session


def test_video_editor_preset_has_system_prompt() -> None:
    preset = load_preset("video_editor")
    assert preset.system_prompt is not None
    assert "analyze_video" in preset.system_prompt


def test_coder_preset_has_no_system_prompt() -> None:
    preset = load_preset("coder")
    assert preset.system_prompt is None


def test_context_builder_prepends_system_prompt() -> None:
    session = Session()
    session.append("user/message", {"content": "hello"}, turn=1, step=0)

    built = ContextBuilder().build(
        session,
        tools=[],
        system_prompt="You are a video editor.",
    )

    assert len(built.messages) == 2
    assert built.messages[0] == Message(role="system", content="You are a video editor.")
    assert built.messages[1].role == "user"
