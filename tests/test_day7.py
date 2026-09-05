"""Day 7 tests — plugins, presets, memory, and RAG."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from liteness.context import Context
from liteness.harness import create_runtime, dispose_runtime
from liteness.llm import MockLLMProvider, MockStep, ToolCallDraft
from liteness.loop import AgentLoop
from liteness.plugins.base import PluginConfigError
from liteness.plugins.memory import JSONLMemoryStore
from liteness.plugins.rag import RAGPlugin, chunk_markdown, hash_embed
from liteness.plugins.telemetry import TelemetryPlugin
from liteness.presets import load_preset
from liteness.session import Session
from liteness.testing import registry_with_plugins


def test_load_builtin_presets() -> None:
    researcher = load_preset("researcher")
    coder = load_preset("coder")
    video_editor = load_preset("video_editor")
    assert researcher.name == "researcher"
    assert [p.name for p in researcher.plugins] == ["memory", "rag"]
    assert [p.name for p in coder.plugins] == ["filesystem", "terminal"]
    assert video_editor.name == "video_editor"
    assert [p.name for p in video_editor.plugins] == ["filesystem", "video"]
    assert video_editor.system_prompt is not None


def test_plugin_install_registers_tools() -> None:
    registry = registry_with_plugins("filesystem", "terminal")
    assert "read_file" in registry.names()
    assert "read_directory" in registry.names()
    assert "create_file" in registry.names()
    assert "edit_file" in registry.names()
    assert "delete_file" in registry.names()
    assert "delete_directory" in registry.names()
    assert "run_command" in registry.names()


def test_memory_store_and_search_cross_project(tmp_path: Path) -> None:
    store = JSONLMemoryStore(tmp_path / "memory")
    store.store("project-a", "User prefers Python")
    store.store("project-b", "User prefers Go")

    hits = store.search("project-a", "python prefers", k=5)
    assert len(hits) == 1
    assert "Python" in hits[0].content


def test_memory_plugin_emits_session_event(tmp_path: Path) -> None:
    session = Session()
    ctx = Context()
    events: list[str] = []

    def session_append(event_type: str, payload: dict) -> None:
        session.append(event_type, payload)
        events.append(event_type)

    from liteness.plugins.memory import MemoryPlugin

    MemoryPlugin().install(
        ctx,
        {
            "project_id": "demo",
            "store_path": str(tmp_path / "memory"),
            "session_append": session_append,
        },
    )

    result = ctx.tools.execute(
        "call_store",
        "memory.store",
        {"content": "Project is Liteness"},
    )
    assert result.is_error is False
    assert "memory/write" in events


def test_rag_ingest_search_and_persist_index(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    doc = docs / "architecture.md"
    doc.write_text(
        "# Session Recovery\n\nOrphan tool calls can be recovered.\n",
        encoding="utf-8",
    )

    plugin = RAGPlugin()
    count = plugin.ingest_path(docs)
    assert count >= 1

    index_path = tmp_path / "rag-index.json"
    plugin.save_index(index_path)

    fresh = RAGPlugin()
    loaded = fresh.load_index(index_path)
    assert loaded >= 1

    ctx = Context()
    fresh.install(ctx, {"index_path": str(index_path)})
    result = ctx.tools.execute(
        "call_search",
        "knowledge_search",
        {"query": "session recovery"},
    )
    assert result.is_error is False
    payload = json.loads(result.content)
    assert payload
    assert payload[0]["source"].endswith("architecture.md")


def test_create_runtime_researcher_preset(tmp_path: Path) -> None:
    session = Session()
    runtime = create_runtime(
        preset_name="researcher",
        session=session,
        project_id="demo",
    )
    try:
        names = runtime.ctx.tools.names()
        assert "memory.store" in names
        assert "knowledge_search" in names
        assert "read_file" not in names
    finally:
        dispose_runtime(runtime)


def test_telemetry_plugin_records_events() -> None:
    ctx = Context()
    TelemetryPlugin().install(ctx, {})
    ctx.emit("session/event", "tool/call", {"name": "read_file"})
    events = ctx.services["telemetry_events"]
    assert len(events) == 1
    assert events[0]["event"] == "tool/call"


def test_invalid_preset_raises() -> None:
    with pytest.raises(PluginConfigError):
        load_preset("does-not-exist")


def test_memory_requires_project_id() -> None:
    ctx = Context()
    from liteness.plugins.memory import MemoryPlugin

    with pytest.raises(PluginConfigError):
        MemoryPlugin().install(ctx, {})


def test_coder_preset_run_with_filesystem_and_terminal(tmp_path: Path) -> None:
    script = tmp_path / "note.txt"
    script.write_text("hello harness", encoding="utf-8")

    if Path.cwd().drive != tmp_path.drive:
        pytest.skip("shell cwd semantics differ on this platform")

    llm = MockLLMProvider(
        steps=[
            MockStep(
                step=1,
                tool_calls=[
                    ToolCallDraft(
                        call_id="read1",
                        name="read_file",
                        arguments={"path": str(script)},
                    )
                ],
            ),
            MockStep(step=2, content="read complete"),
        ]
    )

    session = Session()
    runtime = create_runtime(preset_name="coder", session=session, project_id="demo")
    try:
        loop = AgentLoop(llm=llm, tools=runtime.ctx.tools)
        result = loop.run_turn(session, "read the note")
        assert result.status == "completed"
        assert any(e.type == "tool/call" for e in session.events)
    finally:
        dispose_runtime(runtime)


def test_chunk_markdown_respects_headings() -> None:
    text = "# Title\n\nParagraph one.\n\n## Section\n\nParagraph two."
    chunks = chunk_markdown(text, "doc.md", max_chars=40)
    assert len(chunks) >= 2
    assert chunks[0].start_line == 1


def test_hash_embed_is_deterministic() -> None:
    assert hash_embed("session recovery") == hash_embed("session recovery")
