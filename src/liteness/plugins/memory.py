"""Memory plugin — durable per-project facts via explicit store/search tools."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from liteness.context import Context
from liteness.plugins.base import PluginConfigError
from liteness.tools import ToolDefinition, ToolResult


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Memory:
    id: str
    content: str
    created_at: str
    updated_at: str


class MemoryStore(Protocol):
    def store(self, project_id: str, content: str) -> Memory: ...

    def search(self, project_id: str, query: str, k: int) -> list[Memory]: ...


class JSONLMemoryStore:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, project_id: str) -> Path:
        safe_id = project_id.replace("/", "_").replace("\\", "_")
        return self._base_dir / f"{safe_id}.jsonl"

    def store(self, project_id: str, content: str) -> Memory:
        now = _utc_now().isoformat()
        memory = Memory(
            id=uuid.uuid4().hex[:12],
            content=content,
            created_at=now,
            updated_at=now,
        )
        path = self._path_for(project_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(memory), ensure_ascii=False) + "\n")
        return memory

    def search(self, project_id: str, query: str, k: int) -> list[Memory]:
        path = self._path_for(project_id)
        if not path.exists():
            return []

        query_lower = query.lower()
        tokens = [t for t in query_lower.split() if t]
        scored: list[tuple[int, Memory]] = []

        with path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                data = json.loads(stripped)
                memory = Memory(**data)
                content_lower = memory.content.lower()
                score = sum(1 for token in tokens if token in content_lower)
                if score > 0 or not tokens:
                    scored.append((score, memory))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [memory for _, memory in scored[:k]]


class MemoryPlugin:
    name = "memory"

    def install(self, ctx: Context, config: dict[str, Any]) -> None:
        project_id = config.get("project_id")
        if not isinstance(project_id, str) or not project_id:
            raise PluginConfigError("MemoryPlugin requires config.project_id")

        store_path = config.get("store_path", "./memory")
        store = JSONLMemoryStore(Path(store_path))
        ctx.services["memory_store"] = store
        ctx.services["memory_project_id"] = project_id

        session_append: Callable[[str, dict[str, Any]], None] | None = config.get(
            "session_append"
        )

        def store_handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
            content = arguments.get("content")
            if not isinstance(content, str) or not content.strip():
                return ToolResult(
                    call_id=call_id,
                    name="memory.store",
                    content="content must be a non-empty string",
                    is_error=True,
                    error_code="INVALID_ARGS",
                )
            memory = store.store(project_id, content.strip())
            if session_append is not None:
                session_append(
                    "memory/write",
                    {
                        "memory_id": memory.id,
                        "content": memory.content,
                        "project_id": project_id,
                    },
                )
            return ToolResult(
                call_id=call_id,
                name="memory.store",
                content=json.dumps(asdict(memory), ensure_ascii=False),
            )

        def search_handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                return ToolResult(
                    call_id=call_id,
                    name="memory.search",
                    content="query must be a non-empty string",
                    is_error=True,
                    error_code="INVALID_ARGS",
                )
            k = arguments.get("k", 5)
            if not isinstance(k, int) or k < 1:
                k = 5
            results = store.search(project_id, query.strip(), k)
            payload = [asdict(memory) for memory in results]
            return ToolResult(
                call_id=call_id,
                name="memory.search",
                content=json.dumps(payload, ensure_ascii=False),
            )

        ctx.register_tool(
            ToolDefinition(
                name="memory.store",
                description="Store a durable fact in long-term project memory.",
                parameters={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Fact to remember across sessions",
                        },
                    },
                    "required": ["content"],
                },
                handler=store_handler,
            )
        )
        ctx.register_tool(
            ToolDefinition(
                name="memory.search",
                description="Search project memory for relevant stored facts.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Maximum results to return",
                        },
                    },
                    "required": ["query"],
                },
                handler=search_handler,
            )
        )

    def uninstall(self, ctx: Context) -> None:
        ctx.services.pop("memory_store", None)
        ctx.services.pop("memory_project_id", None)
