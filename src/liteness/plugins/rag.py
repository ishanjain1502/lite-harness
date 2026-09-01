"""RAG plugin — markdown ingest and knowledge_search retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from liteness.context import Context
from liteness.plugins.base import PluginConfigError
from liteness.tools import ToolDefinition, ToolResult

DEFAULT_CHUNK_CHARS = 1200
DEFAULT_TOP_K = 5
MAX_RESULT_CHARS = 8000


@dataclass
class ChunkMetadata:
    source: str
    start_line: int
    end_line: int
    text: str


@dataclass
class ScoredChunk:
    text: str
    source: str
    start_line: int
    end_line: int
    score: float


class VectorStore(Protocol):
    def add(self, vector: list[float], metadata: ChunkMetadata) -> None: ...

    def search(self, vector: list[float], k: int) -> list[ScoredChunk]: ...

    def clear(self) -> None: ...

    @property
    def size(self) -> int: ...


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._entries: list[tuple[list[float], ChunkMetadata]] = []

    @property
    def size(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def add(self, vector: list[float], metadata: ChunkMetadata) -> None:
        self._entries.append((vector, metadata))

    def search(self, vector: list[float], k: int) -> list[ScoredChunk]:
        if not self._entries:
            return []
        scored = [
            (cosine_similarity(vector, stored), metadata)
            for stored, metadata in self._entries
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            ScoredChunk(
                text=metadata.text,
                source=metadata.source,
                start_line=metadata.start_line,
                end_line=metadata.end_line,
                score=score,
            )
            for score, metadata in scored[:k]
        ]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def hash_embed(text: str, dims: int = 64) -> list[float]:
    """Deterministic fake embedding from token hashes."""
    vector = [0.0] * dims
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = digest[0] % dims
        vector[index] += 1.0
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def chunk_markdown(text: str, source: str, max_chars: int) -> list[ChunkMetadata]:
    lines = text.splitlines()
    chunks: list[ChunkMetadata] = []
    current: list[str] = []
    start_line = 1

    def flush(end_line: int) -> None:
        nonlocal current, start_line
        if not current:
            return
        chunk_text = "\n".join(current).strip()
        if chunk_text:
            chunks.append(
                ChunkMetadata(
                    source=source,
                    start_line=start_line,
                    end_line=end_line,
                    text=chunk_text,
                )
            )
        current = []

    for index, line in enumerate(lines, start=1):
        if line.startswith("#") and current:
            flush(index - 1)
            start_line = index
        current.append(line)
        if sum(len(part) + 1 for part in current) >= max_chars:
            flush(index)
            start_line = index + 1

    if current:
        flush(len(lines))

    return chunks


class RAGPlugin:
    name = "rag"

    def __init__(self) -> None:
        self._store = InMemoryVectorStore()

    def install(self, ctx: Context, config: dict[str, Any]) -> None:
        index_path = config.get("index_path", ".liteness/rag-index.json")
        loaded = self.load_index(Path(index_path))
        ctx.services["rag_store"] = self._store
        ctx.services["rag_plugin"] = self
        ctx.services["rag_index_path"] = str(index_path)
        ctx.services["rag_index_loaded"] = loaded

        def search_handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                return ToolResult(
                    call_id=call_id,
                    name="knowledge_search",
                    content="query must be a non-empty string",
                    is_error=True,
                    error_code="INVALID_ARGS",
                )
            if self._store.size == 0:
                return ToolResult(
                    call_id=call_id,
                    name="knowledge_search",
                    content=(
                        "knowledge index is empty; run "
                        "'liteness index <path>' before searching"
                    ),
                    is_error=True,
                    error_code="INDEX_EMPTY",
                )
            k = arguments.get("k", DEFAULT_TOP_K)
            if not isinstance(k, int) or k < 1:
                k = DEFAULT_TOP_K
            vector = hash_embed(query.strip())
            hits = self._store.search(vector, k)
            payload = [asdict(hit) for hit in hits]
            content = json.dumps(payload, ensure_ascii=False)
            if len(content) > MAX_RESULT_CHARS:
                content = content[:MAX_RESULT_CHARS] + "..."
            return ToolResult(
                call_id=call_id,
                name="knowledge_search",
                content=content,
            )

        ctx.register_tool(
            ToolDefinition(
                name="knowledge_search",
                description=(
                    "Search indexed project documents and return relevant chunks "
                    "with source citations."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Maximum number of chunks to return",
                        },
                    },
                    "required": ["query"],
                },
                handler=search_handler,
            )
        )

    def uninstall(self, ctx: Context) -> None:
        ctx.services.pop("rag_store", None)
        ctx.services.pop("rag_plugin", None)

    def ingest_path(
        self,
        path: Path,
        *,
        max_chars: int = DEFAULT_CHUNK_CHARS,
    ) -> int:
        target = path.resolve()
        if not target.exists():
            raise PluginConfigError(f"path does not exist: {target}")

        self._store.clear()
        files: list[Path]
        if target.is_file():
            files = [target]
        else:
            files = sorted(target.rglob("*.md"))

        if not files:
            raise PluginConfigError(f"no markdown files found under {target}")

        count = 0
        for file_path in files:
            text = file_path.read_text(encoding="utf-8")
            rel_source = str(file_path)
            for chunk in chunk_markdown(text, rel_source, max_chars):
                vector = hash_embed(chunk.text)
                self._store.add(vector, chunk)
                count += 1
        return count

    def save_index(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "vector": vector,
                "metadata": asdict(metadata),
            }
            for vector, metadata in self._store._entries  # noqa: SLF001
        ]
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load_index(self, path: Path) -> int:
        if not path.exists():
            return 0
        data = json.loads(path.read_text(encoding="utf-8"))
        self._store.clear()
        for entry in data:
            metadata = ChunkMetadata(**entry["metadata"])
            self._store.add(entry["vector"], metadata)
        return self._store.size
