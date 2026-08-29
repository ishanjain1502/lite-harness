"""Tool registry and read_file — the first domain tool."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from liteness.llm import ToolSchema


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str
    is_error: bool = False
    error_code: str | None = None

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "content": self.content,
            "is_error": self.is_error,
            "error_code": self.error_code,
        }


ToolHandler = Callable[[str, dict[str, Any]], ToolResult]


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


def read_file_handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
    path_arg = arguments.get("path")
    if not isinstance(path_arg, str):
        return ToolResult(
            call_id=call_id,
            name="read_file",
            content="path must be a string",
            is_error=True,
            error_code="INVALID_ARGS",
        )
    path = Path(path_arg)
    if not path.exists():
        return ToolResult(
            call_id=call_id,
            name="read_file",
            content=f"file not found: {path}",
            is_error=True,
            error_code="NOT_FOUND",
        )
    if not path.is_file():
        return ToolResult(
            call_id=call_id,
            name="read_file",
            content=f"not a file: {path}",
            is_error=True,
            error_code="NOT_A_FILE",
        )
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        return ToolResult(
            call_id=call_id,
            name="read_file",
            content=str(exc),
            is_error=True,
            error_code="READ_ERROR",
        )
    return ToolResult(
        call_id=call_id,
        name="read_file",
        content=content,
    )


READ_FILE_TOOL = ToolDefinition(
    name="read_file",
    description="Read the contents of a file at the given path.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Filesystem path to read"},
        },
        "required": ["path"],
    },
    handler=read_file_handler,
)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def schemas(self) -> list[ToolSchema]:
        return [t.schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def execute(self, call_id: str, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                call_id=call_id,
                name=name,
                content=f"unknown tool: {name}",
                is_error=True,
                error_code="UNKNOWN_TOOL",
            )
        return tool.handler(call_id, arguments)


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(READ_FILE_TOOL)
    return registry
