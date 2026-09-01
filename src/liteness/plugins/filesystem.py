"""Filesystem plugin — read_file tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from liteness.context import Context
from liteness.plugins.base import PluginConfigError
from liteness.tools import ToolDefinition, ToolResult


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
    return ToolResult(call_id=call_id, name="read_file", content=content)


class FilesystemPlugin:
    name = "filesystem"

    def install(self, ctx: Context, config: dict[str, Any]) -> None:
        ctx.register_tool(
            ToolDefinition(
                name="read_file",
                description="Read the contents of a file at the given path.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Filesystem path to read",
                        },
                    },
                    "required": ["path"],
                },
                handler=read_file_handler,
            )
        )

    def uninstall(self, ctx: Context) -> None:
        pass
