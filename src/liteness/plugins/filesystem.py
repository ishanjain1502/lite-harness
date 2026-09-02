"""Filesystem plugin — read, write, edit, and delete files and directories."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from liteness.context import Context
from liteness.tools import ToolDefinition, ToolResult


def _invalid_args(call_id: str, name: str, message: str) -> ToolResult:
    return ToolResult(
        call_id=call_id,
        name=name,
        content=message,
        is_error=True,
        error_code="INVALID_ARGS",
    )


def _path_arg(arguments: dict[str, Any], key: str = "path") -> str | None:
    value = arguments.get(key)
    if isinstance(value, str) and value:
        return value
    return None


def read_file_handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
    path_arg = _path_arg(arguments)
    if path_arg is None:
        return _invalid_args(call_id, "read_file", "path must be a non-empty string")

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


def read_directory_handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
    path_arg = _path_arg(arguments)
    if path_arg is None:
        return _invalid_args(
            call_id, "read_directory", "path must be a non-empty string"
        )

    path = Path(path_arg)
    if not path.exists():
        return ToolResult(
            call_id=call_id,
            name="read_directory",
            content=f"directory not found: {path}",
            is_error=True,
            error_code="NOT_FOUND",
        )
    if not path.is_dir():
        return ToolResult(
            call_id=call_id,
            name="read_directory",
            content=f"not a directory: {path}",
            is_error=True,
            error_code="NOT_A_DIRECTORY",
        )

    try:
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as exc:
        return ToolResult(
            call_id=call_id,
            name="read_directory",
            content=str(exc),
            is_error=True,
            error_code="READ_ERROR",
        )

    if not entries:
        return ToolResult(
            call_id=call_id,
            name="read_directory",
            content=f"{path} (empty)",
        )

    lines: list[str] = []
    for entry in entries:
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"{entry.name}{suffix}")
    return ToolResult(
        call_id=call_id,
        name="read_directory",
        content="\n".join(lines),
    )


def create_file_handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
    path_arg = _path_arg(arguments)
    if path_arg is None:
        return _invalid_args(call_id, "create_file", "path must be a non-empty string")

    content = arguments.get("content")
    if not isinstance(content, str):
        return _invalid_args(call_id, "create_file", "content must be a string")

    path = Path(path_arg)
    if path.exists():
        return ToolResult(
            call_id=call_id,
            name="create_file",
            content=f"file already exists: {path}",
            is_error=True,
            error_code="ALREADY_EXISTS",
        )

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        return ToolResult(
            call_id=call_id,
            name="create_file",
            content=str(exc),
            is_error=True,
            error_code="WRITE_ERROR",
        )

    return ToolResult(
        call_id=call_id,
        name="create_file",
        content=f"created {path} ({len(content.encode('utf-8'))} bytes)",
    )


def edit_file_handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
    path_arg = _path_arg(arguments)
    if path_arg is None:
        return _invalid_args(call_id, "edit_file", "path must be a non-empty string")

    old_text = arguments.get("old_text")
    new_text = arguments.get("new_text")
    if not isinstance(old_text, str):
        return _invalid_args(call_id, "edit_file", "old_text must be a string")
    if not isinstance(new_text, str):
        return _invalid_args(call_id, "edit_file", "new_text must be a string")

    path = Path(path_arg)
    if not path.exists():
        return ToolResult(
            call_id=call_id,
            name="edit_file",
            content=f"file not found: {path}",
            is_error=True,
            error_code="NOT_FOUND",
        )
    if not path.is_file():
        return ToolResult(
            call_id=call_id,
            name="edit_file",
            content=f"not a file: {path}",
            is_error=True,
            error_code="NOT_A_FILE",
        )

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        return ToolResult(
            call_id=call_id,
            name="edit_file",
            content=str(exc),
            is_error=True,
            error_code="READ_ERROR",
        )

    if old_text not in content:
        return ToolResult(
            call_id=call_id,
            name="edit_file",
            content=f"old_text not found in {path}",
            is_error=True,
            error_code="TEXT_NOT_FOUND",
        )

    updated = content.replace(old_text, new_text, 1)
    try:
        path.write_text(updated, encoding="utf-8")
    except OSError as exc:
        return ToolResult(
            call_id=call_id,
            name="edit_file",
            content=str(exc),
            is_error=True,
            error_code="WRITE_ERROR",
        )

    return ToolResult(
        call_id=call_id,
        name="edit_file",
        content=f"updated {path}",
    )


def delete_file_handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
    path_arg = _path_arg(arguments)
    if path_arg is None:
        return _invalid_args(call_id, "delete_file", "path must be a non-empty string")

    path = Path(path_arg)
    if not path.exists():
        return ToolResult(
            call_id=call_id,
            name="delete_file",
            content=f"file not found: {path}",
            is_error=True,
            error_code="NOT_FOUND",
        )
    if not path.is_file():
        return ToolResult(
            call_id=call_id,
            name="delete_file",
            content=f"not a file: {path}",
            is_error=True,
            error_code="NOT_A_FILE",
        )

    try:
        path.unlink()
    except OSError as exc:
        return ToolResult(
            call_id=call_id,
            name="delete_file",
            content=str(exc),
            is_error=True,
            error_code="DELETE_ERROR",
        )

    return ToolResult(
        call_id=call_id,
        name="delete_file",
        content=f"deleted {path}",
    )


def delete_directory_handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
    path_arg = _path_arg(arguments)
    if path_arg is None:
        return _invalid_args(
            call_id, "delete_directory", "path must be a non-empty string"
        )

    recursive = arguments.get("recursive", False)
    if not isinstance(recursive, bool):
        return _invalid_args(
            call_id, "delete_directory", "recursive must be a boolean"
        )

    path = Path(path_arg)
    if not path.exists():
        return ToolResult(
            call_id=call_id,
            name="delete_directory",
            content=f"directory not found: {path}",
            is_error=True,
            error_code="NOT_FOUND",
        )
    if not path.is_dir():
        return ToolResult(
            call_id=call_id,
            name="delete_directory",
            content=f"not a directory: {path}",
            is_error=True,
            error_code="NOT_A_DIRECTORY",
        )

    try:
        if recursive:
            shutil.rmtree(path)
        else:
            path.rmdir()
    except OSError as exc:
        return ToolResult(
            call_id=call_id,
            name="delete_directory",
            content=str(exc),
            is_error=True,
            error_code="DELETE_ERROR",
        )

    mode = "recursively deleted" if recursive else "deleted"
    return ToolResult(
        call_id=call_id,
        name="delete_directory",
        content=f"{mode} {path}",
    )


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
                idempotent=True,
            )
        )
        ctx.register_tool(
            ToolDefinition(
                name="read_directory",
                description="List files and subdirectories at the given path.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to list",
                        },
                    },
                    "required": ["path"],
                },
                handler=read_directory_handler,
                idempotent=True,
            )
        )
        ctx.register_tool(
            ToolDefinition(
                name="create_file",
                description="Create a new file with the given UTF-8 content.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Filesystem path for the new file",
                        },
                        "content": {
                            "type": "string",
                            "description": "File content to write",
                        },
                    },
                    "required": ["path", "content"],
                },
                handler=create_file_handler,
            )
        )
        ctx.register_tool(
            ToolDefinition(
                name="edit_file",
                description=(
                    "Replace the first occurrence of old_text with new_text in a file."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Filesystem path to edit",
                        },
                        "old_text": {
                            "type": "string",
                            "description": "Text to find in the file",
                        },
                        "new_text": {
                            "type": "string",
                            "description": "Replacement text",
                        },
                    },
                    "required": ["path", "old_text", "new_text"],
                },
                handler=edit_file_handler,
            )
        )
        ctx.register_tool(
            ToolDefinition(
                name="delete_file",
                description="Delete a file at the given path.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Filesystem path to delete",
                        },
                    },
                    "required": ["path"],
                },
                handler=delete_file_handler,
            )
        )
        ctx.register_tool(
            ToolDefinition(
                name="delete_directory",
                description="Delete a directory. Use recursive=true for non-empty trees.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to delete",
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "Delete directory contents recursively",
                        },
                    },
                    "required": ["path"],
                },
                handler=delete_directory_handler,
            )
        )

    def uninstall(self, ctx: Context) -> None:
        pass
