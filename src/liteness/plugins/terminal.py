"""Terminal plugin — run_command tool."""

from __future__ import annotations

import subprocess
from typing import Any

from liteness.context import Context
from liteness.tools import ToolDefinition, ToolResult


def run_command_handler(call_id: str, arguments: dict[str, Any]) -> ToolResult:
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        return ToolResult(
            call_id=call_id,
            name="run_command",
            content="command must be a non-empty string",
            is_error=True,
            error_code="INVALID_ARGS",
        )

    cwd = arguments.get("cwd")
    cwd_path = str(cwd) if isinstance(cwd, str) and cwd else None

    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=cwd_path,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            call_id=call_id,
            name="run_command",
            content="command timed out after 120s",
            is_error=True,
            error_code="TIMEOUT",
        )
    except OSError as exc:
        return ToolResult(
            call_id=call_id,
            name="run_command",
            content=str(exc),
            is_error=True,
            error_code="EXEC_ERROR",
        )

    output = completed.stdout
    if completed.stderr:
        output = f"{output}\n{completed.stderr}".strip()
    if not output:
        output = f"(exit code {completed.returncode})"

    return ToolResult(
        call_id=call_id,
        name="run_command",
        content=output,
        is_error=completed.returncode != 0,
        error_code="NON_ZERO_EXIT" if completed.returncode != 0 else None,
    )


class TerminalPlugin:
    name = "terminal"

    def install(self, ctx: Context, config: dict[str, Any]) -> None:
        ctx.register_tool(
            ToolDefinition(
                name="run_command",
                description=(
                    "Run a shell command and return combined stdout/stderr output."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute",
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Optional working directory",
                        },
                    },
                    "required": ["command"],
                },
                handler=run_command_handler,
            )
        )

    def uninstall(self, ctx: Context) -> None:
        pass
