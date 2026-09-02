"""Tool registry — domain tools are registered by plugins."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any, Callable

from liteness.llm import ToolSchema
from liteness.types import CancelToken


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


@dataclass
class DispatchCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


ToolHandler = Callable[[str, dict[str, Any]], ToolResult]


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    timeout_s: float | None = None
    idempotent: bool = False

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
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

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

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
        try:
            return tool.handler(call_id, arguments)
        except Exception as exc:  # noqa: BLE001 — harness normalizes tool throws
            return ToolResult(
                call_id=call_id,
                name=name,
                content=str(exc),
                is_error=True,
                error_code="TOOL_EXCEPTION",
            )

    def dispatch(
        self,
        calls: list[DispatchCall],
        *,
        default_timeout_s: float,
        cancel: CancelToken | None = None,
    ) -> list[ToolResult]:
        """Execute multiple tool calls concurrently; preserve input order."""
        if not calls:
            return []

        if len(calls) == 1:
            return [self._execute_one(calls[0], default_timeout_s, cancel)]

        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            futures = [
                pool.submit(self._execute_one, call, default_timeout_s, cancel)
                for call in calls
            ]
            return [future.result() for future in futures]

    def _execute_one(
        self,
        call: DispatchCall,
        default_timeout_s: float,
        cancel: CancelToken | None,
    ) -> ToolResult:
        if cancel is not None and cancel.cancelled:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                content="execution cancelled",
                is_error=True,
                error_code="CANCELLED",
            )

        tool = self._tools.get(call.name)
        timeout_s = (
            tool.timeout_s if tool is not None and tool.timeout_s is not None else default_timeout_s
        )

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                self.execute, call.call_id, call.name, call.arguments
            )
            try:
                return future.result(timeout=timeout_s)
            except FuturesTimeoutError:
                return ToolResult(
                    call_id=call.call_id,
                    name=call.name,
                    content=f"tool timed out after {timeout_s}s",
                    is_error=True,
                    error_code="TOOL_TIMEOUT",
                )


def default_registry() -> ToolRegistry:
    """Empty registry — tools are contributed by plugins."""
    return ToolRegistry()
