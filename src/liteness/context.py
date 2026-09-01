"""Harness context — plugin-facing API backed by tools and events."""

from __future__ import annotations

from typing import Any, Callable

from liteness.events import EventBus, EventHandler
from liteness.tools import ToolDefinition, ToolRegistry


class Context:
    """Minimal plugin surface: tools, events, reversible effects, shared services."""

    def __init__(self) -> None:
        self.tools = ToolRegistry()
        self._events = EventBus()
        self._disposers: list[Callable[[], None]] = []
        self.services: dict[str, Any] = {}

    def on(self, event: str, handler: EventHandler) -> None:
        disposer = self._events.subscribe(event, handler)
        self.effect(disposer)

    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        self._events.emit(event, *args, **kwargs)

    def effect(self, disposer: Callable[[], None]) -> None:
        self._disposers.append(disposer)

    def register_tool(self, tool: ToolDefinition) -> None:
        self.tools.register(tool)
        name = tool.name

        def unregister() -> None:
            self.tools.unregister(name)

        self.effect(unregister)

    def dispose(self) -> None:
        while self._disposers:
            disposer = self._disposers.pop()
            disposer()
        self._events.clear()
        self.services.clear()
