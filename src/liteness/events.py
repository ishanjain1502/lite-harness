"""In-process event bus for plugin and harness extension points."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable


EventHandler = Callable[..., None]


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event: str, handler: EventHandler) -> Callable[[], None]:
        self._handlers[event].append(handler)

        def unsubscribe() -> None:
            handlers = self._handlers.get(event, [])
            if handler in handlers:
                handlers.remove(handler)

        return unsubscribe

    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        for handler in list(self._handlers.get(event, [])):
            try:
                handler(*args, **kwargs)
            except Exception:  # noqa: BLE001 — observers must not break the loop
                continue

    def clear(self) -> None:
        self._handlers.clear()
