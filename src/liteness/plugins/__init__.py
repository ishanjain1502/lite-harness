"""Built-in plugin registry."""

from __future__ import annotations

from typing import Callable

from liteness.plugins.base import Plugin, PluginConfigError
from liteness.plugins.filesystem import FilesystemPlugin
from liteness.plugins.memory import MemoryPlugin
from liteness.plugins.rag import RAGPlugin
from liteness.plugins.telemetry import TelemetryPlugin
from liteness.plugins.terminal import TerminalPlugin
from liteness.plugins.video import VideoPlugin

_BUILTIN_FACTORIES: dict[str, Callable[[], Plugin]] = {
    "filesystem": FilesystemPlugin,
    "terminal": TerminalPlugin,
    "memory": MemoryPlugin,
    "rag": RAGPlugin,
    "telemetry": TelemetryPlugin,
    "video": VideoPlugin,
}


def get_plugin(name: str) -> Plugin:
    factory = _BUILTIN_FACTORIES.get(name)
    if factory is None:
        raise PluginConfigError(f"unknown plugin: {name}")
    return factory()


def list_plugins() -> list[str]:
    return sorted(_BUILTIN_FACTORIES.keys())
