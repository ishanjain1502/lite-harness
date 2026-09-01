"""Plugin protocol and install lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from liteness.context import Context


class PluginConfigError(Exception):
    """Raised when plugin configuration is invalid at install time."""


class Plugin(Protocol):
    name: str

    def install(self, ctx: Context, config: dict[str, Any]) -> None: ...

    def uninstall(self, ctx: Context) -> None: ...


@dataclass
class InstalledPlugin:
    plugin: Plugin
    config: dict[str, Any]


def install_plugins(
    ctx: Context,
    plugins: list[tuple[Plugin, dict[str, Any]]],
) -> list[InstalledPlugin]:
    """Install plugins in order; unwind partial installs on failure."""
    installed: list[InstalledPlugin] = []
    try:
        for plugin, config in plugins:
            plugin.install(ctx, dict(config))
            installed.append(InstalledPlugin(plugin, dict(config)))
    except Exception:
        for entry in reversed(installed):
            entry.plugin.uninstall(ctx)
        ctx.dispose()
        raise
    return installed


def uninstall_plugins(ctx: Context, installed: list[InstalledPlugin]) -> None:
    for entry in reversed(installed):
        entry.plugin.uninstall(ctx)
    ctx.dispose()
