"""Helpers for tests and local development."""

from __future__ import annotations

from liteness.context import Context
from liteness.harness import _plugin_config
from liteness.plugins import get_plugin
from liteness.presets import PluginSpec
from liteness.session import Session
from liteness.tools import ToolRegistry


def registry_with_plugins(*plugin_names: str, project_id: str = "test") -> ToolRegistry:
    """Install named plugins into a fresh Context and return its tool registry."""
    if not plugin_names:
        plugin_names = ("filesystem",)

    ctx = Context()
    session = Session()
    for name in plugin_names:
        plugin = get_plugin(name)
        config = _plugin_config(
            PluginSpec(name=name),
            project_id=project_id,
            session=session,
        )
        plugin.install(ctx, config)
    return ctx.tools
