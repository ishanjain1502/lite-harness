"""Harness composition — presets, plugins, and session runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from liteness.context import Context
from liteness.plugins import get_plugin, list_plugins
from liteness.plugins.base import (
    InstalledPlugin,
    PluginConfigError,
    install_plugins,
    uninstall_plugins,
)
from liteness.presets import Preset, PluginSpec, load_preset
from liteness.session import Session


@dataclass
class HarnessRuntime:
    preset: Preset
    project_id: str
    ctx: Context
    installed: list[InstalledPlugin]
    session: Session


def _plugin_config(
    spec: PluginSpec,
    *,
    project_id: str,
    session: Session,
) -> dict[str, Any]:
    config = dict(spec.config)

    if spec.name == "memory":
        config.setdefault("project_id", project_id)
        config.setdefault("store_path", "./memory")

        def session_append(event_type: str, payload: dict[str, Any]) -> None:
            session.append(event_type, payload)

        config["session_append"] = session_append

    if spec.name == "rag":
        config.setdefault("index_path", ".liteness/rag-index.json")

    return config


def create_runtime(
    *,
    preset_name: str,
    session: Session,
    project_id: str = "default",
    presets_dir: Path | None = None,
    extra_plugins: list[str] | None = None,
) -> HarnessRuntime:
    preset = load_preset(preset_name, presets_dir=presets_dir)
    plugin_specs = list(preset.plugins)
    if extra_plugins:
        existing = {spec.name for spec in plugin_specs}
        for name in extra_plugins:
            if name not in existing:
                plugin_specs.append(PluginSpec(name=name))

    for spec in plugin_specs:
        if spec.name not in list_plugins():
            raise PluginConfigError(f"unknown plugin in preset: {spec.name}")

    ctx = Context()
    install_pairs = [
        (
            get_plugin(spec.name),
            _plugin_config(spec, project_id=project_id, session=session),
        )
        for spec in plugin_specs
    ]
    installed = install_plugins(ctx, install_pairs)
    return HarnessRuntime(
        preset=preset,
        project_id=project_id,
        ctx=ctx,
        installed=installed,
        session=session,
    )


def dispose_runtime(runtime: HarnessRuntime) -> None:
    uninstall_plugins(runtime.ctx, runtime.installed)
