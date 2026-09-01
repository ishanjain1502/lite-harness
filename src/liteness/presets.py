"""Agent preset loading from built-in YAML files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from liteness.plugins.base import PluginConfigError


@dataclass
class PluginSpec:
    name: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class Preset:
    name: str
    plugins: list[PluginSpec]


def _package_presets_dir() -> Path:
    return Path(__file__).resolve().parent / "builtin_presets"


def load_preset(name: str, *, presets_dir: Path | None = None) -> Preset:
    root = presets_dir or _package_presets_dir()
    path = root / f"{name}.yaml"
    if not path.exists():
        raise PluginConfigError(f"preset not found: {name}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise PluginConfigError(f"invalid preset file: {path}")

    preset_name = raw.get("name")
    if not isinstance(preset_name, str) or not preset_name:
        raise PluginConfigError(f"preset {path} missing 'name'")

    plugin_rows = raw.get("plugins")
    if not isinstance(plugin_rows, list) or not plugin_rows:
        raise PluginConfigError(f"preset {path} missing 'plugins'")

    plugins: list[PluginSpec] = []
    for row in plugin_rows:
        if isinstance(row, str):
            plugins.append(PluginSpec(name=row))
            continue
        if isinstance(row, dict) and isinstance(row.get("name"), str):
            config = row.get("config", {})
            if not isinstance(config, dict):
                raise PluginConfigError(
                    f"plugin config for {row['name']} must be a mapping"
                )
            plugins.append(PluginSpec(name=row["name"], config=config))
            continue
        raise PluginConfigError(f"invalid plugin entry in preset {path}")

    return Preset(name=preset_name, plugins=plugins)
