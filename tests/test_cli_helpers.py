from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from liteness.cli import _build_runtime
from liteness.harness import HarnessRuntime
from liteness.plugins.base import PluginConfigError
from liteness.session import Session


def _args(**overrides) -> argparse.Namespace:
    base = dict(preset=None, project_id="default")
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_runtime_returns_none_when_no_preset(tmp_path: Path) -> None:
    session = Session()
    assert _build_runtime(_args(), session) is None


def test_build_runtime_returns_runtime_for_known_preset(tmp_path: Path) -> None:
    session = Session()
    runtime = _build_runtime(_args(preset="coder"), session)
    assert isinstance(runtime, HarnessRuntime)
    assert "read_file" in runtime.ctx.tools.names()


def test_build_runtime_raises_for_unknown_preset(tmp_path: Path) -> None:
    session = Session()
    with pytest.raises(PluginConfigError):
        _build_runtime(_args(preset="nope"), session)


def test_repl_subcommand_is_registered() -> None:
    from liteness.cli import main

    with pytest.raises(SystemExit):
        main(["repl", "--help"])

