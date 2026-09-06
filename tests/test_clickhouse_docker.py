from __future__ import annotations

from pathlib import Path

import pytest

from liteness.clickhouse.docker import (
    clickhouse_is_healthy,
    default_compose_file,
    ensure_clickhouse_running,
)


def test_default_compose_file_finds_bundled_path() -> None:
    path = default_compose_file()
    assert path is not None
    assert path.name == "docker-compose.yml"
    assert path.parent.name == "clickhouse"


def test_clickhouse_is_healthy_true(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Response:
        status_code = 200
        text = "Ok.\n"

    def fake_get(*args, **kwargs):
        return _Response()

    monkeypatch.setattr("httpx.get", fake_get)
    assert clickhouse_is_healthy("http://localhost:8123")


def test_ensure_clickhouse_running_skips_when_already_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "liteness.clickhouse.docker.clickhouse_is_healthy",
        lambda *args, **kwargs: True,
    )

    def fail_run(*args, **kwargs):
        raise AssertionError("docker compose should not run")

    monkeypatch.setattr("subprocess.run", fail_run)
    assert ensure_clickhouse_running() is True


def test_ensure_clickhouse_running_starts_docker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    compose = tmp_path / "docker-compose.yml"
    compose.write_text("services: {}\n", encoding="utf-8")
    calls: list[list[str]] = []
    health_checks = [False, False, True]

    def fake_healthy(*args, **kwargs) -> bool:
        return health_checks.pop(0) if health_checks else True

    monkeypatch.setattr(
        "liteness.clickhouse.docker.clickhouse_is_healthy", fake_healthy
    )
    monkeypatch.setattr("liteness.clickhouse.docker.shutil.which", lambda _: "docker")

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr("subprocess.run", fake_run)
    assert ensure_clickhouse_running(compose_file=compose, timeout_s=5) is True
    assert calls == [
        ["docker", "compose", "-f", str(compose.resolve()), "up", "-d"]
    ]


def test_repl_clickhouse_defaults_on() -> None:
    import argparse

    from liteness.cli import _add_clickhouse_flags

    repl_parser = argparse.ArgumentParser()
    _add_clickhouse_flags(repl_parser, repl_defaults=True)
    assert repl_parser.parse_args([]).clickhouse is True
    assert repl_parser.parse_args(["--no-clickhouse"]).clickhouse is False
