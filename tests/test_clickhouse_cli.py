from __future__ import annotations

import pytest

from liteness.cli import main


def test_cli_mutual_exclusion() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["run", "--clickhouse", "--clickhouse-full", "hello"])
    assert exc.value.code == 2


def test_run_clickhouse_down_does_not_fail_turn(tmp_path, monkeypatch) -> None:
    from liteness.clickhouse.client import FakeClickHouseClient
    from liteness.plugins.clickhouse import ClickHousePlugin

    client = FakeClickHouseClient(fail_with=RuntimeError("down"))
    original_install = ClickHousePlugin.install

    def install(self, ctx, config):
        config = dict(config)
        config["client"] = client
        config["spill_path"] = str(tmp_path / "spill.jsonl")
        config["flush_interval_s"] = 0.01
        return original_install(self, ctx, config)

    monkeypatch.setattr(ClickHousePlugin, "install", install)
    rc = main(["run", "--clickhouse", "--readme", "README.md", "summarize the project"])
    assert rc in (0, 1)
