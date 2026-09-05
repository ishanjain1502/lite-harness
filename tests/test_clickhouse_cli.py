from __future__ import annotations

import json
from pathlib import Path

import pytest

from liteness.cli import main
from liteness.plugins.base import PluginConfigError


def _inserted_event_ids(client) -> set[str]:
    return {
        row["event_id"]
        for table, rows in client.calls
        if table == "session_events"
        for row in rows
        if row.get("channel") == "ledger"
    }


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
    assert rc == 0


def test_export_clickhouse_backfill(tmp_path, monkeypatch, capsys) -> None:
    from liteness.clickhouse.client import FakeClickHouseClient
    from liteness.cli import _make_clickhouse_exporter
    from liteness.session import Session

    session = Session(log_path=tmp_path / "s.jsonl")
    session.append("user/message", {"content": "hello"}, turn=1, step=0)
    session.append("assistant/chunk", {"content_delta": "a"}, turn=1, step=1)
    session.append("assistant/chunk", {"content_delta": "b"}, turn=1, step=1)
    expected_ids = {session.events[0].id, session.events[1].id}
    client = FakeClickHouseClient()

    def fake_make(config):
        config = dict(config)
        config["client"] = client
        config["spill_path"] = str(tmp_path / "spill.jsonl")
        return _make_clickhouse_exporter(config)

    monkeypatch.setattr("liteness.cli._make_clickhouse_exporter", fake_make)
    rc = main(["export-clickhouse", str(session.log_path)])
    assert rc == 0
    assert expected_ids <= _inserted_event_ids(client)
    assert "Exported 2 events" in capsys.readouterr().out


def test_eval_session_exports_eval_results(tmp_path, monkeypatch) -> None:
    from liteness.clickhouse.client import FakeClickHouseClient
    from liteness.cli import _make_clickhouse_exporter

    client = FakeClickHouseClient()

    def fake_make(config):
        config = dict(config)
        config["client"] = client
        config["spill_path"] = str(tmp_path / "spill.jsonl")
        return _make_clickhouse_exporter(config)

    monkeypatch.setattr("liteness.cli._make_clickhouse_exporter", fake_make)
    fixtures = Path(__file__).parent / "fixtures" / "evals"
    examples = Path(__file__).parent.parent / "examples" / "evals"
    from liteness.clickhouse.projector import ClickHouseProjector
    from liteness.session import Session

    source_session = Session.load_from_jsonl(fixtures / "weather-001.jsonl")
    source_projector = ClickHouseProjector()
    expected_event_ids = {
        row.event_id
        for event in source_session.events
        if (row := source_projector.project_event(event)) is not None
    }
    rc = main([
        "eval", "session", str(fixtures / "weather-001.jsonl"),
        "--eval-file", str(examples / "basic.yaml"),
        "--case", "weather-001",
        "--clickhouse",
    ])
    assert rc in (0, 1)
    assert any(table == "eval_results" for table, _ in client.calls)
    assert expected_event_ids <= _inserted_event_ids(client)


def test_eval_clickhouse_export_failure_does_not_change_exit_code(
    tmp_path, monkeypatch
) -> None:
    fixtures = Path(__file__).parent / "fixtures" / "evals"
    examples = Path(__file__).parent.parent / "examples" / "evals"
    base_args = [
        "eval", "session", str(fixtures / "weather-001.jsonl"),
        "--eval-file", str(examples / "basic.yaml"),
        "--case", "weather-001",
    ]
    expected_rc = main(base_args)

    def raise_export(args, report):
        raise RuntimeError("clickhouse down")

    monkeypatch.setattr("liteness.cli._export_eval_report", raise_export)
    rc = main([*base_args, "--clickhouse"])
    assert rc == expected_rc


def test_run_clickhouse_config_error_is_clean(monkeypatch, capsys) -> None:
    from liteness.plugins.clickhouse import ClickHousePlugin

    def fail_install(self, ctx, config):
        raise PluginConfigError("httpx unavailable")

    monkeypatch.setattr(ClickHousePlugin, "install", fail_install)

    rc = main(["run", "--clickhouse", "hi"])

    assert rc == 1
    assert "Error: httpx unavailable" in capsys.readouterr().err


def test_run_clickhouse_full_exports_raw_payload(tmp_path, monkeypatch) -> None:
    from liteness.clickhouse.client import FakeClickHouseClient
    from liteness.plugins.clickhouse import ClickHousePlugin

    client = FakeClickHouseClient()
    original_install = ClickHousePlugin.install

    def install(self, ctx, config):
        config = dict(config)
        config["client"] = client
        config["spill_path"] = str(tmp_path / "spill.jsonl")
        config["flush_interval_s"] = 0.01
        return original_install(self, ctx, config)

    monkeypatch.setattr(ClickHousePlugin, "install", install)
    rc = main(["run", "--clickhouse-full", "--readme", "README.md", "raw secret"])

    assert rc == 0
    user_rows = [
        row
        for table, rows in client.calls
        if table == "session_events"
        for row in rows
        if row.get("event_type") == "user/message"
    ]
    assert user_rows
    assert json.loads(user_rows[0]["payload"])["content"] == "raw secret"


def test_run_preset_clickhouse_exports_ledger_events(tmp_path, monkeypatch) -> None:
    from liteness.clickhouse.client import FakeClickHouseClient
    from liteness.plugins.clickhouse import ClickHousePlugin

    client = FakeClickHouseClient()
    original_install = ClickHousePlugin.install

    def install(self, ctx, config):
        config = dict(config)
        config["client"] = client
        config["spill_path"] = str(tmp_path / "spill.jsonl")
        config["flush_interval_s"] = 0.01
        return original_install(self, ctx, config)

    monkeypatch.setattr(ClickHousePlugin, "install", install)
    rc = main([
        "run",
        "--preset", "coder",
        "--clickhouse",
        "--readme", "README.md",
        "summarize the project",
    ])

    assert rc == 0
    assert _inserted_event_ids(client)


def test_eval_results_insert_failure_spills_rows(tmp_path, monkeypatch) -> None:
    from liteness.clickhouse.client import FakeClickHouseClient
    from liteness.cli import _make_clickhouse_exporter

    class EvalFailingClient(FakeClickHouseClient):
        def insert_rows(self, table, rows):
            if table == "eval_results":
                raise RuntimeError("eval insert failed")
            super().insert_rows(table, rows)

    client = EvalFailingClient()
    spill = tmp_path / "spill.jsonl"

    def fake_make(config):
        config = dict(config)
        config["client"] = client
        config["spill_path"] = str(spill)
        return _make_clickhouse_exporter(config)

    monkeypatch.setattr("liteness.cli._make_clickhouse_exporter", fake_make)
    fixtures = Path(__file__).parent / "fixtures" / "evals"
    examples = Path(__file__).parent.parent / "examples" / "evals"
    rc = main([
        "eval", "session", str(fixtures / "weather-001.jsonl"),
        "--eval-file", str(examples / "basic.yaml"),
        "--case", "weather-001",
        "--clickhouse",
    ])

    assert rc in (0, 1)
    spilled = [
        json.loads(line)
        for line in spill.read_text(encoding="utf-8").splitlines()
    ]
    assert any(entry["table"] == "eval_results" for entry in spilled)
