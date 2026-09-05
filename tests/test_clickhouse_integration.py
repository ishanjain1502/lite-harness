from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("LITENESS_CLICKHOUSE_URL"),
    reason="LITENESS_CLICKHOUSE_URL not set",
)


@pytest.mark.clickhouse_integration
def test_http_insert_and_count() -> None:
    import httpx

    from liteness.clickhouse.client import HttpClickHouseClient
    from liteness.clickhouse.projector import ClickHouseProjector
    from liteness.session import SessionEvent
    from datetime import datetime, timezone

    url = os.environ["LITENESS_CLICKHOUSE_URL"]
    client = HttpClickHouseClient(url=url, database="liteness")
    client.ensure_schema()
    projector = ClickHouseProjector(mode="redacted")
    event = SessionEvent(
        id="integration-e1",
        type="turn/start",
        timestamp=datetime.now(timezone.utc),
        session_id="integration-s1",
        turn=1,
        step=0,
        payload={"turn": 1},
    )
    row = projector.project_event(event)
    assert row is not None
    client.insert_rows("session_events", [row.to_insert_dict()])
    response = httpx.get(
        url,
        params={
            "database": "liteness",
            "query": (
                "SELECT count() FROM session_events "
                "WHERE event_id = 'integration-e1'"
            ),
        },
        timeout=5.0,
    )
    response.raise_for_status()
    assert int(response.text.strip()) >= 1
