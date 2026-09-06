"""Start local ClickHouse via docker compose when needed."""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path

from liteness.clickhouse.client import clickhouse_credentials_from_config

logger = logging.getLogger("liteness.clickhouse")

_DEFAULT_URL = "http://localhost:8123"
_COMPOSE_REL = Path("docker/clickhouse/docker-compose.yml")


def default_compose_file() -> Path | None:
    """Resolve the bundled ClickHouse compose file from cwd or package tree."""
    candidates = [
        _COMPOSE_REL,
        Path(__file__).resolve().parents[3] / _COMPOSE_REL,
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def clickhouse_is_healthy(
    url: str,
    *,
    user: str | None = None,
    password: str | None = None,
    timeout_s: float = 2.0,
) -> bool:
    try:
        import httpx  # noqa: PLC0415
    except ImportError:
        return False

    auth = (user, password) if user and password else None
    try:
        response = httpx.get(
            f"{url.rstrip('/')}/ping",
            auth=auth,
            timeout=timeout_s,
        )
    except Exception:
        return False
    return response.status_code == 200 and response.text.strip() == "Ok."


def ensure_clickhouse_running(
    *,
    url: str | None = None,
    user: str | None = None,
    password: str | None = None,
    compose_file: Path | None = None,
    timeout_s: float = 60.0,
) -> bool:
    """Start the bundled ClickHouse container and wait until HTTP /ping succeeds.

    Returns True when ClickHouse is reachable, False when startup was skipped or failed.
    """
    config = {"url": url, "user": user, "password": password}
    resolved_url = url or _DEFAULT_URL
    resolved_user, resolved_password = clickhouse_credentials_from_config(config)

    if clickhouse_is_healthy(
        resolved_url, user=resolved_user, password=resolved_password
    ):
        return True

    if shutil.which("docker") is None:
        logger.warning("docker not found on PATH; skipping ClickHouse auto-start")
        return False

    compose = compose_file or default_compose_file()
    if compose is None:
        logger.warning(
            "ClickHouse compose file not found; expected %s", _COMPOSE_REL
        )
        return False

    logger.info("Starting ClickHouse via %s", compose)
    try:
        subprocess.run(
            ["docker", "compose", "-f", str(compose), "up", "-d"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "docker compose up failed (%s): %s",
            exc.returncode,
            (exc.stderr or exc.stdout or "").strip(),
        )
        return False

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if clickhouse_is_healthy(
            resolved_url, user=resolved_user, password=resolved_password
        ):
            logger.info("ClickHouse is ready at %s", resolved_url)
            return True
        time.sleep(0.5)

    logger.warning("ClickHouse did not become healthy within %.0fs", timeout_s)
    return False
