"""Local HTTP server for the analytics dashboard."""

from __future__ import annotations

import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from liteness.analytics.store import AnalyticsStore
from liteness.analytics.time_range import parse_since

logger = logging.getLogger(__name__)


def _static_index_path() -> Path:
    return Path(__file__).resolve().parent / "static" / "index.html"


class AnalyticsHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        store: AnalyticsStore,
    ) -> None:
        self.store = store
        super().__init__(server_address, AnalyticsRequestHandler)


class AnalyticsRequestHandler(BaseHTTPRequestHandler):
    server: AnalyticsHTTPServer  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._serve_index()
            return
        if parsed.path.startswith("/api/"):
            self._serve_api(parsed)
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def _serve_index(self) -> None:
        index_path = _static_index_path()
        if not index_path.exists():
            self._send_text("analytics UI missing", status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        body = index_path.read_text(encoding="utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def _serve_api(self, parsed) -> None:
        params = parse_qs(parsed.query)
        since_raw = params.get("since", ["7d"])[0]
        try:
            since = parse_since(since_raw)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        store = self.server.store
        path = parsed.path.rstrip("/")

        if path == "/api/overview":
            self._send_json(store.overview(since=since).to_dict())
            return

        if path == "/api/sessions":
            limit = int(params.get("limit", ["50"])[0])
            offset = int(params.get("offset", ["0"])[0])
            sessions = [item.to_dict() for item in store.list_sessions(since=since, limit=limit, offset=offset)]
            self._send_json({"sessions": sessions})
            return

        if path == "/api/tokens/timeseries":
            bucket = params.get("bucket", ["day"])[0]
            if bucket not in {"day", "hour"}:
                self._send_json({"error": "bucket must be day or hour"}, status=HTTPStatus.BAD_REQUEST)
                return
            buckets = [item.to_dict() for item in store.timeseries(since=since, bucket=bucket)]
            self._send_json({"buckets": buckets})
            return

        if path == "/api/tools/error-rates":
            rows = [item.to_dict() for item in store.tool_error_rates(since=since)]
            self._send_json({"tools": rows})
            return

        if path == "/api/stop-reasons":
            rows = [item.to_dict() for item in store.stop_reasons(since=since)]
            self._send_json({"stop_reasons": rows})
            return

        if path.startswith("/api/sessions/"):
            session_id = path.removeprefix("/api/sessions/")
            if not session_id or "/" in session_id:
                self._send_json({"error": "invalid session id"}, status=HTTPStatus.BAD_REQUEST)
                return
            detail = store.get_session(session_id)
            if detail is None:
                self._send_json({"error": "session not found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(detail.to_dict())
            return

        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def _send_json(self, payload: dict, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2)
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_text(self, text: str, *, status: HTTPStatus) -> None:
        encoded = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def serve_analytics(
    store: AnalyticsStore,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    server = AnalyticsHTTPServer((host, port), store)
    url = f"http://{host}:{port}/"
    print(f"lite-ness analytics dashboard at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down analytics server.")
    finally:
        server.server_close()
