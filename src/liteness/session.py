"""Append-only session log, JSONL persistence, and message projection."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class SessionEvent:
    id: str
    type: str
    timestamp: datetime
    session_id: str
    turn: int
    step: int
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "turn": self.turn,
            "step": self.step,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionEvent:
        return cls(
            id=data["id"],
            type=data["type"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            session_id=data["session_id"],
            turn=data["turn"],
            step=data["step"],
            payload=data["payload"],
        )


@dataclass
class Message:
    role: str
    content: str
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id is not None:
            out["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            out["name"] = self.name
        return out


@dataclass
class OrphanToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]
    turn: int
    step: int
    event: SessionEvent


def find_orphan_tool_calls(events: list[SessionEvent]) -> list[OrphanToolCall]:
    """Return tool/call events with no matching tool/result (by call_id)."""
    calls: dict[str, SessionEvent] = {}
    result_ids: set[str] = set()

    for event in events:
        if event.type == "tool/call":
            calls[event.payload["call_id"]] = event
        elif event.type == "tool/result":
            result_ids.add(event.payload["call_id"])

    orphans: list[OrphanToolCall] = []
    for call_id, event in calls.items():
        if call_id in result_ids:
            continue
        orphans.append(
            OrphanToolCall(
                call_id=call_id,
                name=event.payload["name"],
                arguments=event.payload.get("arguments", {}),
                turn=event.turn,
                step=event.step,
                event=event,
            )
        )
    return orphans


@dataclass
class Session:
    session_id: str = field(default_factory=_new_id)
    events: list[SessionEvent] = field(default_factory=list)
    _turn: int = 0
    _step: int = 0
    log_path: Path | None = None

    def append(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        turn: int | None = None,
        step: int | None = None,
        fsync: bool | None = None,
    ) -> SessionEvent:
        event = SessionEvent(
            id=_new_id(),
            type=event_type,
            timestamp=_utc_now(),
            session_id=self.session_id,
            turn=self._turn if turn is None else turn,
            step=self._step if step is None else step,
            payload=payload,
        )
        self.events.append(event)
        if self.log_path is not None:
            self._append_to_disk(event, fsync=fsync)
        return event

    def append_to_disk(self, event: SessionEvent, *, fsync: bool = True) -> None:
        """Append a single event line to the session JSONL log."""
        if self.log_path is None:
            raise ValueError("session has no log_path configured")
        self._append_to_disk(event, fsync=fsync)

    def _append_to_disk(self, event: SessionEvent, *, fsync: bool | None = None) -> None:
        assert self.log_path is not None
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.to_dict(), ensure_ascii=False) + "\n"
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            if fsync is not False:
                os.fsync(handle.fileno())

    def export_jsonl(self, path: Path | str) -> None:
        """Write the full in-memory log to a JSONL file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @classmethod
    def load_from_jsonl(
        cls,
        path: Path | str,
        *,
        recover: bool = False,
    ) -> Session:
        """Load a session from a JSONL log and restore turn/step counters."""
        log_path = Path(path)
        events = list(_iter_events_from_jsonl(log_path))
        session_id = events[0].session_id if events else _new_id()
        session = cls(session_id=session_id, events=events, log_path=log_path)
        session.restore_counters_from_log()
        if recover:
            recover_orphans(session)
        return session

    @classmethod
    def open(cls, path: Path | str, *, recover: bool = False) -> Session:
        """Open an existing JSONL session or create a new durable session at path."""
        log_path = Path(path)
        if log_path.exists():
            return cls.load_from_jsonl(log_path, recover=recover)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return cls(log_path=log_path)

    def restore_counters_from_log(self) -> None:
        """Set _turn and _step from the event log (for resume after load)."""
        max_turn = 0
        for event in self.events:
            if event.type in ("turn/start", "turn/end"):
                max_turn = max(max_turn, event.payload.get("turn", event.turn))

        self._turn = max_turn
        self._step = 0
        for event in reversed(self.events):
            if event.type == "step/end" and event.turn == max_turn:
                self._step = event.payload.get("step", event.step)
                break

    def set_turn(self, turn: int) -> None:
        self._turn = turn

    def set_step(self, step: int) -> None:
        self._step = step

    def derive_messages(self) -> list[Message]:
        """Project session log into model-facing conversation history."""
        messages: list[Message] = []
        for event in self.events:
            if event.type == "user/message":
                messages.append(Message(role="user", content=event.payload["content"]))
            elif event.type == "assistant/message":
                messages.append(
                    Message(role="assistant", content=event.payload["content"])
                )
            elif event.type == "tool/result":
                messages.append(
                    Message(
                        role="tool",
                        content=event.payload["content"],
                        tool_call_id=event.payload["call_id"],
                        name=event.payload.get("name"),
                    )
                )
            elif event.type == "inject/context":
                messages.append(
                    Message(role="user", content=event.payload["content"])
                )
        return messages

    def fsync_log(self) -> None:
        """Ensure the JSONL file exists on disk (no-op if no log_path)."""
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.touch()
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())


def _iter_events_from_jsonl(path: Path) -> Iterator[SessionEvent]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
            yield SessionEvent.from_dict(data)


def recover_orphans(session: Session) -> list[SessionEvent]:
    """Append synthetic tool/result events for orphan tool/call entries."""
    recovered: list[SessionEvent] = []
    for orphan in find_orphan_tool_calls(session.events):
        event = session.append(
            "tool/result",
            {
                "call_id": orphan.call_id,
                "name": orphan.name,
                "content": (
                    "tool execution interrupted before result was recorded"
                ),
                "is_error": True,
                "error_code": "RECOVERY_INCOMPLETE",
            },
            turn=orphan.turn,
            step=orphan.step,
        )
        recovered.append(event)
    return recovered
