"""Append-only session log and message projection."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


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
class Session:
    session_id: str = field(default_factory=_new_id)
    events: list[SessionEvent] = field(default_factory=list)
    _turn: int = 0
    _step: int = 0

    def append(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        turn: int | None = None,
        step: int | None = None,
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
        return event

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
