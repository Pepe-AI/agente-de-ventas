"""Test doubles for the fast (no-service) suite."""

from __future__ import annotations

from typing import Any

from app.domain.state import (
    ConversationState,
    Phase,
    from_payload,
    to_payload,
)


class InMemoryStateStore:
    """In-memory :class:`~app.domain.state.StateStore` for the fast suite.

    Mirrors Postgres semantics by storing the *serialized* payload and rebuilding
    on load: no aliasing with the caller's object, and the shared wire format is
    exercised on every round-trip.
    """

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}

    async def load(self, conversation_id: str) -> ConversationState | None:
        payload = self._rows.get(conversation_id)
        return from_payload(payload) if payload is not None else None

    async def save(self, conversation_id: str, state: ConversationState) -> None:
        self._rows[conversation_id] = to_payload(state)

    async def find_expired_deadlines(
        self, now: float
    ) -> list[tuple[str, ConversationState]]:
        # Mirrors the Postgres sweep: due deadline AND still collecting.
        return [
            (conversation_id, from_payload(payload))
            for conversation_id, payload in self._rows.items()
            if (deadline := payload.get("inactivity_deadline")) is not None
            and deadline <= now
            and payload.get("phase") == Phase.COLLECTING.value
        ]
