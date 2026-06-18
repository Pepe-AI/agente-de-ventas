"""Persistent per-sender conversation state (Redis-backed).

Holds the trip type (which schema is in use), the slots captured so far, the
conversation phase, and the last slot the bot asked for. Persisted as JSON so it
survives between turns and across workers; the slot values are already
JSON-friendly (scalars / lists / nested dicts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, cast

from app.understanding.schemas import TripType


class Phase(StrEnum):
    """Where the conversation is in its lifecycle."""

    COLLECTING = "collecting"
    # Terminal: handed off to a human for ANY reason (completa / atorado /
    # pidió_humano). Not "completed" — the bot stays silent from here on.
    HANDED_OFF = "handed_off"


def _empty_slots() -> dict[str, Any]:
    return {}


def _empty_asked() -> set[str]:
    return set()


def _empty_attempts() -> dict[str, int]:
    return {}


def _empty_pending() -> set[str]:
    return set()


@dataclass(slots=True)
class ConversationState:
    """Mutable per-turn working state for one sender."""

    # ``None`` until the campaign router picks a trip type (the routing pre-phase).
    trip_type: TripType | None = None
    slots: dict[str, Any] = field(default_factory=_empty_slots)
    phase: Phase = Phase.COLLECTING
    last_asked: str | None = None
    # Slots already asked (so optionals are not asked twice). Persisted as a
    # list in JSON since sets are not JSON-serializable.
    asked: set[str] = field(default_factory=_empty_asked)
    # Failed-attempt counts per required slot (unusable answers only).
    attempts: dict[str, int] = field(default_factory=_empty_attempts)
    # Required slots given up on after too many failed attempts (persisted as a
    # list, like ``asked``).
    pending: set[str] = field(default_factory=_empty_pending)
    # The last message the bot sent, for answering follow-up questions ("¿y eso?").
    last_bot_message: str | None = None


def to_payload(state: ConversationState) -> dict[str, Any]:
    """Serialize state to a JSON-safe dict (the single transport-agnostic form).

    Sets become sorted lists; the trip-type enum becomes its value. Reused by
    every backend (Redis today, Postgres in increment 5) so the wire format
    never diverges.
    """
    return {
        "trip_type": state.trip_type.value if state.trip_type is not None else None,
        "slots": state.slots,
        "phase": state.phase.value,
        "last_asked": state.last_asked,
        "asked": sorted(state.asked),
        "attempts": state.attempts,
        "pending": sorted(state.pending),
        "last_bot_message": state.last_bot_message,
    }


def from_payload(data: dict[str, Any]) -> ConversationState:
    """Rebuild state from :func:`to_payload`'s dict.

    Tolerates payloads written before later fields existed (4a-core / 4a-extra-1).
    """
    stored_trip_type = data["trip_type"]
    return ConversationState(
        trip_type=TripType(stored_trip_type) if stored_trip_type is not None else None,
        slots=data["slots"],
        phase=Phase(data["phase"]),
        last_asked=data["last_asked"],
        asked=set(data.get("asked", [])),
        attempts=data.get("attempts", {}),
        pending=set(data.get("pending", [])),
        last_bot_message=data.get("last_bot_message"),
    )


class StateStore(Protocol):
    """Durable conversation-state store (the source of truth for state).

    A port, like the LLM port: the domain depends on this abstraction, not on a
    driver. ``conversation_id`` is the same per-conversation key the state used
    in Redis (the sender).
    """

    async def load(self, conversation_id: str) -> ConversationState | None:
        """Return the stored state, or ``None`` if the conversation is new."""
        ...

    async def save(self, conversation_id: str, state: ConversationState) -> None:
        """Persist (upsert) the state for ``conversation_id``."""
        ...


def merge_slots(
    existing: dict[str, Any], filled: dict[str, Any]
) -> dict[str, Any]:
    """Merge newly ``filled`` slots into ``existing`` (returns a new dict).

    A null incoming value (a slot not mentioned this turn) keeps the prior
    accumulated value -- this is self-contained, not reliant on the engine
    filtering top-level nulls. Non-null values are last-write-wins: scalars and
    lists replace; nested (dict) slots are deep-merged so partial info given
    across turns accumulates (e.g. adults in one turn, minor ages in the next).
    Inputs are not mutated.
    """
    merged = dict(existing)
    for name, value in filled.items():
        if value is None:
            continue  # top-level null: keep the accumulated value, never clobber
        current = merged.get(name)
        if isinstance(value, dict) and isinstance(current, dict):
            combined = dict(cast("dict[str, Any]", current))
            for inner_key, inner_value in cast("dict[str, Any]", value).items():
                if inner_value is not None:
                    combined[inner_key] = inner_value
            merged[name] = combined
        else:
            merged[name] = value
    return merged
