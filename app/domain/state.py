"""Persistent per-sender conversation state (Redis-backed).

Holds the trip type (which schema is in use), the slots captured so far, the
conversation phase, and the last slot the bot asked for. Persisted as JSON so it
survives between turns and across workers; the slot values are already
JSON-friendly (scalars / lists / nested dicts).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

from redis.asyncio import Redis

from app.concurrency.keys import KeyPrefix, make_key
from app.understanding.schemas import TripType


class Phase(StrEnum):
    """Where the conversation is in its lifecycle."""

    COLLECTING = "collecting"
    COMPLETED = "completed"


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

    trip_type: TripType
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


def _key(sender: str) -> str:
    return make_key(KeyPrefix.STATE, sender)


async def load_state(
    redis: Redis, sender: str, default_trip_type: TripType
) -> ConversationState:
    """Load the sender's state, or initialize it on ``default_trip_type``.

    A stored trip type wins over the default: once a conversation started on a
    schema, the configured default must not switch it mid-conversation.
    """
    raw = await redis.get(_key(sender))
    if raw is None:
        return ConversationState(trip_type=default_trip_type)
    data: dict[str, Any] = json.loads(raw)
    return ConversationState(
        trip_type=TripType(data["trip_type"]),
        slots=data["slots"],
        phase=Phase(data["phase"]),
        last_asked=data["last_asked"],
        # Tolerate states persisted before these fields existed (4a-core /
        # 4a-extra-1).
        asked=set(data.get("asked", [])),
        attempts=data.get("attempts", {}),
        pending=set(data.get("pending", [])),
    )


async def save_state(redis: Redis, sender: str, state: ConversationState) -> None:
    """Persist the sender's state."""
    payload = json.dumps(
        {
            "trip_type": state.trip_type.value,
            "slots": state.slots,
            "phase": state.phase.value,
            "last_asked": state.last_asked,
            "asked": sorted(state.asked),
            "attempts": state.attempts,
            "pending": sorted(state.pending),
        }
    )
    await redis.set(_key(sender), payload)
