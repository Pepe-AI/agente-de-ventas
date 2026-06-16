"""Tests for the persistent conversation state (Redis-backed, per sender)."""

from __future__ import annotations

import json

from fakeredis import FakeAsyncRedis

from app.concurrency.keys import KeyPrefix, make_key
from app.domain.state import Phase, load_state, merge_slots, save_state
from app.understanding.schemas import TripType

SENDER = "whatsapp:+5215512345678"


def test_merge_replaces_scalar_slots() -> None:
    merged = merge_slots({"nombre_cliente": "Ana"}, {"ruta_crucero": "Caribe"})

    assert merged == {"nombre_cliente": "Ana", "ruta_crucero": "Caribe"}


def test_merge_deep_merges_nested_keeping_prior_non_null() -> None:
    # Turn 1 captured adults + that minors exist; turn 2 adds only the ages.
    existing = {
        "pasajeros_crucero": {
            "adults": 2,
            "minors_mentioned": True,
            "minor_ages": None,
        }
    }
    incoming = {
        "pasajeros_crucero": {
            "adults": None,
            "minors_mentioned": None,
            "minor_ages": [8],
        }
    }

    merged = merge_slots(existing, incoming)

    assert merged["pasajeros_crucero"] == {
        "adults": 2,
        "minors_mentioned": True,
        "minor_ages": [8],
    }


def test_merge_does_not_mutate_inputs() -> None:
    existing = {"pasajeros_crucero": {"adults": 2}}

    merge_slots(existing, {"pasajeros_crucero": {"minor_ages": [8]}})

    assert existing == {"pasajeros_crucero": {"adults": 2}}


def test_merge_corrects_scalar_on_same_key() -> None:
    # Case 3: a new non-null value for an already-filled scalar wins.
    merged = merge_slots({"fechas_crucero": "julio"}, {"fechas_crucero": "agosto"})

    assert merged == {"fechas_crucero": "agosto"}


def test_merge_replaces_list_not_concatenate() -> None:
    # Case 4: a new list replaces the prior list; it is never concatenated.
    existing = {"pasajeros_crucero": {"minor_ages": [8, 12]}}
    incoming = {"pasajeros_crucero": {"minor_ages": [8, 10]}}

    merged = merge_slots(existing, incoming)

    assert merged["pasajeros_crucero"]["minor_ages"] == [8, 10]


def test_merge_preserves_prior_on_top_level_null() -> None:
    # Case 5a: a top-level null (slot not mentioned this turn) keeps the prior
    # value instead of clobbering it -- without relying on the engine's filter.
    merged = merge_slots({"nombre_cliente": "Ana"}, {"nombre_cliente": None})

    assert merged == {"nombre_cliente": "Ana"}


async def test_load_initializes_when_absent() -> None:
    redis = FakeAsyncRedis(decode_responses=True)

    state = await load_state(redis, SENDER, TripType.CRUISE)

    assert state.trip_type is TripType.CRUISE
    assert state.slots == {}
    assert state.phase is Phase.COLLECTING
    assert state.last_asked is None


async def test_save_then_load_round_trips() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    state = await load_state(redis, SENDER, TripType.CRUISE)
    state.slots = {
        "nombre_cliente": "Ana",
        "pasajeros_crucero": {"adults": 2, "minors_mentioned": False},
    }
    state.last_asked = "fechas_crucero"
    state.phase = Phase.COLLECTING

    await save_state(redis, SENDER, state)
    reloaded = await load_state(redis, SENDER, TripType.CRUISE)

    assert reloaded.slots == state.slots
    assert reloaded.last_asked == "fechas_crucero"
    assert reloaded.trip_type is TripType.CRUISE


async def test_asked_defaults_empty_when_absent() -> None:
    redis = FakeAsyncRedis(decode_responses=True)

    state = await load_state(redis, SENDER, TripType.CRUISE)

    assert state.asked == set()


async def test_asked_set_persists_round_trip() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    state = await load_state(redis, SENDER, TripType.CRUISE)
    state.asked = {"nombre_cliente", "cabinas_crucero"}

    await save_state(redis, SENDER, state)
    reloaded = await load_state(redis, SENDER, TripType.CRUISE)

    assert reloaded.asked == {"nombre_cliente", "cabinas_crucero"}


async def test_load_tolerates_state_without_asked_key() -> None:
    # A state persisted by 4a-core (before `asked` existed) loads cleanly.
    redis = FakeAsyncRedis(decode_responses=True)
    legacy = json.dumps(
        {
            "trip_type": "cruise",
            "slots": {"nombre_cliente": "Ana"},
            "phase": "collecting",
            "last_asked": "ruta_crucero",
        }
    )
    await redis.set(make_key(KeyPrefix.STATE, SENDER), legacy)

    state = await load_state(redis, SENDER, TripType.CRUISE)

    assert state.asked == set()
    assert state.slots == {"nombre_cliente": "Ana"}


async def test_attempts_and_pending_default_empty() -> None:
    redis = FakeAsyncRedis(decode_responses=True)

    state = await load_state(redis, SENDER, TripType.CRUISE)

    assert state.attempts == {}
    assert state.pending == set()


async def test_attempts_and_pending_persist_round_trip() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    state = await load_state(redis, SENDER, TripType.CRUISE)
    state.attempts = {"presupuesto_crucero": 2}
    state.pending = {"fechas_crucero"}

    await save_state(redis, SENDER, state)
    reloaded = await load_state(redis, SENDER, TripType.CRUISE)

    assert reloaded.attempts == {"presupuesto_crucero": 2}
    assert reloaded.pending == {"fechas_crucero"}


async def test_load_tolerates_state_without_attempts_or_pending() -> None:
    # A state persisted by 4a-extra-1 (before attempts/pending) loads cleanly.
    redis = FakeAsyncRedis(decode_responses=True)
    legacy = json.dumps(
        {
            "trip_type": "cruise",
            "slots": {},
            "phase": "collecting",
            "last_asked": None,
            "asked": ["nombre_cliente"],
        }
    )
    await redis.set(make_key(KeyPrefix.STATE, SENDER), legacy)

    state = await load_state(redis, SENDER, TripType.CRUISE)

    assert state.attempts == {}
    assert state.pending == set()
    assert state.asked == {"nombre_cliente"}


async def test_stored_trip_type_wins_over_default() -> None:
    # Once a conversation started on a schema, a different default must not switch it.
    redis = FakeAsyncRedis(decode_responses=True)
    state = await load_state(redis, SENDER, TripType.EUROPE)
    await save_state(redis, SENDER, state)

    reloaded = await load_state(redis, SENDER, TripType.CRUISE)

    assert reloaded.trip_type is TripType.EUROPE
