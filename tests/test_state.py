"""Tests for ConversationState: serialization (to/from payload) and merge_slots.

State persistence is exercised by the StateStore contract (test_storage.py); here
we test the transport-agnostic serialization and the slot-merge logic.
"""

from __future__ import annotations

import json

from app.domain.state import (
    ConversationState,
    Phase,
    from_payload,
    merge_slots,
    to_payload,
)
from app.understanding.schemas import TripType

# --- Serialization ----------------------------------------------------------


def test_serialization_round_trips() -> None:
    state = ConversationState(
        trip_type=TripType.CRUISE,
        slots={"nombre_cliente": "Ana", "pasajeros_crucero": {"adults": 2}},
        phase=Phase.HANDED_OFF,
        last_asked="fechas_crucero",
        asked={"ruta_crucero", "fechas_crucero"},
        attempts={"presupuesto_crucero": 2},
        pending={"presupuesto_crucero"},
        last_bot_message="¿En qué fechas?",
        chat_id="chat-uuid-123",
        lead_id=9364406,
        contact_id=552211,
    )

    restored = from_payload(to_payload(state))

    assert restored == state


def test_payload_is_json_safe() -> None:
    state = ConversationState(trip_type=TripType.EUROPE, asked={"a"}, pending={"b"})

    payload = to_payload(state)

    # Round-trips through JSON unchanged (sets serialized as lists, enum as value).
    assert from_payload(json.loads(json.dumps(payload))) == state


def test_from_payload_none_trip_type() -> None:
    state = ConversationState()  # not routed yet

    assert from_payload(to_payload(state)).trip_type is None


def test_from_payload_tolerates_legacy_payload_missing_late_fields() -> None:
    # A payload written before asked/attempts/pending/last_bot_message existed
    # (4a-core) deserializes cleanly with their defaults.
    legacy = {
        "trip_type": "cruise",
        "slots": {"nombre_cliente": "Ana"},
        "phase": "collecting",
        "last_asked": "ruta_crucero",
    }

    state = from_payload(legacy)

    assert state.slots == {"nombre_cliente": "Ana"}
    assert state.asked == set()
    assert state.attempts == {}
    assert state.pending == set()
    assert state.last_bot_message is None
    assert state.chat_id is None  # B1 field absent in legacy payloads
    assert state.inactivity_deadline is None  # inactivity-timer field absent too
    assert state.lead_id is None  # handoff idempotency marker absent in legacy
    assert state.contact_id is None


# --- merge_slots ------------------------------------------------------------


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
