"""Tests for slot completeness — the orchestrator's required/missing logic.

Completeness is computed from the descriptor metadata + the captured state
(values stored as plain JSON-friendly dicts, as they are after merge/persist).
"""

from __future__ import annotations

from app.domain.completeness import is_satisfied, next_required_slot
from app.understanding.schemas import SlotRule, SlotSpec, TripType, descriptor_for

_PLAIN = SlotSpec("nombre_cliente", str, True, "?")
_DESTINATION_ESCAPE = SlotSpec(
    "paises_europa", str, True, "?", rule=SlotRule.DESTINATION,
    escape_slot="experiencia_europa",
)
_DESTINATION_NO_ESCAPE = SlotSpec(
    "ruta_crucero", str, True, "?", rule=SlotRule.DESTINATION,
)
_BUDGET = SlotSpec("presupuesto", str, True, "?", rule=SlotRule.BUDGET)
_PASSENGERS = SlotSpec("pasajeros", str, True, "?", rule=SlotRule.PASSENGERS)


# --- PLAIN ------------------------------------------------------------------


def test_plain_satisfied_when_non_null() -> None:
    assert is_satisfied(_PLAIN, {"nombre_cliente": "Ana"})


def test_plain_unsatisfied_when_absent() -> None:
    assert not is_satisfied(_PLAIN, {})


# --- DESTINATION ------------------------------------------------------------


def test_destination_satisfied_by_concrete_value() -> None:
    assert is_satisfied(_DESTINATION_ESCAPE, {"paises_europa": "Italia"})


def test_destination_satisfied_by_experience_escape() -> None:
    # No concrete country, but a free-text experience was captured instead.
    state = {"experiencia_europa": "algo romántico y tranquilo"}
    assert is_satisfied(_DESTINATION_ESCAPE, state)


def test_destination_without_escape_needs_concrete_value() -> None:
    # A cruise route has no experience escape; an unrelated field does not count.
    state = {"experiencia_crucero": "fiesta a bordo"}
    assert not is_satisfied(_DESTINATION_NO_ESCAPE, state)


# --- BUDGET -----------------------------------------------------------------


def test_budget_satisfied_by_amount() -> None:
    assert is_satisfied(_BUDGET, {"presupuesto": {"amount": "2000-3000 USD"}})


def test_budget_satisfied_by_defer_to_advisor() -> None:
    assert is_satisfied(_BUDGET, {"presupuesto": {"defer_to_advisor": True}})


def test_budget_unsatisfied_when_empty() -> None:
    state = {"presupuesto": {"amount": None, "defer_to_advisor": None}}
    assert not is_satisfied(_BUDGET, state)


# --- PASSENGERS -------------------------------------------------------------


def test_passengers_satisfied_with_adults_and_no_minors() -> None:
    state = {"pasajeros": {"adults": 2, "minors_mentioned": False}}
    assert is_satisfied(_PASSENGERS, state)


def test_passengers_unsatisfied_without_adults() -> None:
    assert not is_satisfied(_PASSENGERS, {"pasajeros": {"adults": None}})


def test_passengers_unsatisfied_when_minors_mentioned_without_ages() -> None:
    state = {"pasajeros": {"adults": 2, "minors_mentioned": True, "minor_ages": None}}
    assert not is_satisfied(_PASSENGERS, state)


def test_passengers_satisfied_when_minor_ages_present() -> None:
    state = {"pasajeros": {"adults": 2, "minors_mentioned": True, "minor_ages": [8]}}
    assert is_satisfied(_PASSENGERS, state)


# --- next_required_slot -----------------------------------------------------


def test_next_required_is_first_unsatisfied_in_order() -> None:
    descriptor = descriptor_for(TripType.CRUISE)
    nxt = next_required_slot(descriptor, {})

    assert nxt is not None
    assert nxt.name == "nombre_cliente"


def test_next_required_skips_satisfied_required_slots() -> None:
    descriptor = descriptor_for(TripType.CRUISE)
    state = {"nombre_cliente": "Ana", "ruta_crucero": "Caribe"}

    nxt = next_required_slot(descriptor, state)

    assert nxt is not None
    assert nxt.name == "fechas_crucero"


def test_next_required_ignores_optional_slots() -> None:
    descriptor = descriptor_for(TripType.CRUISE)
    # All required filled, every optional left empty -> nothing left to ask.
    state = {
        "nombre_cliente": "Ana",
        "ruta_crucero": "Caribe",
        "fechas_crucero": "julio",
        "pasajeros_crucero": {"adults": 2, "minors_mentioned": False},
        "presupuesto_crucero": {"defer_to_advisor": True},
    }

    assert next_required_slot(descriptor, state) is None
