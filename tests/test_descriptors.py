"""Tests for the real trip descriptors and the derived extraction models.

A descriptor lists every slot with its type, whether it is required, and any
special completeness rule. The engine-facing *extraction model* is derived from
it: a pure Pydantic model of business slots only (no ``question``).
"""

from __future__ import annotations

from app.understanding.engine import QUESTION_FIELD
from app.understanding.schemas import (
    Budget,
    Passengers,
    SlotRule,
    TripType,
    descriptor_for,
    extraction_model_for,
)

CRUISE_SLOT_ORDER = [
    "nombre_cliente",
    "ruta_crucero",
    "fechas_crucero",
    "pasajeros_crucero",
    "cabinas_crucero",
    "tipo_cabina",
    "experiencia_crucero",
    "ciudad_salida_crucero",
    "presupuesto_crucero",
    "pasaporte_crucero",
    "servicios_crucero",
]

CRUISE_REQUIRED = {
    "nombre_cliente",
    "ruta_crucero",
    "fechas_crucero",
    "pasajeros_crucero",
    "presupuesto_crucero",
}


def test_cruise_descriptor_keeps_slot_order() -> None:
    descriptor = descriptor_for(TripType.CRUISE)

    assert [slot.name for slot in descriptor.slots] == CRUISE_SLOT_ORDER


def test_cruise_required_slots() -> None:
    descriptor = descriptor_for(TripType.CRUISE)

    required = {slot.name for slot in descriptor.slots if slot.required}

    assert required == CRUISE_REQUIRED


def test_cruise_destination_has_no_experience_escape() -> None:
    descriptor = descriptor_for(TripType.CRUISE)
    ruta = next(s for s in descriptor.slots if s.name == "ruta_crucero")

    assert ruta.rule is SlotRule.DESTINATION
    assert ruta.escape_slot is None


def test_europe_destination_escapes_to_experience() -> None:
    descriptor = descriptor_for(TripType.EUROPE)
    paises = next(s for s in descriptor.slots if s.name == "paises_europa")

    assert paises.rule is SlotRule.DESTINATION
    assert paises.escape_slot == "experiencia_europa"


def test_asia_mirrors_europe_with_suffix() -> None:
    descriptor = descriptor_for(TripType.ASIA)
    names = {slot.name for slot in descriptor.slots}

    assert "destinos_asia" in names
    assert "experiencia_asia" in names
    destino = next(s for s in descriptor.slots if s.name == "destinos_asia")
    assert destino.escape_slot == "experiencia_asia"


def test_experience_escape_slots_are_not_askable() -> None:
    # The destination-experience escapes are captured passively, never asked.
    for trip, name in (
        (TripType.EUROPE, "experiencia_europa"),
        (TripType.ASIA, "experiencia_asia"),
    ):
        slot = next(s for s in descriptor_for(trip).slots if s.name == name)
        assert slot.askable is False


def test_cruise_experience_is_askable() -> None:
    descriptor = descriptor_for(TripType.CRUISE)
    slot = next(s for s in descriptor.slots if s.name == "experiencia_crucero")

    assert slot.askable is True


def test_slots_are_askable_by_default() -> None:
    descriptor = descriptor_for(TripType.CRUISE)
    slot = next(s for s in descriptor.slots if s.name == "nombre_cliente")

    assert slot.askable is True


def test_extraction_model_is_pure_no_question() -> None:
    model = extraction_model_for(TripType.CRUISE)

    assert QUESTION_FIELD not in model.model_fields
    assert set(CRUISE_SLOT_ORDER) <= set(model.model_fields)


def test_extraction_model_uses_nested_models_for_special_slots() -> None:
    model = extraction_model_for(TripType.CRUISE)

    instance = model(
        pasajeros_crucero=Passengers(adults=2, minors_mentioned=False),
        presupuesto_crucero=Budget(defer_to_advisor=True),
    )
    dumped = instance.model_dump()

    assert dumped["pasajeros_crucero"] == {
        "adults": 2,
        "minors_mentioned": False,
        "minor_ages": None,
    }
    assert dumped["presupuesto_crucero"] == {
        "amount": None,
        "defer_to_advisor": True,
    }
    # Slots the model could not fill stay null (never invented).
    assert dumped["nombre_cliente"] is None


def test_slot_prompts_use_formal_usted_register() -> None:
    # Spot-check the tone change (tú -> usted) across the three trip types.
    cruise = {s.name: s.prompt for s in descriptor_for(TripType.CRUISE).slots}
    assert cruise["nombre_cliente"] == "Para empezar, ¿me podría compartir su nombre?"
    assert "presupuesto aproximado en mente" in cruise["presupuesto_crucero"]
    assert cruise["pasaporte_crucero"] == "¿Cuenta con pasaporte vigente?"
    # "Por último" lives on the real last askable slot (servicios), not pasaporte.
    assert cruise["servicios_crucero"].startswith("Por último, ")

    europe = {s.name: s.prompt for s in descriptor_for(TripType.EUROPE).slots}
    assert "le gustaría visitar" in europe["paises_europa"]
    assert "incluyamos los vuelos en la propuesta" in europe["vuelos_europa"]
    assert europe["servicios_europa"].startswith("Por último, ")

    asia = {s.name: s.prompt for s in descriptor_for(TripType.ASIA).slots}
    assert asia["nombre_cliente"] == "Para empezar, ¿me podría compartir su nombre?"
    assert "presupuesto aproximado en mente" in asia["presupuesto_asia"]
