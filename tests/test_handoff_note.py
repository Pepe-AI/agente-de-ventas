"""Offline tests for the handoff-note composer (pure: reason + pending -> str).

The note explains the WHY of the handoff (the reason) — it never repeats the trip
DATA, which already lives in the lead's custom fields. For STUCK it translates the
internal pending slot names to advisor-readable Spanish labels.
"""

from __future__ import annotations

from app.domain.handoff_note import compose_handoff_note
from app.domain.models import HandoffReason
from app.understanding.schemas import TripType, descriptor_for

_COMPLETE_NOTE = (
    "Calificación completa. El cliente proporcionó todos los datos requeridos. "
    "Listo para preparar propuesta."
)
_HUMAN_NOTE = "El cliente solicitó hablar directamente con un asesor."


def test_complete_note_is_the_approved_text() -> None:
    assert compose_handoff_note(HandoffReason.COMPLETE) == _COMPLETE_NOTE


def test_human_requested_note_is_the_approved_text() -> None:
    assert compose_handoff_note(HandoffReason.HUMAN_REQUESTED) == _HUMAN_NOTE


def test_stuck_note_single_pending_slot_uses_readable_label() -> None:
    note = compose_handoff_note(HandoffReason.STUCK, ["presupuesto_europa"])
    assert note == (
        "El bot no logró obtener: el presupuesto tras varios intentos. "
        "El resto de la información está en los campos del lead."
    )


def test_stuck_note_several_pending_slots_joined_with_y() -> None:
    note = compose_handoff_note(
        HandoffReason.STUCK,
        ["paises_europa", "fechas_europa", "presupuesto_europa"],
    )
    assert note == (
        "El bot no logró obtener: el destino, las fechas y el presupuesto "
        "tras varios intentos. El resto de la información está en los campos "
        "del lead."
    )


def test_stuck_note_two_pending_slots_joined_with_y() -> None:
    note = compose_handoff_note(
        HandoffReason.STUCK, ["fechas_asia", "pasajeros_asia"]
    )
    assert "El bot no logró obtener: las fechas y los pasajeros tras varios" in note


def test_stuck_note_labels_nombre_cliente_even_though_it_is_not_a_lead_field() -> None:
    # nombre_cliente is required (can be pending) but is set on the contact, so it
    # is absent from SLOT_CONCEPTS; the note still needs a label for it.
    note = compose_handoff_note(HandoffReason.STUCK, ["nombre_cliente"])
    assert "obtener: el nombre del cliente tras varios intentos" in note


def test_note_never_contains_internal_slot_names_or_data_values() -> None:
    # The composer only receives reason + slot NAMES, never slot values, so a
    # custom-field value can never leak into the note. It also renders a readable
    # label, not the raw internal slot name.
    note = compose_handoff_note(HandoffReason.STUCK, ["presupuesto_europa"])
    assert "presupuesto_europa" not in note
    for leaked_value in ("Italia", "5000", "Septiembre", "2 adultos"):
        assert leaked_value not in note


def test_every_required_slot_resolves_to_a_readable_label() -> None:
    # Only required slots can land in state.pending; each must translate to a
    # Spanish label (never surface its raw internal name in the note).
    for trip in TripType:
        for slot in descriptor_for(trip).slots:
            if not slot.required:
                continue
            note = compose_handoff_note(HandoffReason.STUCK, [slot.name])
            assert slot.name not in note, slot.name
