"""Compose the handoff note: the text the advisor reads when opening the lead.

Pure: ``(reason, pending) -> str``. No network, no Kommo, no I/O. The note explains
the WHY of the handoff (the reason). It deliberately does NOT repeat the trip DATA
— destination, dates, passengers, budget, ... — because that already went to the
lead's custom fields (``build_custom_fields_values``). Restating it here would be
duplication; the note's job is the context the shared "Atención 1 a 1" funnel loses
(atorado and pidió_humano land together, so the note says why each lead is there).

It receives only the reason and the pending slot NAMES, never slot values, so a
custom-field value cannot leak into the note by construction. For STUCK it
translates the internal pending slot names to readable Spanish labels, reusing the
shared slot->concept normalization (:mod:`app.domain.concepts`) plus a concept->
label table; ``nombre_cliente`` (a required slot set on the contact, hence absent
from ``SLOT_CONCEPTS``) is labeled here too.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.domain.concepts import SLOT_CONCEPTS, Concept
from app.domain.models import HandoffReason

_COMPLETE_NOTE = (
    "Calificación completa. El cliente proporcionó todos los datos requeridos. "
    "Listo para preparar propuesta."
)
_HUMAN_REQUESTED_NOTE = "El cliente solicitó hablar directamente con un asesor."
_NO_RESPONSE_NOTE = (
    "El cliente dejó de responder tras iniciar la conversación. "
    "Transferido por inactividad para seguimiento."
)
_STUCK_TEMPLATE = (
    "El bot no logró obtener: {labels} tras varios intentos. "
    "El resto de la información está en los campos del lead."
)

# Concept -> advisor-readable Spanish label (complete: every concept has one, so a
# slot that becomes required later already reads well in the note).
_CONCEPT_LABELS: dict[Concept, str] = {
    Concept.DESTINO: "el destino",
    Concept.FECHA: "las fechas",
    Concept.DURACION: "la duración",
    Concept.PASAJEROS: "los pasajeros",
    Concept.INVERSION: "el presupuesto",
    Concept.CIUDAD_SALIDA: "la ciudad de salida",
    Concept.SERVICIOS: "los servicios adicionales",
    Concept.NIVEL_HOSPEDAJE: "el nivel de hospedaje",
    Concept.VUELOS: "los vuelos",
    Concept.OCASION: "la ocasión del viaje",
    Concept.DOCUMENTACION: "la documentación",
    Concept.CABINAS: "las cabinas",
    Concept.TIPO_CABINA: "el tipo de cabina",
    Concept.EXPERIENCIA_CRUCERO: "la experiencia a bordo",
}

# Required slots that can be pending but are NOT lead custom fields (absent from
# SLOT_CONCEPTS), so they need a label here. nombre_cliente is set on the contact.
_EXTRA_SLOT_LABELS: dict[str, str] = {
    "nombre_cliente": "el nombre del cliente",
}


def compose_handoff_note(
    reason: HandoffReason, pending: Sequence[str] = ()
) -> str:
    """Return the advisor-facing note for a handoff with ``reason``.

    ``pending`` (the required slots given up on) is used only for STUCK, where the
    slot names are translated to readable labels. The dispatch is left open for a
    future reason (the inactivity timer) without building that branch here.
    """
    if reason is HandoffReason.COMPLETE:
        return _COMPLETE_NOTE
    if reason is HandoffReason.HUMAN_REQUESTED:
        return _HUMAN_REQUESTED_NOTE
    if reason is HandoffReason.NO_RESPONSE:
        return _NO_RESPONSE_NOTE
    if reason is HandoffReason.STUCK:
        return _STUCK_TEMPLATE.format(labels=_join_labels(_pending_labels(pending)))
    raise ValueError(f"unsupported handoff reason: {reason}")  # defensive


def _pending_labels(pending: Sequence[str]) -> list[str]:
    """Translate pending slot names to readable labels, in order.

    No dedup needed: a conversation has one trip type and each concept has a single
    required slot per type, so pending never holds two slots of the same concept.
    """
    return [_label_for(slot_name) for slot_name in pending]


def _label_for(slot_name: str) -> str:
    concept = SLOT_CONCEPTS.get(slot_name)
    if concept is not None:
        return _CONCEPT_LABELS[concept]
    # Fallback to the raw name so a gap is visible, never silently dropped; a test
    # asserts every required slot resolves to a real label, so this never fires.
    return _EXTRA_SLOT_LABELS.get(slot_name, slot_name)


def _join_labels(labels: list[str]) -> str:
    """Join labels as a natural Spanish list: 'a, b y c' (or 'a' / 'a y b')."""
    if len(labels) <= 1:
        return labels[0] if labels else ""
    return f"{', '.join(labels[:-1])} y {labels[-1]}"
