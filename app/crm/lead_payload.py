"""Build Kommo's ``custom_fields_values`` from the conversation slot state (core).

Client-agnostic: it takes the per-client mapping (slot->concept, concept->field_id)
as parameters, so the same logic serves every client. It does three things the
Kommo API needs (option B — only filled slots are written):

* normalize the suffix-per-trip slot names onto the shared concepts (several slot
  names collapse to one concept; the first FILLED slot per concept in mapping
  order wins, so a concrete destination beats its experience escape);
* serialize the structured slots (Budget / Passengers) to advisor-readable text,
  since the Kommo fields are plain text;
* omit any slot that is empty or has no mapped concept — Kommo rejects a custom
  field with no value, so absent slots simply do not appear in the payload.

The result is Kommo's shape: ``[{"field_id": int, "values": [{"value": str}]}]``.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import cast

from app.understanding.schemas import Budget, Passengers

# Prefix for a destino field filled from an experience escape (no concrete
# destination given): it is a stated preference, not a confirmed destination.
_ESCAPE_PREFIX = "Sin destino definido — busca: "


def build_custom_fields_values[ConceptT: str](
    slots: dict[str, object],
    *,
    slot_concepts: Mapping[str, ConceptT],
    concept_field_ids: Mapping[ConceptT, int],
    escape_slots: Collection[str] = (),
) -> list[dict[str, object]]:
    """Return the ``custom_fields_values`` for the filled, mapped slots.

    ``slot_concepts`` is iterated in order: the first slot that both maps to a
    concept and has a serializable value claims that concept (mapping order is
    the precedence — concrete-before-escape). Empty/unmapped slots are skipped.

    A concept won by a slot in ``escape_slots`` (a value captured passively, not a
    confirmed answer — see ``schemas.escape_slot_names``) gets its text prefixed so
    the advisor sees it is a preference, not a confirmed value. A concrete slot
    that wins the same concept is written raw (it comes first, so it wins).

    Generic over the concept key (a ``str`` subtype, e.g. a ``StrEnum``) so both
    mapping tables stay the same concept type without invariance friction.
    """
    result: list[dict[str, object]] = []
    seen: set[ConceptT] = set()
    for slot_name, concept in slot_concepts.items():
        if concept in seen:
            continue
        text = _serialize(slots.get(slot_name))
        if text is None:
            continue
        if slot_name in escape_slots:
            text = f"{_ESCAPE_PREFIX}{text}"
        result.append(
            {"field_id": concept_field_ids[concept], "values": [{"value": text}]}
        )
        seen.add(concept)
    return result


def _serialize(value: object) -> str | None:
    """Serialize a slot value to non-empty text, or ``None`` to omit the field."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        return _serialize_structured(cast("dict[str, object]", value))
    # Plain slots are strings; anything else is rendered defensively.
    return str(value).strip() or None


def _serialize_structured(value: dict[str, object]) -> str | None:
    """Serialize a Budget/Passengers sub-model dict, dispatching on its keys."""
    if value.keys() & {"adults", "minors_mentioned", "minor_ages"}:
        return _serialize_passengers(Passengers.model_validate(value))
    if value.keys() & {"amount", "defer_to_advisor"}:
        return _serialize_budget(Budget.model_validate(value))
    return None


def _serialize_budget(budget: Budget) -> str | None:
    if budget.defer_to_advisor:
        return "Prefiere revisarlo con un asesor"
    if budget.amount:
        return budget.amount.strip() or None
    return None


def _serialize_passengers(passengers: Passengers) -> str | None:
    parts: list[str] = []
    if passengers.adults is not None:
        unit = "adulto" if passengers.adults == 1 else "adultos"
        parts.append(f"{passengers.adults} {unit}")
    if passengers.minor_ages:
        count = len(passengers.minor_ages)
        unit = "menor" if count == 1 else "menores"
        ages = ", ".join(str(age) for age in passengers.minor_ages)
        parts.append(f"{count} {unit} ({ages})")
    elif passengers.minors_mentioned:
        parts.append("menores (edades sin especificar)")
    return ", ".join(parts) or None
