"""Slot completeness: the orchestrator's required/missing logic.

The engine only extracts; deciding what is still required lives here. A slot is
judged against the captured ``slots`` state (values are plain JSON-friendly
dicts/scalars, the form they take after merge and Redis round-trip). The two
structured slots are re-validated into their models so the rules read typed
attributes instead of raw dict keys.
"""

from __future__ import annotations

from app.understanding.schemas import (
    Budget,
    Passengers,
    SlotRule,
    SlotSpec,
    TripSchema,
)


def _budget_satisfied(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    budget = Budget.model_validate(value)
    return budget.amount is not None or bool(budget.defer_to_advisor)


def _passengers_satisfied(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    passengers = Passengers.model_validate(value)
    if passengers.adults is None:
        return False
    # Minor ages are optional: a mention of minors is enough, we never block on
    # missing ages. adults (above) is the only required field.
    return True


def is_satisfied(slot: SlotSpec, slots: dict[str, object]) -> bool:
    """Return whether ``slot`` is satisfied given the captured ``slots``."""
    value = slots.get(slot.name)
    match slot.rule:
        case SlotRule.PLAIN:
            return value is not None
        case SlotRule.DESTINATION:
            if value is not None:
                return True
            if slot.escape_slot is None:
                return False
            return slots.get(slot.escape_slot) is not None
        case SlotRule.BUDGET:
            return _budget_satisfied(value)
        case SlotRule.PASSENGERS:
            return _passengers_satisfied(value)


def next_required_slot(
    descriptor: TripSchema, slots: dict[str, object]
) -> SlotSpec | None:
    """Return the first required, unsatisfied slot in order, or ``None``."""
    for slot in descriptor.slots:
        if slot.required and not is_satisfied(slot, slots):
            return slot
    return None


def next_slot_to_ask(
    descriptor: TripSchema,
    slots: dict[str, object],
    asked: set[str],
    pending: set[str],
) -> SlotSpec | None:
    """Return the next askable slot to ask in flow order, or ``None``.

    A slot is asked when it is askable and either required-and-unsatisfied (and
    not given up on), or optional-and-neither-asked-nor-already-satisfied (so
    volunteered values are skipped). A required slot in ``pending`` is treated
    as resolved. ``None`` means nothing askable is left: every required slot is
    satisfied or pending and every askable optional was asked -> ready to
    complete.
    """
    for slot in descriptor.slots:
        if not slot.askable:
            continue
        if is_satisfied(slot, slots):
            continue
        if slot.required:
            if slot.name not in pending:
                return slot
        elif slot.name not in asked:
            return slot
    return None
