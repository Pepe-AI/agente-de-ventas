"""Offline tests for the isolated handoff orchestration (mocked CRM client).

Drives the agreed sequence (find-or-create -> note -> custom fields -> conditional
stage move) and the failure semantics (any CRM error propagates so the caller can
skip the flag/phase flip and retry next turn).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from structlog.testing import capture_logs

from app.crm.kommo_crm import KommoContactMatch, KommoCrmError
from app.domain.concepts import Concept
from app.domain.handoff_orchestration import (
    HandoffMapping,
    HandoffRunner,
    phone_from_sender,
)
from app.domain.models import HandoffReason

_MAPPING = HandoffMapping(
    concept_field_ids={Concept.DESTINO: 111, Concept.INVERSION: 222},
    reason_status_ids={
        HandoffReason.COMPLETE: 555,
        HandoffReason.STUCK: 666,
        HandoffReason.HUMAN_REQUESTED: 666,
    },
    pipeline_id=900,
    # 555=Calificado(20), 666=Atención(30), 777=No respondió(40).
    status_sort={555: 20, 666: 30, 777: 40},
)
_SLOTS: dict[str, object] = {
    "paises_europa": "Italia",
    "presupuesto_europa": {"defer_to_advisor": True},
}
_METHODS = {
    "find_contact_by_phone",
    "create_lead_with_contact",
    "get_lead",
    "add_note",
    "update_lead",
}


def _runner(client: Any) -> HandoffRunner:
    return HandoffRunner(client, _MAPPING)


def _call_order(client: Any) -> list[str]:
    return [c[0] for c in client.mock_calls if c[0] in _METHODS]


def _off_map_warning(logs: list[dict[str, Any]]) -> dict[str, Any]:
    return next(e for e in logs if e["event"] == "handoff_reused_lead_off_sort_map")


def test_phone_from_sender_strips_whatsapp_prefix() -> None:
    assert phone_from_sender("whatsapp:+5215512345678") == "+5215512345678"
    assert phone_from_sender("+5215512345678") == "+5215512345678"


async def test_new_lead_creates_writes_note_fields_and_moves_stage() -> None:
    client = AsyncMock()
    client.find_contact_by_phone.return_value = []  # no contact -> create
    client.create_lead_with_contact.return_value = 999

    lead_id = await _runner(client).run(
        reason=HandoffReason.COMPLETE,
        phone="+5215500",
        customer_name="Ana",
        slots=_SLOTS,
        pending=(),
    )

    assert lead_id == 999
    client.create_lead_with_contact.assert_awaited_once_with("Ana", "Ana", "+5215500")
    client.get_lead.assert_not_awaited()  # new lead: no status read needed
    client.add_note.assert_awaited_once()
    _, kwargs = client.update_lead.await_args
    assert kwargs["status_id"] == 555  # COMPLETE -> Calificado
    assert kwargs["pipeline_id"] == 900
    assert kwargs["custom_fields_values"]  # non-empty


async def test_note_is_written_before_the_publishing_update() -> None:
    client = AsyncMock()
    client.find_contact_by_phone.return_value = []
    client.create_lead_with_contact.return_value = 1

    await _runner(client).run(
        reason=HandoffReason.COMPLETE,
        phone="+1",
        customer_name="X",
        slots=_SLOTS,
        pending=(),
    )

    order = _call_order(client)
    assert order.index("add_note") < order.index("update_lead")


async def test_reused_lead_behind_target_is_moved_forward() -> None:
    # Reused lead in Calificado (sort 20); STUCK targets Atención (sort 30).
    # 20 <= 30 -> still behind the target -> publish (move forward).
    client = AsyncMock()
    client.find_contact_by_phone.return_value = [
        KommoContactMatch(11, (10, 12)),
        KommoContactMatch(22, (7,)),
    ]
    client.get_lead.return_value = {"id": 12, "status_id": 555}  # Calificado (20)

    lead_id = await _runner(client).run(
        reason=HandoffReason.STUCK,
        phone="+1",
        customer_name="X",
        slots=_SLOTS,
        pending=["presupuesto_europa"],
    )

    assert lead_id == 12  # the most recent (highest id) across all matches
    client.create_lead_with_contact.assert_not_awaited()
    client.get_lead.assert_awaited_once_with(12)
    _, kwargs = client.update_lead.await_args
    assert kwargs["status_id"] == 666  # STUCK -> Atención 1 a 1
    assert kwargs["pipeline_id"] == 900
    assert kwargs["custom_fields_values"]


async def test_reused_lead_at_target_stage_is_republished() -> None:
    # Already in the reason's target stage (30 <= 30) -> move (a stage no-op, but
    # the custom fields are still written). Covers the failed-retry re-publish.
    client = AsyncMock()
    client.find_contact_by_phone.return_value = [KommoContactMatch(1, (5,))]
    client.get_lead.return_value = {"id": 5, "status_id": 666}  # Atención (30)

    await _runner(client).run(
        reason=HandoffReason.STUCK,
        phone="+1",
        customer_name="X",
        slots=_SLOTS,
        pending=["presupuesto_europa"],
    )

    _, kwargs = client.update_lead.await_args
    assert kwargs["status_id"] == 666
    assert kwargs["pipeline_id"] == 900
    assert kwargs["custom_fields_values"]


async def test_reused_lead_advanced_by_advisor_is_not_moved() -> None:
    # In "No respondió" (sort 40), past the STUCK target (30): 40 > 30 -> the advisor
    # advanced it; never move it backward. Fields are still written.
    client = AsyncMock()
    client.find_contact_by_phone.return_value = [KommoContactMatch(1, (5,))]
    client.get_lead.return_value = {"id": 5, "status_id": 777}  # No respondió (40)

    await _runner(client).run(
        reason=HandoffReason.STUCK,
        phone="+1",
        customer_name="X",
        slots=_SLOTS,
        pending=["presupuesto_europa"],
    )

    _, kwargs = client.update_lead.await_args
    assert "status_id" not in kwargs
    assert "pipeline_id" not in kwargs
    assert kwargs["custom_fields_values"]  # fields still overwritten


async def test_reused_lead_off_sort_map_is_not_moved_and_warns() -> None:
    # status_id outside STATUS_SORT (unknown stage, or another pipeline) -> assume
    # the advisor owns it -> don't move, and warn (the operating assumption broke).
    client = AsyncMock()
    client.find_contact_by_phone.return_value = [KommoContactMatch(1, (5,))]
    client.get_lead.return_value = {"id": 5, "status_id": 99999, "pipeline_id": 42}

    with capture_logs() as logs:
        await _runner(client).run(
            reason=HandoffReason.COMPLETE,
            phone="+1",
            customer_name="X",
            slots=_SLOTS,
            pending=(),
        )

    _, kwargs = client.update_lead.await_args
    assert "status_id" not in kwargs
    assert "pipeline_id" not in kwargs
    assert kwargs["custom_fields_values"]  # fields still overwritten
    warning = _off_map_warning(logs)
    assert warning["status_id"] == 99999
    assert warning["pipeline_id"] == 42


async def test_reused_lead_with_missing_status_is_not_moved_and_warns() -> None:
    # A malformed lead with no status_id -> current_sort is None -> don't move + warn.
    client = AsyncMock()
    client.find_contact_by_phone.return_value = [KommoContactMatch(1, (5,))]
    client.get_lead.return_value = {"id": 5}  # no status_id

    with capture_logs() as logs:
        await _runner(client).run(
            reason=HandoffReason.COMPLETE,
            phone="+1",
            customer_name="X",
            slots=_SLOTS,
            pending=(),
        )

    _, kwargs = client.update_lead.await_args
    assert "status_id" not in kwargs
    assert "pipeline_id" not in kwargs
    assert kwargs["custom_fields_values"]  # fields still overwritten
    warning = _off_map_warning(logs)
    assert warning["status_id"] is None
    assert warning["pipeline_id"] is None


async def test_create_failure_propagates_before_note() -> None:
    client = AsyncMock()
    client.find_contact_by_phone.return_value = []
    client.create_lead_with_contact.side_effect = KommoCrmError("boom")

    with pytest.raises(KommoCrmError):
        await _runner(client).run(
            reason=HandoffReason.COMPLETE,
            phone="+1",
            customer_name="X",
            slots=_SLOTS,
            pending=(),
        )
    client.add_note.assert_not_awaited()


async def test_add_note_failure_propagates_without_update() -> None:
    client = AsyncMock()
    client.find_contact_by_phone.return_value = []
    client.create_lead_with_contact.return_value = 1
    client.add_note.side_effect = KommoCrmError("boom")

    with pytest.raises(KommoCrmError):
        await _runner(client).run(
            reason=HandoffReason.COMPLETE,
            phone="+1",
            customer_name="X",
            slots=_SLOTS,
            pending=(),
        )
    client.update_lead.assert_not_awaited()


async def test_update_failure_propagates() -> None:
    client = AsyncMock()
    client.find_contact_by_phone.return_value = []
    client.create_lead_with_contact.return_value = 1
    client.update_lead.side_effect = KommoCrmError("boom")

    with pytest.raises(KommoCrmError):
        await _runner(client).run(
            reason=HandoffReason.COMPLETE,
            phone="+1",
            customer_name="X",
            slots=_SLOTS,
            pending=(),
        )
