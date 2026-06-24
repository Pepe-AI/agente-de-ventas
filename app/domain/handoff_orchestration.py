"""Handoff orchestration: given a reason + conversation data, run the CRM sequence.

Isolated and testable: the CRM client and the per-client funnel mapping are
INJECTED, so the core never imports the ``_topviajes`` mapping directly (the
composition root provides the real client + mapping). It only sequences the
existing, validated primitives — it builds none of them.

The sequence (order matters — "a half-done lead is worse than no lead", so the
publishing PATCH and the caller's flag flip come last):

1. find-or-create: reuse the most recent existing lead (highest id — Kommo ids are
   monotonic and a phone search returns no timestamps), else create a new one;
2. add the handoff note (the WHY) — before the publishing PATCH;
3. update the lead's custom fields, and move its stage IF it is new OR a reused
   lead still sitting in "Incoming leads" (created-but-never-published, e.g. a
   prior attempt that failed after creating); a reused lead the advisor already
   moved is left in place;
4. return the lead id.

Any CRM error propagates (never swallowed) so the caller skips the flag/phase flip
and the handoff retries on the next turn. A retry re-finds a lead created in a
failed attempt (the contact now exists) and reuses it instead of duplicating.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from app.crm.kommo_crm import KommoContactMatch
from app.crm.lead_payload import build_custom_fields_values
from app.domain.concepts import SLOT_CONCEPTS, Concept
from app.domain.handoff_note import compose_handoff_note
from app.domain.models import HandoffReason
from app.understanding.schemas import escape_slot_names

_WHATSAPP_PREFIX = "whatsapp:"


class CrmClient(Protocol):
    """The CRM primitives the orchestration uses (KommoCrmClient satisfies it)."""

    async def find_contact_by_phone(self, phone: str) -> list[KommoContactMatch]: ...

    async def create_lead_with_contact(
        self, lead_name: str, contact_name: str, phone: str
    ) -> int: ...

    async def get_lead(self, lead_id: int) -> dict[str, object]: ...

    async def add_note(self, lead_id: int, text: str) -> object: ...

    async def update_lead(
        self,
        lead_id: int,
        *,
        custom_fields_values: list[dict[str, object]] | None = None,
        status_id: int | None = None,
        pipeline_id: int | None = None,
    ) -> object | None: ...


@dataclass(frozen=True, slots=True)
class HandoffMapping:
    """Per-client funnel mapping injected into the orchestration (account IDs)."""

    concept_field_ids: Mapping[Concept, int]
    reason_status_ids: Mapping[HandoffReason, int]
    pipeline_id: int
    incoming_status_id: int


def phone_from_sender(sender: str) -> str:
    """Extract the E.164 phone from a channel sender (strip a ``whatsapp:`` prefix)."""
    if sender.startswith(_WHATSAPP_PREFIX):
        return sender[len(_WHATSAPP_PREFIX) :]
    return sender


class HandoffRunner:
    """Runs the CRM handoff sequence with an injected client + per-client mapping."""

    def __init__(self, client: CrmClient, mapping: HandoffMapping) -> None:
        self._client = client
        self._mapping = mapping

    async def run(
        self,
        *,
        reason: HandoffReason,
        phone: str,
        customer_name: str,
        slots: dict[str, object],
        pending: Sequence[str],
    ) -> int:
        """Execute the handoff against the CRM; return the lead id (raises on error)."""
        lead_id, is_new = await self._resolve_lead(phone, customer_name)
        await self._client.add_note(lead_id, compose_handoff_note(reason, pending))
        custom_fields_values = build_custom_fields_values(
            slots,
            slot_concepts=SLOT_CONCEPTS,
            concept_field_ids=self._mapping.concept_field_ids,
            escape_slots=escape_slot_names(),
        )
        if await self._should_move_stage(is_new, lead_id):
            await self._client.update_lead(
                lead_id,
                custom_fields_values=custom_fields_values,
                status_id=self._mapping.reason_status_ids[reason],
                pipeline_id=self._mapping.pipeline_id,
            )
        else:
            await self._client.update_lead(
                lead_id, custom_fields_values=custom_fields_values
            )
        return lead_id

    async def _resolve_lead(self, phone: str, customer_name: str) -> tuple[int, bool]:
        """Reuse the most recent existing lead, else create one. ``True`` if new."""
        matches = await self._client.find_contact_by_phone(phone)
        lead_ids = [lead_id for match in matches for lead_id in match.lead_ids]
        if lead_ids:
            return max(lead_ids), False  # most recent = highest id
        new_id = await self._client.create_lead_with_contact(
            customer_name, customer_name, phone
        )
        return new_id, True

    async def _should_move_stage(self, is_new: bool, lead_id: int) -> bool:
        """Move a new lead, or a reused one still unpublished in "Incoming leads"."""
        if is_new:
            return True
        lead = await self._client.get_lead(lead_id)
        return lead.get("status_id") == self._mapping.incoming_status_id
