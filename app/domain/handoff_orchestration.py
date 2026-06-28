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
3. update the lead's custom fields ALWAYS, and move its stage only FORWARD: for a
   new lead, or a reused lead whose current stage is at or behind the reason's
   target stage in the funnel's sort order (so a prior failed attempt — which
   landed in the create stage — gets re-published). A reused lead the advisor
   already advanced past the target is left in place; a lead whose stage is off
   the sort map (an unknown stage, or one in another pipeline) is also left in
   place and logs a warning;
4. return the lead id.

Any CRM error propagates (never swallowed) so the caller skips the flag/phase flip
and the handoff retries on the next turn. A retry re-finds a lead created in a
failed attempt (the contact now exists) and reuses it instead of duplicating.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import structlog

from app.crm.kommo_crm import KommoContactMatch, KommoCreatedLead
from app.crm.lead_payload import build_custom_fields_values
from app.domain.concepts import SLOT_CONCEPTS, Concept
from app.domain.handoff_note import compose_handoff_note
from app.domain.models import HandoffReason
from app.understanding.schemas import escape_slot_names

log = structlog.get_logger()

_WHATSAPP_PREFIX = "whatsapp:"


class CrmClient(Protocol):
    """The CRM primitives the orchestration uses (KommoCrmClient satisfies it)."""

    async def find_contact_by_phone(self, phone: str) -> list[KommoContactMatch]: ...

    async def create_lead_with_contact(
        self, lead_name: str, contact_name: str, phone: str
    ) -> KommoCreatedLead: ...

    async def create_lead_for_contact(
        self, lead_name: str, contact_id: int
    ) -> KommoCreatedLead: ...

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
    status_sort: Mapping[int, int]


def phone_from_sender(sender: str) -> str:
    """Extract the E.164 phone from a channel sender (strip a ``whatsapp:`` prefix)."""
    if sender.startswith(_WHATSAPP_PREFIX):
        return sender[len(_WHATSAPP_PREFIX) :]
    return sender


@dataclass(frozen=True, slots=True)
class HandoffResult:
    """What a handoff produced: the lead it landed on, and that lead's contact.

    The contact id lets a later step link a Chats API chat to the existing contact
    (so Kommo does not auto-create a duplicate Incoming lead). The result will grow
    (e.g. a chat id) — a dataclass extends without breaking positional call sites.
    """

    lead_id: int
    contact_id: int


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
    ) -> HandoffResult:
        """Run the handoff against the CRM; return lead+contact (raises on error)."""
        lead_id, contact_id, is_new = await self._resolve_lead(phone, customer_name)
        await self._client.add_note(lead_id, compose_handoff_note(reason, pending))
        custom_fields_values = build_custom_fields_values(
            slots,
            slot_concepts=SLOT_CONCEPTS,
            concept_field_ids=self._mapping.concept_field_ids,
            escape_slots=escape_slot_names(),
        )
        if await self._should_move_stage(is_new, lead_id, reason):
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
        return HandoffResult(lead_id=lead_id, contact_id=contact_id)

    async def _resolve_lead(
        self, phone: str, customer_name: str
    ) -> tuple[int, int, bool]:
        """Reuse the most recent existing lead, reuse an existing contact, or create.

        Returns ``(lead_id, contact_id, is_new)``. Three cases, in order:
        1. a contact has leads -> REUSE the (lead, contact) pair with the highest lead
           id, so contact_id OWNS that lead (not just any contact sharing the phone);
        2. contacts exist but NONE has a lead -> create a lead ON the most recent
           contact (``create_lead_for_contact``), reusing it instead of duplicating it;
        3. no contact at all -> create a fresh lead + contact.
        On create both ids come from the same response.
        """
        matches = await self._client.find_contact_by_phone(phone)
        pairs = [
            (lead_id, match.contact_id)
            for match in matches
            for lead_id in match.lead_ids
        ]
        if pairs:
            lead_id, contact_id = max(pairs, key=lambda pair: pair[0])
            return lead_id, contact_id, False  # most recent = highest lead id
        if matches:
            # Contacts exist for this phone but NONE has a lead (an Incoming
            # auto-created contact, or one whose lead was won/lost/unlinked). Reuse
            # the most recent contact and create the lead ON it, instead of creating a
            # duplicate contact (Fix B prevention). max(lead_id) then stays stable.
            contact_id = max(match.contact_id for match in matches)
            created = await self._client.create_lead_for_contact(
                customer_name, contact_id
            )
            return created.lead_id, created.contact_id, True
        created = await self._client.create_lead_with_contact(
            customer_name, customer_name, phone
        )
        return created.lead_id, created.contact_id, True

    async def _should_move_stage(
        self, is_new: bool, lead_id: int, reason: HandoffReason
    ) -> bool:
        """Whether to move a reused lead, by the funnel's stage order (``sort``).

        A new lead always moves. A reused lead moves only FORWARD — when its current
        stage is at or behind the reason's target stage. If its stage is not in the
        sort map (an unknown stage, or a lead living in another pipeline), the
        advisor is assumed to own it: leave it in place and log a warning (for Top
        Viajes only the handoff pipeline is used, so this should not happen).
        """
        if is_new:
            return True
        lead = await self._client.get_lead(lead_id)
        status_id = lead.get("status_id")
        current_sort: int | None = None
        if isinstance(status_id, int):
            current_sort = self._mapping.status_sort.get(status_id)
        if current_sort is None:
            log.warning(
                "handoff_reused_lead_off_sort_map",
                lead_id=lead_id,
                status_id=status_id,
                pipeline_id=lead.get("pipeline_id"),
            )
            return False
        target_sort = self._mapping.status_sort.get(
            self._mapping.reason_status_ids[reason]
        )
        if target_sort is None:
            return False  # incomplete mapping (a config bug) -> don't move backward
        return current_sort <= target_sort
