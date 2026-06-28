"""Headless inactivity handoff: transfer a silent lead to "No respondió" (the timer).

Runs OUTSIDE a customer turn (no ``IncomingMessage``, no reply to the customer): the
sweeper fires it once a conversation's ``inactivity_deadline`` lapses. It reuses the
core handoff sequence — the CRM ``HandoffRunner`` + the shared chat-connect helper +
the phase/flag flip — but skips the farewell (the customer is gone) and the slot
logic (nothing changed this turn). The chat IS connected so the advisor can reactivate
the lead if the customer returns. Any CRM/chat error PROPAGATES (the sweeper logs it
per conversation and moves on).
"""

from __future__ import annotations

import structlog
from redis.asyncio import Redis

from app.concurrency.handoff import set_handoff
from app.domain.chat_connection import ChatConnector
from app.domain.handoff_orchestration import HandoffRunner, phone_from_sender
from app.domain.models import HandoffReason
from app.domain.orchestrator import connect_chat_at_handoff
from app.domain.state import ConversationState, Phase, StateStore

log = structlog.get_logger()


async def run_inactivity_handoff(
    sender: str,
    state: ConversationState,
    *,
    redis: Redis,
    store: StateStore,
    handoff_runner: HandoffRunner,
    chat_connector: ChatConnector | None,
) -> None:
    """Transfer a silent conversation to "No respondió" (no msg, no reply).

    Derives phone/customer_name from the loaded ``state`` exactly like ``_handoff``,
    runs the CRM sequence with ``reason=NO_RESPONSE`` (``is_new=True`` for a brand-new
    lead → it lands in the "No respondió" stage), connects the chat, then flips the
    phase, clears the deadline, and sets the handoff flag (the point of no return).
    """
    phone = phone_from_sender(sender)
    name = state.slots.get("nombre_cliente")
    customer_name = name if isinstance(name, str) and name.strip() else phone

    # Same idempotency marker as ``_handoff``: run the CRM sequence ONCE; a swept
    # retry reuses the persisted lead/contact and skips run (no duplicate note).
    contact_id = state.contact_id
    if contact_id is None:
        result = await handoff_runner.run(
            reason=HandoffReason.NO_RESPONSE,
            phone=phone,
            customer_name=customer_name,
            slots=state.slots,
            pending=(),
        )
        state.lead_id = result.lead_id
        state.contact_id = contact_id = result.contact_id
        await store.save(sender, state)  # persist the marker BEFORE the link
    await connect_chat_at_handoff(
        state, sender, phone, customer_name, contact_id,
        chat_connector=chat_connector, store=store,
    )
    state.phase = Phase.HANDED_OFF
    state.last_asked = None
    state.inactivity_deadline = None
    await store.save(sender, state)
    await set_handoff(redis, sender)
    log.info("inactivity_handoff_done", sender=sender)
