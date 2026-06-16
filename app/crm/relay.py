"""Relay seam: forward an inbound message to the human agent.

Stub for now. Increment 8 implements the real relay via Kommo's Chats API
without touching the endpoint.
"""

from __future__ import annotations

import structlog

from app.domain.models import HandoffEvent, IncomingMessage

log = structlog.get_logger()


async def relay_to_human(
    message: IncomingMessage, event: HandoffEvent | None = None
) -> None:
    """Forward a message to the human handling this conversation (no-op stub).

    When ``event`` is given (a fresh handoff), its reason/trip/slots ride along;
    increment 8 turns this into a real CRM relay + funnel mapping.
    """
    fields: dict[str, object] = {
        "sender": message.sender,
        "message_id": message.message_id,
    }
    if event is not None:
        fields["reason"] = event.reason.value
        fields["trip_type"] = event.trip_type
        fields["slots"] = event.slots
    log.info("would relay to human", **fields)
