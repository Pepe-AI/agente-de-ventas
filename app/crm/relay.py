"""Relay seam: forward an inbound message to the human agent.

Stub for now. Increment 8 implements the real relay via Kommo's Chats API
without touching the endpoint.
"""

from __future__ import annotations

import structlog

from app.domain.models import IncomingMessage

log = structlog.get_logger()


async def relay_to_human(message: IncomingMessage) -> None:
    """Forward a message to the human handling this conversation (no-op stub)."""
    log.info(
        "would relay to human",
        sender=message.sender,
        message_id=message.message_id,
    )
