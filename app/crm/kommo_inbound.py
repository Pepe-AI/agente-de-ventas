"""Durable enqueue + background relay scheduling for inbound Kommo webhooks.

Reuses the inc-2 concurrency pattern (fast-ack -> background): durably store the
verified payload in Redis, then process out of band. The processing — relay the
agent's message to the end user via Twilio/WhatsApp — is a STUB this increment;
only the integration point is wired.
"""

from __future__ import annotations

import asyncio

import structlog
from redis.asyncio import Redis

from app.concurrency.keys import KeyPrefix, make_key

log = structlog.get_logger()

# Strong refs so fire-and-forget background tasks are not GC'd mid-flight.
_background_tasks: set[asyncio.Task[None]] = set()


async def enqueue_inbound(redis: Redis, scope_id: str, body: bytes) -> None:
    """Durably enqueue a verified inbound webhook, then schedule background relay.

    The raw payload (utf-8 JSON from Kommo) is pushed to Redis BEFORE we ack; a
    failure here propagates to the caller (-> HTTP 500). The relay itself runs in
    the background.
    """
    key = make_key(KeyPrefix.KOMMO_INBOUND, scope_id)
    await redis.rpush(key, body.decode("utf-8"))
    _schedule_relay(scope_id)


def _schedule_relay(scope_id: str) -> None:
    task = asyncio.create_task(_relay_inbound_stub(scope_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _relay_inbound_stub(scope_id: str) -> None:
    """STUB (next increment): drain the queue, parse the payload, and relay the
    agent's message to the end user via Twilio/WhatsApp. Integration point."""
    log.info("kommo_inbound_relay_stub", scope_id=scope_id)
