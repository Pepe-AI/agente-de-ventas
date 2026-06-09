"""Idempotency via Twilio's MessageSid.

A retried/duplicated webhook carries the same MessageSid; we record each sid
once with ``SET NX`` and a TTL so reprocessing is cheap to detect.
"""

from __future__ import annotations

from redis.asyncio import Redis

from app.concurrency.keys import KeyPrefix, make_key

_SEEN = "1"


async def is_duplicate(redis: Redis, message_id: str, ttl_s: int) -> bool:
    """Return ``True`` if ``message_id`` was already seen within the TTL window."""
    key = make_key(KeyPrefix.DEDUP, message_id)
    created = await redis.set(key, _SEEN, nx=True, ex=ttl_s)
    return not created
