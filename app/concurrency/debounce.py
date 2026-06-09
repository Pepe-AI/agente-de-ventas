"""Debounce token bookkeeping.

Each incoming message registers itself as the sender's most recent token (its
MessageSid). A scheduled flush only proceeds if it is still the latest token;
otherwise a newer message arrived and that one's flush will handle the buffer.
"""

from __future__ import annotations

from redis.asyncio import Redis

from app.concurrency.keys import KeyPrefix, make_key


async def set_token(redis: Redis, sender: str, token: str) -> None:
    """Record ``token`` as the sender's most recent debounce token."""
    await redis.set(make_key(KeyPrefix.DEBOUNCE, sender), token)


async def is_latest_token(redis: Redis, sender: str, token: str) -> bool:
    """Return ``True`` if ``token`` is still the sender's most recent token."""
    current = await redis.get(make_key(KeyPrefix.DEBOUNCE, sender))
    return current == token
