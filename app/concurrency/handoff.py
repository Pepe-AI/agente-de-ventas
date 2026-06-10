"""Per-conversation human-handoff flag.

When set, the bot stays silent for that sender. The flag is *sticky* (no TTL):
it persists until explicitly cleared. Nothing here decides when to set or clear
it — that is the orchestrator's job in a later increment.
"""

from __future__ import annotations

from redis.asyncio import Redis

from app.concurrency.keys import KeyPrefix, make_key

_HANDED_OFF = "1"


async def set_handoff(redis: Redis, sender: str) -> None:
    """Hand the conversation to a human (sticky; no expiry)."""
    await redis.set(make_key(KeyPrefix.HANDOFF, sender), _HANDED_OFF)


async def is_handed_off(redis: Redis, sender: str) -> bool:
    """Return ``True`` while the conversation is in a human's hands."""
    return bool(await redis.exists(make_key(KeyPrefix.HANDOFF, sender)))


async def clear_handoff(redis: Redis, sender: str) -> None:
    """Return the conversation to the bot."""
    await redis.delete(make_key(KeyPrefix.HANDOFF, sender))
