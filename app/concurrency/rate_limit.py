"""Flood protection: fixed-window rate limiting and blocking per sender."""

from __future__ import annotations

from redis.asyncio import Redis

from app.concurrency.keys import KeyPrefix, make_key

_BLOCKED = "1"


async def register_hit(redis: Redis, sender: str, window_s: int) -> int:
    """Increment the sender's hit counter, setting the window TTL on first hit.

    Returns the running count within the current fixed window.
    """
    key = make_key(KeyPrefix.RATE, sender)
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, window_s)
    return count


async def block(redis: Redis, sender: str, cooldown_s: int) -> None:
    """Block a sender for ``cooldown_s`` seconds."""
    await redis.set(make_key(KeyPrefix.BLOCKED, sender), _BLOCKED, ex=cooldown_s)


async def is_blocked(redis: Redis, sender: str) -> bool:
    """Return ``True`` while the sender is within an active block cooldown."""
    return bool(await redis.exists(make_key(KeyPrefix.BLOCKED, sender)))
