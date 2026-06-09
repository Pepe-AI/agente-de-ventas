"""Per-conversation lock.

A flush must hold the sender's lock before processing, so two concurrent
flushes never process the same conversation. Release is owner-safe via an
optimistic WATCH/MULTI transaction: we only delete the key if we still hold it.
"""

from __future__ import annotations

from redis.asyncio import Redis
from redis.exceptions import WatchError

from app.concurrency.keys import KeyPrefix, make_key


async def acquire(redis: Redis, sender: str, token: str, ttl_s: int) -> bool:
    """Try to take the sender's lock. Returns ``True`` if acquired."""
    key = make_key(KeyPrefix.LOCK, sender)
    return bool(await redis.set(key, token, nx=True, ex=ttl_s))


async def release(redis: Redis, sender: str, token: str) -> None:
    """Release the lock only if we are still its owner (compare-and-delete)."""
    key = make_key(KeyPrefix.LOCK, sender)
    async with redis.pipeline() as pipe:
        while True:
            try:
                await pipe.watch(key)
                if await pipe.get(key) == token:
                    pipe.multi()
                    pipe.delete(key)
                    await pipe.execute()
                else:
                    await pipe.unwatch()
                return
            except WatchError:
                # The key changed under us; re-check ownership.
                continue
