"""Per-sender message buffer.

Every accepted message is appended; nothing is dropped. The winning debounce
flush drains the whole buffer at once. The drain is atomic (LRANGE+DELETE in one
MULTI/EXEC transaction), so two concurrent flushes can never read the same items
before they are cleared -- one gets the items, the other gets an empty list.
"""

from __future__ import annotations

from redis.asyncio import Redis

from app.concurrency.keys import KeyPrefix, make_key


async def append(redis: Redis, sender: str, text: str) -> int:
    """Append a message to the sender's buffer; return the new buffer length."""
    return await redis.rpush(make_key(KeyPrefix.BUFFER, sender), text)


async def drain(redis: Redis, sender: str) -> list[str]:
    """Atomically read and clear the sender's whole buffer, preserving order.

    LRANGE + DELETE run in one MULTI/EXEC transaction (``redis.pipeline()`` is
    transactional by default), so the read and the clear are indivisible. No
    WATCH: there is no read-then-decide here, only an atomic drain.
    """
    key = make_key(KeyPrefix.BUFFER, sender)
    async with redis.pipeline() as pipe:
        pipe.lrange(key, 0, -1)
        pipe.delete(key)
        items, _ = await pipe.execute()
    return [str(item) for item in items]
