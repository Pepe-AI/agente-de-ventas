"""Per-sender message buffer.

Every accepted message is appended; nothing is dropped. The winning debounce
flush drains the whole buffer at once (under the conversation lock, so the
non-atomic LRANGE+DEL is safe).
"""

from __future__ import annotations

from redis.asyncio import Redis

from app.concurrency.keys import KeyPrefix, make_key


async def append(redis: Redis, sender: str, text: str) -> int:
    """Append a message to the sender's buffer; return the new buffer length."""
    return await redis.rpush(make_key(KeyPrefix.BUFFER, sender), text)


async def drain(redis: Redis, sender: str) -> list[str]:
    """Read and clear the sender's whole buffer, preserving arrival order."""
    key = make_key(KeyPrefix.BUFFER, sender)
    items = await redis.lrange(key, 0, -1)
    await redis.delete(key)
    return [str(item) for item in items]
