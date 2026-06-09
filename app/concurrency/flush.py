"""Background flush: the debounce winner drains the buffer and replies.

Scheduling is in-process (asyncio); all shared state lives in Redis, so the
debounce/lock logic stays correct even if several app workers run.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import structlog
from redis.asyncio import Redis

from app.channels.base import Channel
from app.concurrency import buffer, debounce, lock, rate_limit
from app.concurrency.config import ConcurrencyConfig
from app.domain.models import IncomingMessage
from app.domain.orchestrator import handle_message

log = structlog.get_logger()

BUFFER_SEPARATOR = "\n"

# Keep strong references so background tasks are not garbage-collected mid-flight.
_background_tasks: set[asyncio.Task[None]] = set()


def schedule_flush(
    redis: Redis,
    channel: Channel,
    sender: str,
    token: str,
    config: ConcurrencyConfig,
) -> None:
    """Schedule a flush for ``sender`` after the debounce window (non-blocking)."""
    task = asyncio.create_task(
        _flush_after_window(redis, channel, sender, token, config)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _flush_after_window(
    redis: Redis,
    channel: Channel,
    sender: str,
    token: str,
    config: ConcurrencyConfig,
) -> None:
    await asyncio.sleep(config.debounce_window_s)
    await flush(redis, channel, sender, token, config)


async def flush(
    redis: Redis,
    channel: Channel,
    sender: str,
    token: str,
    config: ConcurrencyConfig,
) -> None:
    """Process a sender's buffered messages once, if this flush is the winner."""
    if not await debounce.is_latest_token(redis, sender, token):
        return  # a newer message arrived; its flush will handle the buffer

    lock_token = uuid4().hex
    if not await lock.acquire(redis, sender, lock_token, config.lock_ttl_s):
        return  # another flush is already processing this sender

    try:
        parts = await buffer.drain(redis, sender)
        if not parts:
            return

        # A sender blocked between buffering and flush must not get a reply.
        if await rate_limit.is_blocked(redis, sender):
            log.info("flush_aborted_blocked", sender=sender)
            return

        combined = BUFFER_SEPARATOR.join(parts)
        reply = handle_message(
            IncomingMessage(sender=sender, text=combined, message_id=token)
        )
        await channel.send(sender, reply)
        log.info("flush_sent", sender=sender, parts=len(parts))
    finally:
        await lock.release(redis, sender, lock_token)
