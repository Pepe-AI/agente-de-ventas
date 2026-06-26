"""Inactivity sweeper: hand off conversations that went silent past their deadline.

A periodic task (started in the app lifespan) calls :func:`sweep_once` every
``SWEEP_INTERVAL_S``. Each tick takes a single-sweeper Redis lock (so a future
multi-instance / multi-worker deploy never double-sweeps — today there is one
instance), queries Postgres for due deadlines, and per conversation takes the
SAME per-sender lock the flush uses: if a flush holds it, the inbound message wins
(it will have re-pushed the deadline), so the sweeper backs off. A re-load under
the lock re-checks the deadline before firing (it may have just been re-pushed or
the conversation handed off). A failure on one conversation is logged and the
sweep continues — one stuck lead never blocks the rest.
"""

from __future__ import annotations

from uuid import uuid4

import structlog
from redis.asyncio import Redis

from app.concurrency import lock
from app.domain.chat_connection import ChatConnector
from app.domain.handoff_orchestration import HandoffRunner
from app.domain.inactivity import run_inactivity_handoff
from app.domain.state import Phase, StateStore

log = structlog.get_logger()

SWEEP_INTERVAL_S = 300  # 5 minutes between sweeps
# Single-sweeper lock: a fixed identifier (no real sender equals it), so only one
# sweeper runs per tick. TTL > a sweep, < the interval, so a crashed sweeper frees it.
_SWEEP_LOCK_ID = "inactivity_sweep"
_SWEEP_LOCK_TTL_S = 240
_SENDER_LOCK_TTL_S = 30  # matches the flush lock so the two contend correctly


async def sweep_once(
    now: float,
    *,
    redis: Redis,
    store: StateStore,
    handoff_runner: HandoffRunner,
    chat_connector: ChatConnector | None,
) -> None:
    """Run one sweep: under the single-sweeper lock, hand off every due deadline."""
    token = uuid4().hex
    if not await lock.acquire(redis, _SWEEP_LOCK_ID, token, _SWEEP_LOCK_TTL_S):
        return  # another sweeper owns this tick
    try:
        for sender, _state in await store.find_expired_deadlines(now):
            await _sweep_one(
                sender, now, redis=redis, store=store,
                handoff_runner=handoff_runner, chat_connector=chat_connector,
            )
    finally:
        await lock.release(redis, _SWEEP_LOCK_ID, token)


async def _sweep_one(
    sender: str,
    now: float,
    *,
    redis: Redis,
    store: StateStore,
    handoff_runner: HandoffRunner,
    chat_connector: ChatConnector | None,
) -> None:
    """Hand off ONE conversation, under its per-sender lock, with a fresh re-check."""
    token = uuid4().hex
    if not await lock.acquire(redis, sender, token, _SENDER_LOCK_TTL_S):
        return  # a flush holds it: the inbound message wins, the deadline is re-pushed
    try:
        # Re-load under the lock: a flush between the query and here may have
        # re-pushed the deadline or already handed off.
        state = await store.load(sender)
        if state is None or state.phase is not Phase.COLLECTING:
            return
        if state.inactivity_deadline is None or state.inactivity_deadline > now:
            return
        await run_inactivity_handoff(
            sender, state, redis=redis, store=store,
            handoff_runner=handoff_runner, chat_connector=chat_connector,
        )
    except Exception:
        # One conversation's failure (CRM/chat error) must not abort the sweep.
        log.exception("inactivity_handoff_failed", sender=sender)
    finally:
        await lock.release(redis, sender, token)
