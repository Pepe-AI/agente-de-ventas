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
from app.concurrency import buffer, lock, rate_limit
from app.concurrency.config import ConcurrencyConfig
from app.domain.chat_connection import ChatConnector
from app.domain.handoff_orchestration import HandoffRunner
from app.domain.models import IncomingMessage
from app.domain.orchestrator import handle_message
from app.domain.state import StateStore
from app.llm.base import LLM, LLMUnavailableError
from app.routing.campaign import RoutingConfig

log = structlog.get_logger()

BUFFER_SEPARATOR = "\n"

# Graceful replies when a turn cannot be processed (the bot must never go silent).
_LLM_FALLBACK = (
    "Estoy teniendo un problema momentáneo para procesar tu mensaje. "
    "¿Me lo reenvías en un momento, por favor? 🙏"
)
_GENERIC_APOLOGY = (
    "Disculpa, tuve un inconveniente para responderte. "
    "¿Me reenvías tu último mensaje, por favor?"
)

# Keep strong references so background tasks are not garbage-collected mid-flight.
_background_tasks: set[asyncio.Task[None]] = set()
# Real per-sender debounce, in memory only (NOT persisted in Redis): the single
# pending flush timer per sender, and the burst's cap anchor (the loop time of the
# burst's first message). Losing these on a redeploy is benign -- a burst is then
# flushed a little early, never duplicated nor dropped.
_pending_flushes: dict[str, asyncio.Task[None]] = {}
_burst_anchors: dict[str, float] = {}


def _now() -> float:
    """Monotonic clock (event-loop time); a seam so tests can drive the timer."""
    return asyncio.get_running_loop().time()


async def _sleep(delay: float) -> None:
    """Indirection over ``asyncio.sleep`` so tests can drive the debounce."""
    await asyncio.sleep(delay)


def schedule_flush(
    redis: Redis,
    channel: Channel,
    llm: LLM,
    store: StateStore,
    routing: RoutingConfig,
    corpus: str,
    handoff_runner: HandoffRunner,
    chat_connector: ChatConnector | None,
    sender: str,
    token: str,
    config: ConcurrencyConfig,
) -> None:
    """(Re)arm the sender's single debounce timer (non-blocking).

    A real debounce: each message cancels the pending timer and reschedules it to
    fire at ``min(now + debounce_window_s, anchor + max_buffer_wait_s)`` -- the
    debounce, capped so a nonstop typer is still flushed at ``anchor + cap``. The
    cap anchor (first message of the burst) is set on the first message and kept
    until the flush fires.
    """
    now = _now()
    anchor = _burst_anchors.setdefault(sender, now)
    cap_remaining = anchor + config.max_buffer_wait_s - now
    delay = max(0.0, min(config.debounce_window_s, cap_remaining))

    existing = _pending_flushes.get(sender)
    if existing is not None:
        existing.cancel()  # supersede the pending timer; it aborts in _sleep

    task = asyncio.create_task(
        _debounced_flush(
            redis, channel, llm, store, routing, corpus, handoff_runner,
            chat_connector, sender, token, config, delay,
        )
    )
    _pending_flushes[sender] = task
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _debounced_flush(
    redis: Redis,
    channel: Channel,
    llm: LLM,
    store: StateStore,
    routing: RoutingConfig,
    corpus: str,
    handoff_runner: HandoffRunner,
    chat_connector: ChatConnector | None,
    sender: str,
    token: str,
    config: ConcurrencyConfig,
    delay: float,
) -> None:
    try:
        await _sleep(delay)
    except asyncio.CancelledError:
        raise  # a newer message rescheduled us; that timer now owns the burst
    # Committed to firing: drop the burst tracking BEFORE any further await, so a
    # message arriving during processing starts a fresh burst and cannot cancel us
    # (CancelledError can only ever land in the _sleep above, never inside flush).
    _burst_anchors.pop(sender, None)
    if _pending_flushes.get(sender) is asyncio.current_task():
        del _pending_flushes[sender]
    await flush(
        redis, channel, llm, store, routing, corpus, handoff_runner,
        chat_connector, sender, token, config,
    )


async def flush(
    redis: Redis,
    channel: Channel,
    llm: LLM,
    store: StateStore,
    routing: RoutingConfig,
    corpus: str,
    handoff_runner: HandoffRunner,
    chat_connector: ChatConnector | None,
    sender: str,
    token: str,
    config: ConcurrencyConfig,
) -> None:
    """Process a sender's buffered messages once, under the conversation lock."""
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
        try:
            reply = await handle_message(
                IncomingMessage(sender=sender, text=combined, message_id=token),
                llm,
                redis,
                store,
                routing,
                corpus,
                handoff_runner,
                chat_connector,
                config.inactivity_deadline_s,
            )
        except LLMUnavailableError:
            # Transient LLM outage after retries. handle_message only persists on
            # success, so state is untouched — ask the user to resend. Handled.
            log.warning("flush_llm_unavailable", sender=sender)
            await _safe_send(channel, sender, _LLM_FALLBACK)
            return
        except Exception:
            # Unexpected bug: apologize and never let the background task crash.
            log.exception("flush_handle_message_failed", sender=sender)
            await _safe_send(channel, sender, _GENERIC_APOLOGY)
            return

        if reply is None:
            # Already handed off: the orchestrator stays silent.
            log.info("flush_silent", sender=sender)
            return
        await _safe_send(channel, sender, reply)
        log.info("flush_sent", sender=sender, parts=len(parts))
    finally:
        await lock.release(redis, sender, lock_token)


async def _safe_send(channel: Channel, sender: str, text: str) -> None:
    """Send ``text``, swallowing transport errors so a flush never crashes."""
    try:
        await channel.send(sender, text)
    except Exception:
        log.exception("flush_send_failed", sender=sender)
