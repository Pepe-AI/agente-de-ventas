"""Durable enqueue + background drain of inbound Kommo webhooks (advisor -> client).

Reuses the inc-2 fast-ack -> background pattern: the webhook verifies + durably
rpush'es the raw v1 payload to Redis, then returns; the drain runs out of band,
pops each queued advisor message (FIFO) and relays its text to the customer via
Twilio. This is the reverse leg of Direction B — the advisor writing inside Kommo
reaches the customer's WhatsApp.

NO idempotency key: the v1 webhook carries no per-message id, so a (rare) Kommo
retry would resend. Kommo delivers each webhook once without retries (see the
receiver), and a content hash would wrongly drop legit repeats ("ok"/"👍"), so the
duplicate risk is ACCEPTED for the MVP. Delivery-status is NOT reported (v1 has no
msgid to report against; it is cosmetic — the message still arrives).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import cast

import structlog
from redis.asyncio import Redis

from app.channels.base import Channel
from app.concurrency.keys import KeyPrefix, make_key

log = structlog.get_logger()

_WHATSAPP_PREFIX = "whatsapp:"
_TEXT_TYPE = "text"

# Strong refs so fire-and-forget background tasks are not GC'd mid-flight.
_background_tasks: set[asyncio.Task[None]] = set()


@dataclass(frozen=True, slots=True)
class InboundAdvisorMessage:
    """A parsed v1 inbound chat webhook (advisor reply) — only what the drain needs.

    ``conversation_id`` is the customer phone in E.164 (B1's routing key), which maps
    straight to the Twilio destination. ``msg_type`` is the payload ``type`` (the
    drain relays only ``"text"``).
    """

    conversation_id: str
    text: str
    msg_type: str


async def enqueue_inbound(redis: Redis, scope_id: str, body: bytes) -> None:
    """Durably enqueue a verified inbound webhook, then schedule the background drain.

    The raw payload (utf-8 JSON from Kommo) is pushed to Redis BEFORE we ack; a
    failure here propagates to the caller (-> HTTP 500). The drain runs in the
    background and pops whatever it finds.
    """
    key = make_key(KeyPrefix.KOMMO_INBOUND, scope_id)
    await redis.rpush(key, body.decode("utf-8"))
    _schedule_relay(redis, scope_id)


def _schedule_relay(redis: Redis, scope_id: str) -> None:
    # Deferred import: app.main imports this module, so a top-level `import app.main`
    # would be a cycle. At call time (request handling) app.main is fully loaded, and
    # get_channel is an lru_cache singleton, so this is cheap.
    from app.main import get_channel

    task = asyncio.create_task(_drain(redis, get_channel(), scope_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def _parse_v1(raw: str) -> InboundAdvisorMessage | None:
    """Parse a v1 inbound payload; return ``None`` for STRUCTURALLY unusable input.

    Never raises. Rejects (``None`` + warning) only what cannot be relayed at all:
    bad JSON, a non-object, or a missing/blank ``conversation_id`` (no destination).
    Policy skips (non-text, empty text) are the drain's job, not the parser's, so a
    legit picture (``type`` != text, blank ``text``) is parsed fine and skipped BY
    TYPE downstream.
    """
    try:
        data: object = json.loads(raw)
    except ValueError:
        log.warning("drain_parse_failed", reason="not_json")
        return None
    if not isinstance(data, dict):
        log.warning("drain_parse_failed", reason="not_object")
        return None
    fields = cast("dict[str, object]", data)
    conversation_id = fields.get("conversation_id")
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        log.warning("drain_parse_failed", reason="no_conversation_id")
        return None
    text = fields.get("text")
    msg_type = fields.get("type")
    return InboundAdvisorMessage(
        conversation_id=conversation_id,
        text=text if isinstance(text, str) else "",
        msg_type=msg_type if isinstance(msg_type, str) else "",
    )


async def _drain(redis: Redis, channel: Channel, scope_id: str) -> None:
    """Pop and relay EVERY queued advisor message to the customer via Twilio.

    Loops lpop until the queue is empty so a message can never be orphaned by a
    webhook<->relay imbalance; lpop is atomic, so concurrent drains don't
    double-process (one pops, the other sees empty — worst case is reordering, which
    we accept). Each send has its OWN try/except: one failing message (e.g. outside
    WhatsApp's 24h window) is logged and the loop CONTINUES with the rest. Nothing
    propagates to the background task.
    """
    key = make_key(KeyPrefix.KOMMO_INBOUND, scope_id)
    while True:
        raw = await redis.lpop(key)
        if not isinstance(raw, str):
            # None when the queue is empty; decode_responses=True guarantees str
            # for any actual element, so a non-str means there is nothing to relay.
            return
        msg = _parse_v1(raw)
        if msg is None:
            continue  # already logged drain_parse_failed
        if msg.msg_type != _TEXT_TYPE:
            log.warning("drain_skipped_non_text", type=msg.msg_type)
            continue
        if not msg.text.strip():
            log.warning("drain_skipped_empty_text")
            continue
        to = f"{_WHATSAPP_PREFIX}{msg.conversation_id}"
        try:
            await channel.send(to, msg.text)
            log.info("drain_sent", to=to)
        except Exception:
            # Per-message guard: a single failure must not stop draining the rest,
            # and must never surface as an orphaned background-task exception.
            log.exception("drain_failed", to=to)
