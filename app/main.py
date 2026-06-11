"""HTTP layer and composition root.

The endpoint stays thin: it deals with HTTP concerns and sequences the small
Redis-backed input checks, then fast-acks with 200. The heavy work (running the
orchestrator and sending the reply) happens in a background flush.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, Request, Response
from redis.asyncio import Redis, from_url
from twilio.request_validator import RequestValidator
from twilio.rest import Client

from app.channels.base import Channel
from app.channels.twilio import InvalidPayloadError, TwilioChannel
from app.concurrency import buffer, debounce, dedup, handoff, rate_limit
from app.concurrency.config import ConcurrencyConfig
from app.concurrency.flush import schedule_flush
from app.config import HttpHeader, get_settings
from app.crm.relay import relay_to_human
from app.domain.models import IncomingMessage
from app.llm.base import LLM
from app.llm.gemini import build_gemini_llm

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)

log = structlog.get_logger()

WEBHOOK_PATH = "/webhook/whatsapp"
EMPTY_TWIML = "<Response></Response>"
TWIML_MEDIA_TYPE = "application/xml"

app = FastAPI(title="WhatsApp Echo Agent")


@lru_cache
def get_channel() -> Channel:
    """Build the transport adapter once (composition root)."""
    settings = get_settings()
    token = settings.twilio_auth_token.get_secret_value()
    return TwilioChannel(
        validator=RequestValidator(token),
        client=Client(settings.twilio_account_sid, token),
        from_=settings.twilio_whatsapp_from,
    )


@lru_cache
def get_redis() -> Redis:
    """Build the async Redis client once (composition root)."""
    return from_url(get_settings().redis_url, decode_responses=True)


@lru_cache
def get_llm() -> LLM:
    """Build the Gemini LLM adapter once (composition root)."""
    settings = get_settings()
    return build_gemini_llm(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.llm_model,
    )


def get_concurrency_config() -> ConcurrencyConfig:
    """Expose the concurrency tunables derived from settings."""
    return ConcurrencyConfig.from_settings(get_settings())


def build_public_url(request: Request) -> str:
    """Reconstruct the public URL Twilio used to sign the request.

    Behind a tunnel/proxy ``request.url`` is the internal address, which would
    make signature validation fail. We rebuild it from the forwarded headers,
    falling back to ``request.url`` for local development.
    """
    proto = request.headers.get(HttpHeader.X_FORWARDED_PROTO)
    host = request.headers.get(HttpHeader.HOST)
    if proto and host:
        url = f"{proto}://{host}{request.url.path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        return url
    return str(request.url)


def _ack() -> Response:
    """Fast 200 with empty TwiML so Twilio does not flag the webhook."""
    return Response(content=EMPTY_TWIML, media_type=TWIML_MEDIA_TYPE)


@app.post(WEBHOOK_PATH)
async def whatsapp_webhook(
    request: Request,
    channel: Annotated[Channel, Depends(get_channel)],
    redis: Annotated[Redis, Depends(get_redis)],
    llm: Annotated[LLM, Depends(get_llm)],
    config: Annotated[ConcurrencyConfig, Depends(get_concurrency_config)],
) -> Response:
    """Receive a WhatsApp message: validate, run input checks, fast-ack."""
    # Twilio posts urlencoded fields; keep only str values (drop any uploads).
    form = {k: v for k, v in (await request.form()).items() if isinstance(v, str)}
    signature = request.headers.get(HttpHeader.X_TWILIO_SIGNATURE, "")
    url = build_public_url(request)

    if not channel.verify_signature(url, form, signature):
        log.warning("invalid_signature", url=url)
        return Response(status_code=403)

    try:
        msg: IncomingMessage = channel.parse_incoming(form)
    except InvalidPayloadError as exc:
        log.warning("malformed_payload", error=str(exc))
        return Response(status_code=400)

    sender = msg.sender

    # 1. Blocked senders are dropped silently (no buffering, no reply).
    if await rate_limit.is_blocked(redis, sender):
        log.info("blocked_discard", sender=sender)
        return _ack()

    # 2. Idempotency: a repeated MessageSid is a retry/duplicate.
    if await dedup.is_duplicate(redis, msg.message_id, config.dedup_ttl_s):
        log.info("duplicate_discard", message_id=msg.message_id)
        return _ack()

    # 2b. Human handoff: bot stays silent; relay to the human and ack only.
    if await handoff.is_handed_off(redis, sender):
        await relay_to_human(msg)
        log.info("handoff_relay", sender=sender)
        return _ack()

    # 3. Flood: rate-limit and block on threshold.
    hits = await rate_limit.register_hit(redis, sender, config.rate_window_s)
    if hits > config.rate_threshold:
        await rate_limit.block(redis, sender, config.block_cooldown_s)
        log.warning("flood_blocked", sender=sender, hits=hits)
        return _ack()

    # 4. Buffer the message; an overlong buffer is treated as flood.
    size = await buffer.append(redis, sender, msg.text)
    if size > config.buffer_max:
        await rate_limit.block(redis, sender, config.block_cooldown_s)
        log.warning("buffer_overflow_blocked", sender=sender, size=size)
        return _ack()

    # 5. Register debounce token and schedule the background flush.
    await debounce.set_token(redis, sender, msg.message_id)
    schedule_flush(redis, channel, llm, sender, msg.message_id, config)

    log.info("incoming_buffered", sender=sender, message_id=msg.message_id)
    return _ack()
