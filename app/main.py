"""HTTP layer and composition root.

The endpoint stays thin: it deals with HTTP concerns and sequences the small
Redis-backed input checks, then fast-acks with 200. The heavy work (running the
orchestrator and sending the reply) happens in a background flush.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

import asyncpg
import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from redis.asyncio import Redis, from_url
from twilio.request_validator import RequestValidator
from twilio.rest import Client

from app.answering.corpus import load_corpus
from app.channels.base import Channel
from app.channels.twilio import InvalidPayloadError, TwilioChannel
from app.concurrency import buffer, debounce, dedup, handoff, rate_limit
from app.concurrency.config import ConcurrencyConfig
from app.concurrency.flush import schedule_flush
from app.config import HttpHeader, get_settings
from app.crm.kommo_inbound import enqueue_inbound
from app.crm.kommo_signing import KommoHeader, KommoSigner
from app.crm.relay import relay_to_human
from app.domain.models import IncomingMessage
from app.domain.state import StateStore
from app.llm.base import LLM
from app.llm.gemini import build_gemini_llm, is_transient_gemini_error
from app.llm.retry import RetryingLLM
from app.routing.campaign import RoutingConfig
from app.storage.postgres import PostgresStateStore, create_pool

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)

log = structlog.get_logger()

WEBHOOK_PATH = "/webhook/whatsapp"
KOMMO_WEBHOOK_PATH = "/kommo/chats/webhook/{scope_id}"
EMPTY_TWIML = "<Response></Response>"
TWIML_MEDIA_TYPE = "application/xml"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Validate boot-critical config, create the Postgres pool, close at shutdown.

    Fail-fast: the web app cannot serve the Kommo webhook without the channel
    secret, so it refuses to boot if it is unset (checked BEFORE the pool so it
    fails without a DB). This lives ONLY here — the migration runner never needs
    the secret and must keep running without it. The request path still returns
    503 (defense in depth). Migrations are NOT run here (they race across
    instances) — see ``app.storage.migrate``.
    """
    settings = get_settings()
    if settings.kommo_channel_secret is None:
        raise RuntimeError("KOMMO_CHANNEL_SECRET must be set to run the web app")
    pool = await create_pool(settings.database_url)
    app.state.pool = pool
    try:
        yield
    finally:
        await pool.close()


app = FastAPI(title="WhatsApp Echo Agent", lifespan=lifespan)


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
    """Build the Gemini LLM, wrapped in transient-failure retries (root).

    Wrapping at the port covers both LLM calls (understand_turn + the answerer)
    without touching their call sites.
    """
    settings = get_settings()
    gemini = build_gemini_llm(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.llm_model,
    )
    return RetryingLLM(gemini, is_transient_gemini_error)


def get_concurrency_config() -> ConcurrencyConfig:
    """Expose the concurrency tunables derived from settings."""
    return ConcurrencyConfig.from_settings(get_settings())


def get_routing_config() -> RoutingConfig:
    """Expose the campaign pre-fill phrases for trip-type routing."""
    return RoutingConfig.from_settings(get_settings())


def get_pool(request: Request) -> asyncpg.Pool:
    """Return the Postgres pool created in the lifespan."""
    return request.app.state.pool


def get_store(
    pool: Annotated[asyncpg.Pool, Depends(get_pool)],
) -> StateStore:
    """Build the durable state store over the pool (composition root)."""
    return PostgresStateStore(pool)


@lru_cache
def get_corpus() -> str:
    """Load the knowledge corpus once (composition root)."""
    return load_corpus(get_settings().corpus_path)


def get_kommo_signer() -> KommoSigner:
    """Build the Kommo signer from the channel secret (composition root).

    The secret is optional in Settings (to not couple the migration runner); a
    missing secret fails clearly here, only on the Kommo webhook path.
    """
    secret = get_settings().kommo_channel_secret
    if secret is None:
        raise HTTPException(
            status_code=503, detail="Kommo channel secret not configured"
        )
    return KommoSigner(secret.get_secret_value())


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
    store: Annotated[StateStore, Depends(get_store)],
    routing: Annotated[RoutingConfig, Depends(get_routing_config)],
    corpus: Annotated[str, Depends(get_corpus)],
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
    schedule_flush(
        redis, channel, llm, store, routing, corpus, sender, msg.message_id, config
    )

    log.info("incoming_buffered", sender=sender, message_id=msg.message_id)
    return _ack()


@app.post(KOMMO_WEBHOOK_PATH)
async def kommo_chats_webhook(
    scope_id: str,
    request: Request,
    signer: Annotated[KommoSigner, Depends(get_kommo_signer)],
    redis: Annotated[Redis, Depends(get_redis)],
) -> Response:
    """Inbound Kommo Chats webhook: verify (raw bytes) -> fast-ack -> enqueue.

    Kommo sends each webhook once, without retries, on a tight deadline, so we
    verify and durably enqueue, then return — the relay runs in the background.
    """
    # Raw bytes: never declare a body model, which would consume/re-parse them.
    body = await request.body()
    signature = request.headers.get(KommoHeader.SIGNATURE, "")
    if not signature or not signer.verify(body, signature):
        log.warning("kommo_invalid_signature", scope_id=scope_id)
        return Response(status_code=401)

    try:
        await enqueue_inbound(redis, scope_id, body)
    except Exception:
        log.exception("kommo_enqueue_failed", scope_id=scope_id)
        return Response(status_code=500)

    log.info("kommo_inbound_enqueued", scope_id=scope_id)
    return Response(status_code=200)
