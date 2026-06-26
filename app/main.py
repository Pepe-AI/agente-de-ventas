"""HTTP layer and composition root.

The endpoint stays thin: it deals with HTTP concerns and sequences the small
Redis-backed input checks, then fast-acks with 200. The heavy work (running the
orchestrator and sending the reply) happens in a background flush.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
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
from app.concurrency.inactivity import SWEEP_INTERVAL_S, sweep_once
from app.concurrency.mirror import schedule_mirror
from app.config import HttpHeader, get_settings
from app.crm import kommo_mapping_topviajes as kommo_mapping
from app.crm.kommo_chats import KommoChatsClient, KommoChatsError
from app.crm.kommo_crm import KommoCrmClient
from app.crm.kommo_inbound import enqueue_inbound
from app.crm.kommo_signing import KommoHeader, KommoSigner
from app.domain.chat_connection import ChatConnector
from app.domain.chat_mirror import ChatMirror
from app.domain.handoff_orchestration import HandoffMapping, HandoffRunner
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

    Fail-fast: the web app needs the Kommo credentials (the Chats channel secret
    and the CRM long-lived token), so it refuses to boot if either is unset
    (checked BEFORE the pool, so it fails without a DB). This lives ONLY here —
    the migration runner never needs them and must keep running without them.
    Migrations are NOT run here (they race across instances) — see
    ``app.storage.migrate``.
    """
    settings = get_settings()
    if settings.kommo_channel_secret is None:
        raise RuntimeError("KOMMO_CHANNEL_SECRET must be set to run the web app")
    if settings.kommo_long_lived_token is None:
        raise RuntimeError("KOMMO_LONG_LIVED_TOKEN must be set to run the web app")
    if settings.kommo_crm_base_url is None:
        raise RuntimeError("KOMMO_CRM_BASE_URL must be set to run the web app")
    if settings.kommo_channel_id is None:
        raise RuntimeError("KOMMO_CHANNEL_ID must be set to run the web app")
    if settings.kommo_amojo_id is None:
        raise RuntimeError("KOMMO_AMOJO_ID must be set to run the web app")
    pool = await create_pool(settings.database_url)
    app.state.pool = pool
    # Resolve the Chats scope_id at boot (derived, never stored). A connect FAILURE
    # degrades: the app still serves (the CRM handoff works), only chat connection is
    # skipped until a restart re-resolves it. See _resolve_scope_id_at_boot.
    app.state.scope_id = await _resolve_scope_id_at_boot()
    # Periodic inactivity sweep (durable deadlines in Postgres survive restarts).
    sweeper = asyncio.create_task(_inactivity_loop(app))
    try:
        yield
    finally:
        sweeper.cancel()
        with suppress(asyncio.CancelledError):
            await sweeper
        await pool.close()


async def _inactivity_loop(app: FastAPI) -> None:
    """Sweep for lapsed inactivity deadlines every ``SWEEP_INTERVAL_S`` (lifespan task).

    Each tick rebuilds its dependencies from the composition root and never lets a
    tick failure kill the loop; cancellation (shutdown) propagates cleanly.
    """
    while True:
        try:
            await asyncio.sleep(SWEEP_INTERVAL_S)
            await sweep_once(
                time.time(),
                redis=get_redis(),
                store=PostgresStateStore(app.state.pool),
                handoff_runner=get_handoff_runner(),
                chat_connector=_build_chat_connector(app.state.scope_id),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("inactivity_loop_tick_failed")


async def _resolve_scope_id_at_boot() -> str | None:
    """Connect the Chats channel at boot → scope_id, or ``None`` to degrade.

    A down chat channel must NOT take down the web app: the bot keeps qualifying and
    the CRM handoff keeps writing lead+note+fields; only the chat connection is
    skipped. ``None`` here makes ``get_chat_connector`` return ``None`` and ``_handoff``
    skip (and warn). channel_id/amojo_id presence is guaranteed by the fail-fast above.
    """
    amojo_id = get_settings().kommo_amojo_id
    if amojo_id is None:
        return None
    try:
        scope_id = await get_kommo_chats_client().connect(amojo_id)
        log.info("kommo_channel_connected", scope_id=scope_id)
        return scope_id
    except KommoChatsError as exc:
        log.warning("kommo_channel_connect_failed", error=str(exc))
        return None


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


@lru_cache
def get_kommo_crm_client() -> KommoCrmClient:
    """Build the Kommo CRM API client once (Bearer auth, composition root).

    The token + base URL are optional in Settings (to not couple migrate.py), but
    the lifespan fail-fast guarantees both are set before the web app serves, so a
    missing value here is a misconfiguration, not a request-time condition.
    """
    settings = get_settings()
    token = settings.kommo_long_lived_token
    base_url = settings.kommo_crm_base_url
    if token is None or base_url is None:
        raise RuntimeError("KOMMO_LONG_LIVED_TOKEN and KOMMO_CRM_BASE_URL must be set")
    return KommoCrmClient(token.get_secret_value(), base_url)


@lru_cache
def get_kommo_chats_client() -> KommoChatsClient:
    """Build the Kommo Chats API client once (HMAC signing, composition root).

    The channel id is optional in Settings (to not couple migrate.py); the lifespan
    fail-fast guarantees it is set before the web app serves.
    """
    settings = get_settings()
    if settings.kommo_channel_id is None:
        raise RuntimeError("KOMMO_CHANNEL_ID must be set to run the web app")
    return KommoChatsClient(get_kommo_signer(), settings.kommo_channel_id)


@lru_cache
def get_handoff_runner() -> HandoffRunner:
    """Build the handoff runner: CRM client + the per-client funnel mapping.

    The per-client ``kommo_mapping_topviajes`` is imported ONLY here (composition
    root); the core orchestration receives the mapping injected, never importing it.
    """
    mapping = HandoffMapping(
        concept_field_ids=kommo_mapping.CONCEPT_FIELD_IDS,
        reason_status_ids=kommo_mapping.REASON_STATUS_IDS,
        pipeline_id=kommo_mapping.PIPELINE_ID,
        status_sort=kommo_mapping.STATUS_SORT,
    )
    return HandoffRunner(get_kommo_crm_client(), mapping)


def _build_chat_connector(scope_id: str | None) -> ChatConnector | None:
    """Build a chat connector for ``scope_id`` (``None`` = degraded channel).

    Shared by the request DI (``get_chat_connector``) and the inactivity loop, which
    has no ``Request`` but reads ``app.state.scope_id`` directly.
    """
    if scope_id is None:
        return None
    return ChatConnector(get_kommo_chats_client(), get_kommo_crm_client(), scope_id)


def get_chat_connector(request: Request) -> ChatConnector | None:
    """Build the chat connector from the boot-resolved scope_id (``None`` = degraded).

    ``None`` when the channel failed to connect at boot — ``_handoff`` then skips chat
    connection (and warns), while the CRM handoff still completes and flips the phase.
    """
    scope_id: str | None = request.app.state.scope_id
    return _build_chat_connector(scope_id)


def get_chat_mirror(request: Request) -> ChatMirror | None:
    """Build the post-handoff chat mirror from the boot-resolved scope_id.

    ``None`` when the channel is degraded (no scope_id) — the mirror is then skipped.
    """
    scope_id: str | None = request.app.state.scope_id
    if scope_id is None:
        return None
    return ChatMirror(get_kommo_chats_client(), scope_id)


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
    handoff_runner: Annotated[HandoffRunner, Depends(get_handoff_runner)],
    chat_connector: Annotated[
        ChatConnector | None, Depends(get_chat_connector)
    ],
    chat_mirror: Annotated[ChatMirror | None, Depends(get_chat_mirror)],
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

    # 2b. Human handoff: bot stays silent; mirror the message into the Kommo chat.
    if await handoff.is_handed_off(redis, sender):
        schedule_mirror(chat_mirror, store, msg)
        log.info("handoff_mirror_scheduled", sender=sender)
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
        redis, channel, llm, store, routing, corpus, handoff_runner,
        chat_connector, sender, msg.message_id, config,
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
    if not signature or not signer.verify_inbound(body, signature):
        log.warning("kommo_invalid_signature", scope_id=scope_id)
        return Response(status_code=401)

    try:
        await enqueue_inbound(redis, scope_id, body)
    except Exception:
        log.exception("kommo_enqueue_failed", scope_id=scope_id)
        return Response(status_code=500)

    log.info("kommo_inbound_enqueued", scope_id=scope_id)
    return Response(status_code=200)
