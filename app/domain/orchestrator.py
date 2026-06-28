"""Per-turn orchestrator: the conversation backbone (route, collect, answer).

One turn:

* if the conversation is not routed yet (``trip_type is None``), run the routing
  pre-phase: classify the trip type from the message (or the disambiguation
  reply) and either start the flow (first schema question) or ask which trip it
  is. No understanding/slot-filling happens during routing;
* otherwise understand the turn (filled + question + wants_human), merge it, and:
  - if the user asked for a human → hand off immediately (``pidió_humano``);
  - else charge a failed attempt to the last-asked required slot when the answer
    was genuinely unusable, giving up on it after 3 failures (it goes ``pending``);
  - compute the flow continuation: the next askable slot's prompt (retries are
    reformulated) or the handoff farewell (``completa`` if all requireds are
    satisfied, ``atorado`` if a required was given up on);
  - if the turn carried a question, answer it from the corpus (CAG) and prepend
    that answer to the continuation, in a single message.

State is persisted between turns in Redis.
"""

from __future__ import annotations

import json
import time
from typing import cast

import structlog
from redis.asyncio import Redis

from app.answering.answerer import answer_question
from app.concurrency.handoff import set_handoff
from app.crm.kommo_chats import KommoChatUser
from app.crm.kommo_crm import KommoCrmError
from app.domain.chat_connection import ChatConnector
from app.domain.completeness import is_satisfied, next_slot_to_ask
from app.domain.handoff_orchestration import HandoffRunner, phone_from_sender
from app.domain.models import HandoffReason, IncomingMessage
from app.domain.state import (
    ConversationState,
    Phase,
    StateStore,
    merge_slots,
)
from app.llm.base import LLM
from app.routing.campaign import RoutingConfig, classify_trip_type
from app.understanding.engine import TurnContext, Understanding, understand_turn
from app.understanding.schemas import (
    SlotSpec,
    TripSchema,
    TripType,
    descriptor_for,
    extraction_model_for,
)

log = structlog.get_logger()

# completa farewell — personalized with the customer's name when one is captured
# (it always is at completa, since nombre_cliente is required); a clean name-less
# fallback covers the defensive routing-turn handoff where no name exists yet.
_FAREWELL_COMPLETE_NAMED = (
    "¡Muchas gracias, {name}! 😊 Con esto tengo todo lo necesario. En un momento, "
    "un asesor de TOPVIAJES se pondrá en contacto con usted para preparar su "
    "cotización. 🙌"
)
_FAREWELL_COMPLETE_PLAIN = (
    "¡Muchas gracias! 😊 Con esto tengo todo lo necesario. En un momento, un asesor "
    "de TOPVIAJES se pondrá en contacto con usted para preparar su cotización. 🙌"
)
_FAREWELL_STUCK = (
    "¡Gracias! Con lo que me compartiste, un asesor continuará contigo para "
    "afinar los detalles que falten. 🙌"
)
_FAREWELL_HUMAN = (
    "¡Claro! En un momento te comunico con un asesor que continuará tu "
    "atención. 🙌"
)
_FAREWELL_BY_REASON: dict[HandoffReason, str] = {
    HandoffReason.COMPLETE: _FAREWELL_COMPLETE_PLAIN,
    HandoffReason.STUCK: _FAREWELL_STUCK,
    HandoffReason.HUMAN_REQUESTED: _FAREWELL_HUMAN,
}


def _farewell_for(reason: HandoffReason, name: str | None) -> str:
    """The farewell for ``reason``; only completa is personalized with ``name``.

    The other reasons (atorado / pidió_humano) keep their fixed text — adding the
    name to completa does not require touching them.
    """
    if reason is HandoffReason.COMPLETE and isinstance(name, str) and name.strip():
        return _FAREWELL_COMPLETE_NAMED.format(name=name)
    return _FAREWELL_BY_REASON[reason]

# One-time opening greeting, prepended to the first slot question once the campaign
# has resolved the trip type. Trip-aware so it names the destination; the only place
# an emoji appears (slot prompts carry none).
_GREETING_BY_TRIP: dict[TripType, str] = {
    TripType.CRUISE: (
        "¡Hola! 😊 Bienvenido(a) a TOPVIAJES. Con mucho gusto le ayudamos a "
        "planear su crucero."
    ),
    TripType.EUROPE: (
        "¡Hola! 😊 Bienvenido(a) a TOPVIAJES. Con mucho gusto le ayudamos a "
        "planear su viaje por Europa."
    ),
    TripType.ASIA: (
        "¡Hola! 😊 Bienvenido(a) a TOPVIAJES. Con mucho gusto le ayudamos a "
        "planear su viaje por Asia."
    ),
}

# Disambiguation question asked when the trip type cannot be inferred (adjustable).
# It opens with its own greeting (it is the customer's first message when there is
# no campaign placeholder), so it also counts as "greeted".
_DISAMBIGUATION_QUESTION = (
    "¡Hola! 😊 Para ayudarle mejor, ¿su viaje sería un crucero, un viaje por "
    "Europa o un viaje por Asia?"
)

# Re-ask wording escalates so a retry never repeats the question literally.
_REASK_PREFIXES = (
    "Perdona, no logré captar ese dato. ",
    "Para poder darte una cotización necesito este dato. ",
)

# A 3rd failed attempt (count > this) gives up on the required slot.
_MAX_FAILED_ATTEMPTS = 2


async def handle_message(
    msg: IncomingMessage,
    llm: LLM,
    redis: Redis,
    store: StateStore,
    routing: RoutingConfig,
    corpus: str,
    handoff_runner: HandoffRunner,
    chat_connector: ChatConnector | None,
    inactivity_deadline_s: float,
) -> str | None:
    """Run one conversation turn and return the bot's reply (``None`` = silent).

    State is the durable source of truth (``store``); ``redis`` carries only the
    handoff fast-path flag. ``routing`` has the campaign pre-fill phrases;
    ``corpus`` is the answerer's knowledge base; ``handoff_runner`` executes the CRM
    handoff sequence; ``chat_connector`` connects the Chats API chat at handoff
    (``None`` = channel degraded at boot → chat connection skipped);
    ``inactivity_deadline_s`` is how long after the name is captured a silent
    conversation is auto-handed-off (config, threaded from Settings).
    """
    state = await store.load(msg.sender) or ConversationState()

    # Handoff idempotency backstop: if the durable state says this conversation
    # was already handed off, stay silent and do NOT re-trigger handoff — even if
    # the Redis fast-path flag was lost (e.g. a Redis restart). Restore the flag.
    if state.phase is Phase.HANDED_OFF:
        await set_handoff(redis, msg.sender)
        return None

    # Routing pre-phase: a not-yet-routed conversation picks a trip type first.
    if state.trip_type is None:
        return await _route(
            redis, store, msg, state, routing, handoff_runner, chat_connector,
            inactivity_deadline_s,
        )

    trip_type = state.trip_type  # routed: non-None for the rest of the turn
    descriptor = descriptor_for(trip_type)

    understanding = await understand_turn(
        llm,
        extraction_model_for(trip_type),
        msg.text,
        TurnContext(last_asked=state.last_asked, known=state.slots),
    )
    state.slots = merge_slots(state.slots, understanding.filled)

    # A request for a human escalates immediately, before any slot/answer logic.
    if understanding.wants_human:
        reason = HandoffReason.HUMAN_REQUESTED
        farewell = _FAREWELL_BY_REASON[reason]
        return await _handoff(
            redis, store, msg, state, reason, farewell, handoff_runner,
            chat_connector,
        )

    _count_failed_attempt(descriptor, state, understanding)

    # The flow continuation: the next slot to ask, or the handoff farewell.
    nxt = next_slot_to_ask(descriptor, state.slots, state.asked, state.pending)
    if nxt is None:
        reason = HandoffReason.STUCK if state.pending else HandoffReason.COMPLETE
        reply = await _with_answer(
            understanding, llm, corpus, trip_type, state.last_bot_message,
            _farewell_for(reason, state.slots.get("nombre_cliente")),
        )
        return await _handoff(
            redis, store, msg, state, reason, reply, handoff_runner, chat_connector
        )

    state.asked.add(nxt.name)
    state.last_asked = nxt.name
    # Acknowledge the name once, on the first question after it was captured.
    name = state.slots.get("nombre_cliente")
    ack = ""
    if not state.name_acknowledged and isinstance(name, str) and name.strip():
        ack = f"¡Mucho gusto, {name}! 😊 "
        state.name_acknowledged = True
    reply = await _with_answer(
        understanding, llm, corpus, trip_type, state.last_bot_message,
        f"{ack}{_ask_prompt(nxt, state.attempts)}",
    )
    return await _send(store, msg, state, reply, inactivity_deadline_s)


async def _route(
    redis: Redis,
    store: StateStore,
    msg: IncomingMessage,
    state: ConversationState,
    routing: RoutingConfig,
    handoff_runner: HandoffRunner,
    chat_connector: ChatConnector | None,
    inactivity_deadline_s: float,
) -> str:
    """Routing pre-phase: classify the trip type and start the flow, or ask.

    We do NOT run understanding here: a pre-fill / disambiguation reply is not
    slot data. Known limit (out of scope): ``wants_human`` is not detected during
    routing (a 1-2 message pre-flow window); it works normally once routed.
    """
    trip_type = classify_trip_type(msg.text, msg.referral, routing)
    if trip_type is None:
        # The disambiguation greets ("¡Hola!..."), so the customer is now greeted:
        # the first slot question (after they pick a type) must NOT greet again.
        state.greeted = True
        return await _send(
            store, msg, state, _DISAMBIGUATION_QUESTION, inactivity_deadline_s
        )

    state.trip_type = trip_type
    descriptor = descriptor_for(trip_type)
    nxt = next_slot_to_ask(descriptor, state.slots, state.asked, state.pending)
    if nxt is None:  # a schema with no askable slots — defensive, not expected
        reason = HandoffReason.COMPLETE
        farewell = _farewell_for(reason, state.slots.get("nombre_cliente"))
        return await _handoff(
            redis, store, msg, state, reason, farewell, handoff_runner,
            chat_connector,
        )

    state.asked.add(nxt.name)
    state.last_asked = nxt.name
    # Prepend the trip-aware greeting on the first slot question — but only if the
    # customer was not already greeted (a prior disambiguation turn). One greeting.
    greeting = "" if state.greeted else f"{_GREETING_BY_TRIP[trip_type]}\n\n"
    state.greeted = True
    return await _send(
        store, msg, state, f"{greeting}{_ask_prompt(nxt, state.attempts)}",
        inactivity_deadline_s,
    )


async def _with_answer(
    understanding: Understanding,
    llm: LLM,
    corpus: str,
    trip_type: TripType,
    last_bot_message: str | None,
    base: str,
) -> str:
    """Prepend the answer to a turn's question (if any) to ``base``.

    ``last_bot_message`` is the previous turn's message (we update it after
    composing) so follow-ups like "¿y eso?" keep their context.
    """
    if not understanding.question:
        return base
    answer = await answer_question(
        llm, corpus, trip_type, understanding.question, last_bot_message
    )
    return f"{answer}\n\n{base}"


def _count_failed_attempt(
    descriptor: TripSchema, state: ConversationState, understanding: Understanding
) -> None:
    """Charge a failed attempt to the last-asked required slot.

    Only a genuinely unusable answer counts: no data extracted and no question.
    Digressions (a question) and out-of-order answers (data for another slot) do
    not penalize the user. The 3rd failure marks the slot ``pending``.
    """
    last = state.last_asked
    if last is None:
        return
    slot = next((s for s in descriptor.slots if s.name == last), None)
    if slot is None or not slot.required:
        return
    if is_satisfied(slot, state.slots):
        return
    if understanding.filled or understanding.question:
        return
    state.attempts[last] = state.attempts.get(last, 0) + 1
    if state.attempts[last] > _MAX_FAILED_ATTEMPTS:
        state.pending.add(last)


def _ask_prompt(slot: SlotSpec, attempts: dict[str, int]) -> str:
    """The prompt to send; reformulated on a retry so it is never literal."""
    failures = attempts.get(slot.name, 0)
    if slot.required and failures > 0:
        prefix = _REASK_PREFIXES[min(failures - 1, len(_REASK_PREFIXES) - 1)]
        return f"{prefix}{slot.prompt}"
    return slot.prompt


async def _send(
    store: StateStore,
    msg: IncomingMessage,
    state: ConversationState,
    reply: str,
    inactivity_deadline_s: float,
) -> str:
    """Record the outgoing message, persist state, and return ``reply``.

    Once the customer's name is captured, (re-)arm the inactivity deadline: every
    continuing turn pushes it forward, so it tracks the last customer activity. A
    handoff never reaches here (it goes through ``_handoff``), so the phase is always
    COLLECTING at this point.
    """
    state.last_bot_message = reply
    name = state.slots.get("nombre_cliente")
    if isinstance(name, str) and name.strip():
        state.inactivity_deadline = time.time() + inactivity_deadline_s
    await store.save(msg.sender, state)
    return reply


def _is_chat_already_linked(exc: KommoCrmError) -> bool:
    """Whether ``exc`` is Kommo's 400 "chat already linked to another entity".

    Robust + never raises: a non-2xx ``KommoCrmError`` carries the response body as a
    ``str`` (or ``None``); parse it defensively and match the specific
    ``AlreadyExists`` code, so ONLY this error is swallowed (any other propagates).
    If we cannot confirm it is AlreadyExists, return ``False`` (do not mask).
    """
    if exc.status != 400 or not exc.body:
        return False
    try:
        payload: object = json.loads(exc.body)
    except ValueError:
        return False
    if not isinstance(payload, dict):
        return False
    return cast("dict[str, object]", payload).get("code") == "AlreadyExists"


async def connect_chat_at_handoff(
    state: ConversationState,
    sender: str,
    phone: str,
    customer_name: str,
    contact_id: int,
    *,
    chat_connector: ChatConnector | None,
    store: StateStore,
) -> None:
    """Connect the B1 Chats chat at handoff: create ONCE + persist + ALWAYS link.

    Shared by the turn-based ``_handoff`` and the inactivity headless handoff. The
    ``conversation_id`` is the customer phone (B1's deterministic map key). The chat
    is created only if not already linked (gated on ``state.chat_id``) and persisted
    IMMEDIATELY, so a link failure never recreates it; the link is idempotent, so a
    retry re-links. A degraded channel (``chat_connector`` None) is skipped + warned.
    Any create/link failure RAISES (the caller then does not flip the phase).
    """
    if chat_connector is None:
        log.warning("handoff_chat_skipped_channel_unavailable", sender=sender)
        return
    chat_id = state.chat_id
    if chat_id is None:
        user = KommoChatUser(id=f"wa-{phone}", name=customer_name, phone=phone)
        chat_id = await chat_connector.create_chat(phone, user)  # 1) create
        state.chat_id = chat_id
        await store.save(sender, state)  # 2) PERSIST between create and link
    try:
        await chat_connector.link(contact_id, chat_id)  # 3) ALWAYS link
    except KommoCrmError as exc:
        if not _is_chat_already_linked(exc):
            raise
        # Multi-contact split (Fix B's territory): the chat is already linked to a
        # contact of this phone, not the one resolved now. NON-FATAL — don't block
        # the flip; the bot unblocks and B2/B3 (phone-keyed) resume. The advisor may
        # need a manual Kommo contact merge to see the chat on the qualified lead.
        log.warning("handoff_chat_already_linked", sender=sender, body=exc.body)


async def _handoff(
    redis: Redis,
    store: StateStore,
    msg: IncomingMessage,
    state: ConversationState,
    reason: HandoffReason,
    reply: str,
    handoff_runner: HandoffRunner,
    chat_connector: ChatConnector | None,
) -> str:
    """Run the CRM handoff + connect the chat, then flip the phase/flag; return reply.

    ``reply`` is the full outgoing message (a farewell, possibly with an answered
    question prepended). Order matters (a half-done lead is worse than no lead):

    1. persist progress first (phase still COLLECTING) so a failure can retry next
       turn against the complete state without losing this turn's slots;
    2. run the CRM sequence (lead+contact, note, fields, stage) — RAISES on failure;
    3. connect the Chats API chat (B1): create it ONCE — gated by ``state.chat_id`` and
       persisted IMMEDIATELY (between create and link) so a link failure never
       recreates it — then ALWAYS link it to the contact (idempotent, so a retried
       turn re-links). A degraded channel (``chat_connector`` None) is skipped with a
       warning. Any create/link failure RAISES → no flip → retry next turn;
    4. only if all of the above succeeded, flip the phase + handoff flag (the point of
       no return); the entry short-circuit then keeps the bot silent.
    """
    state.last_bot_message = reply
    await store.save(msg.sender, state)

    phone = phone_from_sender(msg.sender)
    name = state.slots.get("nombre_cliente")
    customer_name = name if isinstance(name, str) and name.strip() else phone

    # Idempotency marker: run the CRM sequence ONCE. A retry (e.g. after a chat-link
    # failure left phase=COLLECTING) reuses the persisted lead/contact and SKIPS run,
    # so the note/fields are not re-posted and the link targets the SAME contact.
    contact_id = state.contact_id
    if contact_id is None:
        result = await handoff_runner.run(
            reason=reason,
            phone=phone,
            customer_name=customer_name,
            slots=state.slots,
            pending=tuple(sorted(state.pending)),
        )
        state.lead_id = result.lead_id
        state.contact_id = contact_id = result.contact_id
        await store.save(msg.sender, state)  # persist the marker BEFORE the link

    await connect_chat_at_handoff(
        state, msg.sender, phone, customer_name, contact_id,
        chat_connector=chat_connector, store=store,
    )

    state.phase = Phase.HANDED_OFF
    state.last_asked = None
    state.inactivity_deadline = None  # handed off -> the inactivity timer is moot
    await store.save(msg.sender, state)
    await set_handoff(redis, msg.sender)
    return reply
