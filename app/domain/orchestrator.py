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

import structlog
from redis.asyncio import Redis

from app.answering.answerer import answer_question
from app.concurrency.handoff import set_handoff
from app.crm.kommo_chats import KommoChatUser
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

FAREWELL = (
    "¡Gracias! Con esto tengo todo lo necesario. En un momento un asesor "
    "se pondrá en contacto contigo para darte una cotización. 🙌"
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
    HandoffReason.COMPLETE: FAREWELL,
    HandoffReason.STUCK: _FAREWELL_STUCK,
    HandoffReason.HUMAN_REQUESTED: _FAREWELL_HUMAN,
}

# Disambiguation question asked when the trip type cannot be inferred (adjustable).
_DISAMBIGUATION_QUESTION = (
    "¡Hola! Para ayudarte mejor, ¿tu viaje sería un crucero, un viaje a Europa "
    "o un viaje a Asia?"
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
) -> str | None:
    """Run one conversation turn and return the bot's reply (``None`` = silent).

    State is the durable source of truth (``store``); ``redis`` carries only the
    handoff fast-path flag. ``routing`` has the campaign pre-fill phrases;
    ``corpus`` is the answerer's knowledge base; ``handoff_runner`` executes the CRM
    handoff sequence; ``chat_connector`` connects the Chats API chat at handoff
    (``None`` = channel degraded at boot → chat connection skipped).
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
            redis, store, msg, state, routing, handoff_runner, chat_connector
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
            _FAREWELL_BY_REASON[reason],
        )
        return await _handoff(
            redis, store, msg, state, reason, reply, handoff_runner, chat_connector
        )

    state.asked.add(nxt.name)
    state.last_asked = nxt.name
    reply = await _with_answer(
        understanding, llm, corpus, trip_type, state.last_bot_message,
        _ask_prompt(nxt, state.attempts),
    )
    return await _send(store, msg, state, reply)


async def _route(
    redis: Redis,
    store: StateStore,
    msg: IncomingMessage,
    state: ConversationState,
    routing: RoutingConfig,
    handoff_runner: HandoffRunner,
    chat_connector: ChatConnector | None,
) -> str:
    """Routing pre-phase: classify the trip type and start the flow, or ask.

    We do NOT run understanding here: a pre-fill / disambiguation reply is not
    slot data. Known limit (out of scope): ``wants_human`` is not detected during
    routing (a 1-2 message pre-flow window); it works normally once routed.
    """
    trip_type = classify_trip_type(msg.text, msg.referral, routing)
    if trip_type is None:
        return await _send(store, msg, state, _DISAMBIGUATION_QUESTION)

    state.trip_type = trip_type
    descriptor = descriptor_for(trip_type)
    nxt = next_slot_to_ask(descriptor, state.slots, state.asked, state.pending)
    if nxt is None:  # a schema with no askable slots — defensive, not expected
        reason = HandoffReason.COMPLETE
        farewell = _FAREWELL_BY_REASON[reason]
        return await _handoff(
            redis, store, msg, state, reason, farewell, handoff_runner,
            chat_connector,
        )

    state.asked.add(nxt.name)
    state.last_asked = nxt.name
    return await _send(store, msg, state, _ask_prompt(nxt, state.attempts))


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
    store: StateStore, msg: IncomingMessage, state: ConversationState, reply: str
) -> str:
    """Record the outgoing message, persist state, and return ``reply``."""
    state.last_bot_message = reply
    await store.save(msg.sender, state)
    return reply


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

    pending = tuple(sorted(state.pending))
    phone = phone_from_sender(msg.sender)
    name = state.slots.get("nombre_cliente")
    customer_name = name if isinstance(name, str) and name.strip() else phone
    result = await handoff_runner.run(
        reason=reason,
        phone=phone,
        customer_name=customer_name,
        slots=state.slots,
        pending=pending,
    )

    # B1 chat connection (conversation_id = the customer phone — the deterministic
    # map key for the B2 mirror / B3 drain).
    if chat_connector is not None:
        chat_id = state.chat_id
        if chat_id is None:
            user = KommoChatUser(id=f"wa-{phone}", name=customer_name, phone=phone)
            chat_id = await chat_connector.create_chat(phone, user)  # 1) create
            state.chat_id = chat_id
            await store.save(msg.sender, state)  # 2) PERSIST between create and link
        await chat_connector.link(result.contact_id, chat_id)  # 3) ALWAYS link
    else:
        log.warning("handoff_chat_skipped_channel_unavailable", sender=msg.sender)

    state.phase = Phase.HANDED_OFF
    state.last_asked = None
    await store.save(msg.sender, state)
    await set_handoff(redis, msg.sender)
    return reply
