"""Per-turn orchestrator: the conversation backbone with failure escalation.

One turn = load state, understand the turn (filled + question + wants_human),
merge what was filled, then:

* if the user asked for a human → hand off immediately (``pidió_humano``);
* otherwise charge a failed attempt to the last-asked required slot when the
  answer was genuinely unusable (no data, no question), giving up on the slot
  after 3 failures (it goes ``pending``);
* compute the next askable slot (skipping ``pending``). If one remains, ask it
  (retries are reformulated, not literal). If none remains, hand off with
  reason ``completa`` (all requireds satisfied) or ``atorado`` (a required was
  given up on).

State is persisted between turns in Redis.

Out of scope here (later increments): answering user questions (only detected,
to not count them as failures) and selecting the schema by campaign.
"""

from __future__ import annotations

from redis.asyncio import Redis

from app.concurrency.handoff import set_handoff
from app.crm.relay import relay_to_human
from app.domain.completeness import is_satisfied, next_slot_to_ask
from app.domain.models import HandoffEvent, HandoffReason, IncomingMessage
from app.domain.state import (
    ConversationState,
    Phase,
    load_state,
    merge_slots,
    save_state,
)
from app.llm.base import LLM
from app.understanding.engine import TurnContext, Understanding, understand_turn
from app.understanding.schemas import (
    SlotSpec,
    TripSchema,
    descriptor_for,
    extraction_model_for,
)

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

# Re-ask wording escalates so a retry never repeats the question literally.
_REASK_PREFIXES = (
    "Perdona, no logré captar ese dato. ",
    "Para poder darte una cotización necesito este dato. ",
)

# A 3rd failed attempt (count > this) gives up on the required slot.
_MAX_FAILED_ATTEMPTS = 2


async def handle_message(
    msg: IncomingMessage, llm: LLM, redis: Redis, default: TripSchema
) -> str:
    """Run one conversation turn and return the bot's reply.

    ``default`` is the configured schema for a brand-new conversation; an
    in-progress conversation keeps the schema it started on.
    """
    state = await load_state(redis, msg.sender, default.trip_type)
    descriptor = descriptor_for(state.trip_type)

    understanding = await understand_turn(
        llm,
        extraction_model_for(state.trip_type),
        msg.text,
        TurnContext(last_asked=state.last_asked, known=state.slots),
    )
    state.slots = merge_slots(state.slots, understanding.filled)

    # A request for a human escalates immediately, before any slot logic.
    if understanding.wants_human:
        return await _handoff(redis, msg, state, HandoffReason.HUMAN_REQUESTED)

    _count_failed_attempt(descriptor, state, understanding)

    nxt = next_slot_to_ask(descriptor, state.slots, state.asked, state.pending)
    if nxt is None:
        reason = HandoffReason.STUCK if state.pending else HandoffReason.COMPLETE
        return await _handoff(redis, msg, state, reason)

    state.asked.add(nxt.name)
    state.last_asked = nxt.name
    await save_state(redis, msg.sender, state)
    return _ask_prompt(nxt, state.attempts)


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


async def _handoff(
    redis: Redis,
    msg: IncomingMessage,
    state: ConversationState,
    reason: HandoffReason,
) -> str:
    """Persist, hand off to a human with ``reason``, and return the farewell."""
    state.phase = Phase.COMPLETED
    state.last_asked = None
    await save_state(redis, msg.sender, state)

    await set_handoff(redis, msg.sender)
    await relay_to_human(
        msg,
        HandoffEvent(
            reason=reason,
            trip_type=state.trip_type.value,
            slots=state.slots,
            pending=tuple(sorted(state.pending)),
        ),
    )
    return _FAREWELL_BY_REASON[reason]
