"""Per-turn orchestrator: the conversation backbone (happy path).

One turn = load state, understand the turn, merge what was filled, then compute
the next required slot from the descriptor. If one remains, ask it; if none do,
the form is complete: hand off to a human (reason ``completa``) and say goodbye.
State is persisted between turns in Redis.

Out of scope here (later increments): answering user questions, retries / a
``stuck`` reason, and selecting the schema by campaign.
"""

from __future__ import annotations

from redis.asyncio import Redis

from app.concurrency.handoff import set_handoff
from app.crm.relay import relay_to_human
from app.domain.completeness import next_required_slot
from app.domain.models import HandoffEvent, HandoffReason, IncomingMessage
from app.domain.state import (
    ConversationState,
    Phase,
    load_state,
    merge_slots,
    save_state,
)
from app.llm.base import LLM
from app.understanding.engine import TurnContext, understand_turn
from app.understanding.schemas import (
    TripSchema,
    descriptor_for,
    extraction_model_for,
)

FAREWELL = (
    "¡Gracias! Con esto tengo todo lo necesario. En un momento un asesor "
    "se pondrá en contacto contigo para darte una cotización. 🙌"
)


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
    # 4a-core ignores any detected user question (no answerer yet).

    nxt = next_required_slot(descriptor, state.slots)
    if nxt is None:
        return await _complete(redis, msg, state)

    state.last_asked = nxt.name
    await save_state(redis, msg.sender, state)
    return nxt.prompt


async def _complete(
    redis: Redis, msg: IncomingMessage, state: ConversationState
) -> str:
    """All required slots captured: persist, hand off, and say goodbye."""
    state.phase = Phase.COMPLETED
    state.last_asked = None
    await save_state(redis, msg.sender, state)

    await set_handoff(redis, msg.sender)
    await relay_to_human(
        msg,
        HandoffEvent(
            reason=HandoffReason.COMPLETE,
            trip_type=state.trip_type.value,
            slots=state.slots,
        ),
    )
    return FAREWELL
