"""Domain orchestration seam.

The single point where the conversational engine plugs in. It now runs the
turn-understanding engine over the message text and reports what was understood.
The real orchestrator (state machine, slot-vs-question routing) arrives in a
later increment.
"""

from __future__ import annotations

from app.domain.models import IncomingMessage
from app.llm.base import LLM
from app.understanding.engine import understand_turn
from app.understanding.schemas import DummyReservation

_REPLY_TEMPLATE = "Entendí: {filled}. Falta: {missing}. Pregunta: {question}"


async def handle_message(msg: IncomingMessage, llm: LLM) -> str:
    """Understand the turn and report filled/missing slots and any question."""
    understanding = await understand_turn(llm, msg.text, DummyReservation)
    return _REPLY_TEMPLATE.format(
        filled=understanding.filled,
        missing=understanding.missing,
        question=understanding.question,
    )
