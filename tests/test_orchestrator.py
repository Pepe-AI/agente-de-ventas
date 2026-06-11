"""Tests for the domain orchestration seam (handle_message) with a mocked LLM."""

from __future__ import annotations

from app.domain.models import IncomingMessage
from app.domain.orchestrator import handle_message
from app.understanding.schemas import DummyReservation


class FakeLLM:
    def __init__(self, result: DummyReservation) -> None:
        self._result = result

    async def complete_structured(
        self, prompt: str, schema: type[DummyReservation]
    ) -> DummyReservation:
        return self._result


async def test_handle_message_reports_understanding() -> None:
    llm = FakeLLM(
        DummyReservation(
            num_people=2, travel_date=None, has_id=None, question="¿hay wifi?"
        )
    )
    msg = IncomingMessage(
        sender="whatsapp:+1", text="somos 2, ¿hay wifi?", message_id="SM1"
    )

    reply = await handle_message(msg, llm)

    assert reply == (
        "Entendí: {'num_people': 2}. "
        "Falta: ['travel_date', 'has_id']. "
        "Pregunta: ¿hay wifi?"
    )
