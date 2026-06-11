"""Tests for understand_turn with a mocked LLM (no real API calls)."""

from __future__ import annotations

from app.understanding.engine import understand_turn
from app.understanding.schemas import DummyReservation


class FakeLLM:
    """Returns a preset schema instance and records the prompts it received."""

    def __init__(self, result: DummyReservation) -> None:
        self._result = result
        self.prompts: list[str] = []

    async def complete_structured(
        self, prompt: str, schema: type[DummyReservation]
    ) -> DummyReservation:
        self.prompts.append(prompt)
        return self._result


async def test_only_slots_no_question() -> None:
    llm = FakeLLM(
        DummyReservation(
            num_people=3, travel_date="2026-07-01", has_id=True, question=None
        )
    )

    result = await understand_turn(
        llm, "somos 3, viajamos el 1 de julio", DummyReservation
    )

    assert result.filled == {
        "num_people": 3,
        "travel_date": "2026-07-01",
        "has_id": True,
    }
    assert result.missing == []
    assert result.question is None


async def test_only_question_no_slots() -> None:
    llm = FakeLLM(
        DummyReservation(
            num_people=None,
            travel_date=None,
            has_id=None,
            question="¿aceptan tarjeta de crédito?",
        )
    )

    result = await understand_turn(
        llm, "¿aceptan tarjeta de crédito?", DummyReservation
    )

    assert result.filled == {}
    assert result.missing == ["num_people", "travel_date", "has_id"]
    assert result.question == "¿aceptan tarjeta de crédito?"


async def test_slots_and_question_together() -> None:
    llm = FakeLLM(
        DummyReservation(
            num_people=2, travel_date=None, has_id=None, question="¿hay wifi a bordo?"
        )
    )

    result = await understand_turn(llm, "somos 2, ¿hay wifi a bordo?", DummyReservation)

    assert result.filled == {"num_people": 2}
    assert result.missing == ["travel_date", "has_id"]
    assert result.question == "¿hay wifi a bordo?"


async def test_ambiguous_value_stays_missing_not_invented() -> None:
    # The model left num_people null because the turn was ambiguous ("algunos").
    llm = FakeLLM(
        DummyReservation(
            num_people=None, travel_date="2026-08-10", has_id=None, question=None
        )
    )

    result = await understand_turn(llm, "seríamos algunos en agosto", DummyReservation)

    assert "num_people" in result.missing
    assert "num_people" not in result.filled
    assert result.filled == {"travel_date": "2026-08-10"}
    assert result.missing == ["num_people", "has_id"]
