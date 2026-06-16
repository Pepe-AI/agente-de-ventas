"""Tests for understand_turn with a mocked LLM (no real API calls).

The engine is now *pure extraction*: it returns the slots it could fill plus any
detected user question. Deciding what is still missing/required is the
orchestrator's job, so ``Understanding`` no longer carries ``missing``. The
``question`` field is composed onto the pure business model internally, so the
trip descriptors stay clean.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.understanding.engine import TurnContext, understand_turn


class _Mini(BaseModel):
    """A tiny pure extraction model (business fields only, no ``question``)."""

    a: int | None = None
    b: str | None = None


class FakeLLM:
    """Returns a preset instance of the *composed* schema; records prompts.

    The engine composes ``question`` onto the pure model before calling, so the
    ``schema`` this receives already has a ``question`` field.
    """

    def __init__(self, **values: object) -> None:
        self._values = values
        self.prompts: list[str] = []

    async def complete_structured(
        self, prompt: str, schema: type[BaseModel]
    ) -> BaseModel:
        self.prompts.append(prompt)
        return schema(**self._values)


_NO_CONTEXT = TurnContext(last_asked=None, known={})


async def test_fills_slots_and_extracts_question() -> None:
    llm = FakeLLM(a=2, b=None, question="¿hay wifi a bordo?")

    result = await understand_turn(llm, _Mini, "somos 2, ¿hay wifi?", _NO_CONTEXT)

    assert result.filled == {"a": 2}
    assert result.question == "¿hay wifi a bordo?"


async def test_question_never_leaks_into_filled() -> None:
    llm = FakeLLM(a=1, b="x", question="¿precio?")

    result = await understand_turn(llm, _Mini, "texto", _NO_CONTEXT)

    assert "question" not in result.filled
    assert result.filled == {"a": 1, "b": "x"}


async def test_no_question_is_none() -> None:
    llm = FakeLLM(a=None, b="x", question=None)

    result = await understand_turn(llm, _Mini, "texto", _NO_CONTEXT)

    assert result.filled == {"b": "x"}
    assert result.question is None


async def test_context_last_asked_is_passed_to_the_model() -> None:
    # The context lets the model resolve a terse answer ("4") to the right slot.
    llm = FakeLLM(a=4, b=None, question=None)
    context = TurnContext(last_asked="pasajeros_crucero", known={})

    await understand_turn(llm, _Mini, "4", context)

    assert "pasajeros_crucero" in llm.prompts[0]


async def test_context_known_state_is_passed_to_the_model() -> None:
    llm = FakeLLM(a=1, b=None, question=None)
    context = TurnContext(last_asked=None, known={"nombre_cliente": "Ana"})

    await understand_turn(llm, _Mini, "texto", context)

    assert "Ana" in llm.prompts[0]


async def test_detects_wants_human() -> None:
    llm = FakeLLM(a=None, b=None, wants_human=True)

    result = await understand_turn(
        llm, _Mini, "quiero hablar con una persona", _NO_CONTEXT
    )

    assert result.wants_human is True


async def test_wants_human_defaults_false_when_absent() -> None:
    llm = FakeLLM(a=1, b=None, question=None)

    result = await understand_turn(llm, _Mini, "texto", _NO_CONTEXT)

    assert result.wants_human is False
