"""Turn-understanding engine.

One LLM call over the combined turn text returns the slots it could fill plus
any user question. The engine never invents values: whatever the model leaves
null is reported as missing.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from app.llm.base import LLM

QUESTION_FIELD = "question"

_INSTRUCTIONS = (
    "Eres un extractor de información para reservas de viaje. A partir del "
    "mensaje del usuario, llena solo los campos que puedas inferir con certeza. "
    "Deja en null cualquier campo que no sepas o que sea ambiguo: NUNCA "
    "adivines ni inventes valores. Si el usuario hace una pregunta, cópiala "
    f"textual en el campo '{QUESTION_FIELD}'; si no hay pregunta, déjalo en null."
)


@dataclass(frozen=True, slots=True)
class Understanding:
    """The engine's read of one turn."""

    filled: dict[str, object]
    missing: list[str]
    question: str | None


def _build_prompt(text: str) -> str:
    return f"{_INSTRUCTIONS}\n\nMensaje del usuario:\n{text}"


async def understand_turn(
    llm: LLM, text: str, schema: type[BaseModel]
) -> Understanding:
    """Extract filled/missing slots and any question from a turn.

    ``schema`` defines the slots; every field except ``question`` is a slot. A
    slot left null by the model is reported as missing (never invented).
    """
    result = await llm.complete_structured(_build_prompt(text), schema)
    data = result.model_dump()

    question = data.get(QUESTION_FIELD)
    slots = {key: value for key, value in data.items() if key != QUESTION_FIELD}
    filled = {key: value for key, value in slots.items() if value is not None}
    missing = [key for key, value in slots.items() if value is None]

    return Understanding(filled=filled, missing=missing, question=question)
