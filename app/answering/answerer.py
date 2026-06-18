"""CAG answerer: answer a user's question grounded in the full corpus.

Context-augmented generation (no retrieval / embeddings / vector store): the
*entire* corpus is placed in the prompt — it fits Gemini's window with room to
spare. This is a second LLM call, made only when a turn carries a question, and
it reuses the existing :class:`~app.llm.base.LLM` port (no new method) via a
tiny ``Answer`` schema.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.llm.base import LLM
from app.understanding.schemas import TripType

_GUARDRAILS = (
    "Eres el asistente de viajes de la agencia topviajes: cálido y orientado a "
    "ayudar. Responde la pregunta del cliente ÚNICAMENTE con base en el corpus "
    "de conocimiento de abajo. Reglas estrictas: "
    "1) NO cotices un precio final ni confirmes disponibilidad; puedes mencionar "
    "los rangos de referencia del corpus, nunca un monto cerrado. "
    "2) Si la respuesta NO está en el corpus, no inventes: difiere con gracia "
    "diciendo que un asesor puede revisarlo contigo. "
    "3) Sé breve, claro y con un tono cálido y consistente con la agencia."
)


class Answer(BaseModel):
    """Structured answer returned by the answerer LLM call."""

    answer: str


def build_answer_prompt(
    corpus: str,
    trip_type: TripType,
    question: str,
    last_bot_message: str | None,
) -> str:
    """Compose the answerer prompt: guardrails + corpus + context + question."""
    parts = [
        _GUARDRAILS,
        f"Corpus de conocimiento:\n{corpus}",
        f"Tipo de viaje en contexto: {trip_type.value}",
    ]
    if last_bot_message:
        parts.append(f"Último mensaje del bot: {last_bot_message}")
    parts.append(f"Pregunta del cliente: {question}")
    return "\n\n".join(parts)


async def answer_question(
    llm: LLM,
    corpus: str,
    trip_type: TripType,
    question: str,
    last_bot_message: str | None,
) -> str:
    """Answer ``question`` grounded in ``corpus`` (a single async LLM call)."""
    prompt = build_answer_prompt(corpus, trip_type, question, last_bot_message)
    result = await llm.complete_structured(prompt, Answer)
    return result.answer
