"""Turn-understanding engine (pure extraction).

One LLM call over the combined turn text returns the slots it could fill plus
any user question. The engine never invents values: whatever the model leaves
null is simply absent from ``filled``.

Two design points (increment 4 cleanup):

* The caller passes a *pure* extraction model (business slots only). The engine
  composes the ``question`` field onto it internally, so the trip descriptors
  stay free of engine concerns.
* Deciding what is still required/missing is **not** the engine's job; it only
  reports ``filled`` + ``question``. The orchestrator computes completeness from
  the descriptor and the conversation state.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from pydantic import BaseModel, create_model

from app.llm.base import LLM

QUESTION_FIELD = "question"
WANTS_HUMAN_FIELD = "wants_human"

_INSTRUCTIONS = (
    "Eres un extractor de información para reservas de viaje. A partir del "
    "mensaje del usuario, llena solo los campos que puedas inferir con certeza. "
    "Deja en null cualquier campo que no sepas o que sea ambiguo: NUNCA "
    "adivines ni inventes valores. Usa el contexto para resolver respuestas "
    "breves o ambiguas (p. ej. un número suelto que responde a la última "
    "pregunta). Si el usuario hace una pregunta, cópiala textual en el campo "
    f"'{QUESTION_FIELD}'; si no hay pregunta, déjalo en null. Si el usuario "
    "pide explícitamente hablar con una persona, asesor o humano, marca "
    f"'{WANTS_HUMAN_FIELD}' en true; en caso contrario, false."
)


@dataclass(frozen=True, slots=True)
class TurnContext:
    """What the orchestrator knows going into a turn.

    ``last_asked`` is the slot the bot last asked for (or ``None``); ``known``
    is the state captured so far. Both are fed to the model so it can resolve
    terse or ambiguous answers (e.g. map ``"4"`` to the slot just asked).
    """

    last_asked: str | None
    known: dict[str, object]


@dataclass(frozen=True, slots=True)
class Understanding:
    """The engine's read of one turn: filled slots, any question, and whether
    the user asked to talk to a human."""

    filled: dict[str, object]
    question: str | None
    wants_human: bool


@cache
def _compose_with_meta(model: type[BaseModel]) -> type[BaseModel]:
    """Derive ``model`` + the engine's meta fields (cached per pure model).

    The literal keywords must stay in sync with ``QUESTION_FIELD`` /
    ``WANTS_HUMAN_FIELD`` (used below to strip them back out); a drift would
    surface as a failing test.
    """
    return create_model(
        f"{model.__name__}WithMeta",
        __base__=model,
        question=(str | None, None),
        wants_human=(bool | None, None),
    )


def _render_context(context: TurnContext) -> str:
    if context.last_asked is None and not context.known:
        return ""
    lines = ["Contexto de la conversación:"]
    if context.last_asked is not None:
        lines.append(f"- Última pregunta enviada (slot): {context.last_asked}")
    if context.known:
        lines.append(f"- Datos ya capturados: {context.known}")
    return "\n".join(lines)


def _build_prompt(text: str, context: TurnContext) -> str:
    parts = [_INSTRUCTIONS]
    rendered = _render_context(context)
    if rendered:
        parts.append(rendered)
    parts.append(f"Mensaje del usuario:\n{text}")
    return "\n\n".join(parts)


async def understand_turn(
    llm: LLM,
    extraction_model: type[BaseModel],
    text: str,
    context: TurnContext,
) -> Understanding:
    """Extract filled slots and any question from a turn.

    ``extraction_model`` is the pure business model (no meta fields); the engine
    composes ``question`` and ``wants_human`` onto it before calling the model. A
    slot left null by the model is simply absent from ``filled`` (never invented).
    """
    composed = _compose_with_meta(extraction_model)
    result = await llm.complete_structured(_build_prompt(text, context), composed)
    data = result.model_dump()

    question = data.pop(QUESTION_FIELD, None)
    wants_human = bool(data.pop(WANTS_HUMAN_FIELD, None))
    filled = {key: value for key, value in data.items() if value is not None}

    return Understanding(filled=filled, question=question, wants_human=wants_human)
