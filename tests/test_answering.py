"""Tests for the CAG answerer and corpus loader (LLM mocked, no real API)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from app.answering.answerer import Answer, answer_question, build_answer_prompt
from app.answering.corpus import load_corpus
from app.understanding.schemas import TripType

_CORPUS_PATH = Path(__file__).resolve().parent.parent / "app" / "corpus_topviajes.md"


class FakeLLM:
    """Returns a preset Answer; records the prompts and schemas it received."""

    def __init__(self, answer_text: str) -> None:
        self._answer_text = answer_text
        self.prompts: list[str] = []
        self.schemas: list[type[BaseModel]] = []

    async def complete_structured(
        self, prompt: str, schema: type[BaseModel]
    ) -> BaseModel:
        self.prompts.append(prompt)
        self.schemas.append(schema)
        return schema(answer=self._answer_text)


# --- Corpus loader ----------------------------------------------------------


def test_load_corpus_reads_file() -> None:
    corpus = load_corpus(str(_CORPUS_PATH))

    assert "TOPVIAJES" in corpus
    assert len(corpus) > 100


# --- Answerer ---------------------------------------------------------------


async def test_answer_question_returns_the_models_answer() -> None:
    llm = FakeLLM("Todas las cotizaciones incluyen vuelos y hospedaje.")

    result = await answer_question(
        llm,
        "CORPUS",
        TripType.CRUISE,
        "¿qué incluye la cotización?",
        "¿En qué fechas te gustaría viajar?",
    )

    assert result == "Todas las cotizaciones incluyen vuelos y hospedaje."
    assert llm.schemas[0] is Answer


def test_prompt_includes_corpus_question_context_and_last_message() -> None:
    prompt = build_answer_prompt(
        "MARCADOR_DE_CORPUS",
        TripType.EUROPE,
        "¿cuánto cuesta?",
        "¿Qué destinos de Europa te gustaría visitar?",
    )

    assert "MARCADOR_DE_CORPUS" in prompt
    assert "¿cuánto cuesta?" in prompt
    assert "europe" in prompt
    assert "¿Qué destinos de Europa te gustaría visitar?" in prompt


def test_prompt_carries_the_guardrails() -> None:
    prompt = build_answer_prompt("corpus", TripType.ASIA, "¿precio?", None)

    lowered = prompt.lower()
    # Answer only from the corpus; defer gracefully when it is not there.
    assert "corpus" in lowered
    assert "asesor" in lowered
    # Never quote a final price or confirm availability.
    assert "precio final" in lowered
    assert "disponibilidad" in lowered
