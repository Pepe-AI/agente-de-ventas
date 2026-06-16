"""Tests for the per-turn orchestrator loop (mocked LLM, fakeredis).

Covers the happy-path backbone: collecting required slots in order, the budget
and destination escapes, the minors-need-ages condition, terse answers resolved
via context, persistent state, and the ``completa`` handoff.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fakeredis import FakeAsyncRedis
from pydantic import BaseModel

import app.domain.orchestrator as orch
from app.concurrency.handoff import is_handed_off
from app.domain.models import HandoffReason, IncomingMessage
from app.domain.orchestrator import FAREWELL, handle_message
from app.domain.state import ConversationState, Phase, load_state, save_state
from app.understanding.schemas import (
    Budget,
    Passengers,
    SlotSpec,
    TripType,
    descriptor_for,
)

SENDER = "whatsapp:+5215512345678"


class ScriptedLLM:
    """Returns a preset extraction per turn; records the prompts it received."""

    def __init__(self, *turns: dict[str, object]) -> None:
        self._turns = list(turns)
        self.calls = 0
        self.prompts: list[str] = []

    async def complete_structured(
        self, prompt: str, schema: type[BaseModel]
    ) -> BaseModel:
        self.prompts.append(prompt)
        preset = self._turns[self.calls]
        self.calls += 1
        return schema(**preset)


def _msg(text: str) -> IncomingMessage:
    return IncomingMessage(sender=SENDER, text=text, message_id="SM1")


def _prompt_of(trip: TripType, slot_name: str) -> str:
    slot: SlotSpec = next(
        s for s in descriptor_for(trip).slots if s.name == slot_name
    )
    return slot.prompt


async def _seed(
    redis: FakeAsyncRedis, trip: TripType, slots: dict[str, object], last_asked: str
) -> None:
    state = ConversationState(
        trip_type=trip, slots=slots, phase=Phase.COLLECTING, last_asked=last_asked
    )
    await save_state(redis, SENDER, state)


# --- Asking the next required slot in order --------------------------------


async def test_asks_first_required_slot_on_empty_state() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    llm = ScriptedLLM({})  # extracted nothing

    reply = await handle_message(
        _msg("hola"), llm, redis, descriptor_for(TripType.CRUISE)
    )

    assert reply == _prompt_of(TripType.CRUISE, "nombre_cliente")


async def test_state_persists_between_turns() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    llm = ScriptedLLM({"nombre_cliente": "Ana"})

    reply = await handle_message(
        _msg("soy Ana"), llm, redis, descriptor_for(TripType.CRUISE)
    )

    assert reply == _prompt_of(TripType.CRUISE, "ruta_crucero")
    state = await load_state(redis, SENDER, TripType.CRUISE)
    assert state.slots["nombre_cliente"] == "Ana"
    assert state.last_asked == "ruta_crucero"


# --- Budget escape ----------------------------------------------------------


async def test_budget_defer_to_advisor_satisfies_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orch, "relay_to_human", AsyncMock())
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(
        redis,
        TripType.CRUISE,
        {
            "nombre_cliente": "Ana",
            "ruta_crucero": "Caribe",
            "fechas_crucero": "julio",
            "pasajeros_crucero": {"adults": 2, "minors_mentioned": False},
        },
        last_asked="presupuesto_crucero",
    )
    llm = ScriptedLLM({"presupuesto_crucero": Budget(defer_to_advisor=True)})

    reply = await handle_message(
        _msg("prefiero revisarlo con un asesor"),
        llm,
        redis,
        descriptor_for(TripType.CRUISE),
    )

    # Budget satisfied by the advisor escape -> nothing left -> handoff farewell.
    assert reply == FAREWELL


# --- Destination escape (Europe) -------------------------------------------


async def test_vague_destination_satisfied_by_experience() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(
        redis, TripType.EUROPE, {"nombre_cliente": "Ana"}, last_asked="paises_europa"
    )
    # User gives an experience, not a concrete country.
    llm = ScriptedLLM({"experiencia_europa": "algo romántico y tranquilo"})

    reply = await handle_message(
        _msg("algo romántico y tranquilo"),
        llm,
        redis,
        descriptor_for(TripType.EUROPE),
    )

    # Destination requirement is satisfied; we move on, not re-ask it.
    assert reply == _prompt_of(TripType.EUROPE, "fechas_europa")
    assert reply != _prompt_of(TripType.EUROPE, "paises_europa")


# --- Minors need ages -------------------------------------------------------


async def test_minors_without_ages_keeps_asking_then_satisfied() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(
        redis,
        TripType.CRUISE,
        {
            "nombre_cliente": "Ana",
            "ruta_crucero": "Caribe",
            "fechas_crucero": "julio",
        },
        last_asked="pasajeros_crucero",
    )
    llm = ScriptedLLM(
        {"pasajeros_crucero": Passengers(adults=2, minors_mentioned=True)},
        {"pasajeros_crucero": Passengers(minor_ages=[8])},
    )
    descriptor = descriptor_for(TripType.CRUISE)

    # Turn 1: minors mentioned but no ages -> keep asking passengers.
    reply1 = await handle_message(_msg("somos 2 y un niño"), llm, redis, descriptor)
    assert reply1 == _prompt_of(TripType.CRUISE, "pasajeros_crucero")

    # Turn 2: ages given; merge keeps adults=2 -> satisfied -> move to budget.
    reply2 = await handle_message(_msg("tiene 8 años"), llm, redis, descriptor)
    assert reply2 == _prompt_of(TripType.CRUISE, "presupuesto_crucero")


# --- Terse answer resolved via context -------------------------------------


async def test_terse_answer_uses_context_then_resolves_slot() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(
        redis,
        TripType.CRUISE,
        {
            "nombre_cliente": "Ana",
            "ruta_crucero": "Caribe",
            "fechas_crucero": "julio",
        },
        last_asked="pasajeros_crucero",
    )
    llm = ScriptedLLM(
        {"pasajeros_crucero": Passengers(adults=4, minors_mentioned=False)}
    )

    reply = await handle_message(
        _msg("4"), llm, redis, descriptor_for(TripType.CRUISE)
    )

    # The orchestrator fed the last-asked slot as context so "4" could be mapped.
    assert "pasajeros_crucero" in llm.prompts[0]
    # And the resolved passengers satisfy the slot -> we move on to budget.
    assert reply == _prompt_of(TripType.CRUISE, "presupuesto_crucero")


# --- Completion / handoff ---------------------------------------------------


async def test_completion_hands_off_and_relays_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay = AsyncMock()
    monkeypatch.setattr(orch, "relay_to_human", relay)
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(
        redis,
        TripType.CRUISE,
        {
            "nombre_cliente": "Ana",
            "ruta_crucero": "Caribe",
            "fechas_crucero": "julio",
            "pasajeros_crucero": {"adults": 2, "minors_mentioned": False},
        },
        last_asked="presupuesto_crucero",
    )
    llm = ScriptedLLM({"presupuesto_crucero": Budget(amount="2000-3000 USD")})

    reply = await handle_message(
        _msg("entre 2000 y 3000"), llm, redis, descriptor_for(TripType.CRUISE)
    )

    assert reply == FAREWELL
    # Handoff flag set: the bot stays silent afterwards.
    assert await is_handed_off(redis, SENDER)
    # The relay stub got the handoff event with reason/trip/slots.
    relay.assert_awaited_once()
    event = relay.await_args.args[1]
    assert event.reason is HandoffReason.COMPLETE
    assert event.trip_type == TripType.CRUISE.value
    assert event.slots["nombre_cliente"] == "Ana"
    # State recorded as completed.
    state = await load_state(redis, SENDER, TripType.CRUISE)
    assert state.phase is Phase.COMPLETED


async def test_no_reply_logic_runs_after_handoff_is_silent() -> None:
    # Once handed off, the production webhook short-circuits before the loop;
    # the orchestrator's guarantee is simply that the flag is set.
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(
        redis,
        TripType.CRUISE,
        {
            "nombre_cliente": "Ana",
            "ruta_crucero": "Caribe",
            "fechas_crucero": "julio",
            "pasajeros_crucero": {"adults": 2, "minors_mentioned": False},
        },
        last_asked="presupuesto_crucero",
    )
    llm = ScriptedLLM({"presupuesto_crucero": Budget(defer_to_advisor=True)})

    await handle_message(
        _msg("con asesor"), llm, redis, descriptor_for(TripType.CRUISE)
    )

    assert await is_handed_off(redis, SENDER)
