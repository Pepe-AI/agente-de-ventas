"""Tests for the per-turn orchestrator loop (mocked LLM, fakeredis).

Covers the 4a-extra-1 backbone: asking every askable slot (requireds +
optionals) in flow order, optionals asked once, the budget/destination escapes,
the minors-need-ages condition, terse answers resolved via context, out-of-order
answers, persistent state + `asked` tracking, and the `completa` handoff once
nothing askable remains.
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
    SlotRule,
    SlotSpec,
    TripType,
    descriptor_for,
)

SENDER = "whatsapp:+5215512345678"

_REQUIRED_NO_BUDGET = {
    "nombre_cliente": "Ana",
    "ruta_crucero": "Caribe",
    "fechas_crucero": "julio",
    "pasajeros_crucero": {"adults": 2, "minors_mentioned": False},
}
_REQUIRED_ALL = {
    **_REQUIRED_NO_BUDGET,
    "presupuesto_crucero": {"defer_to_advisor": True},
}


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
    slot = next(s for s in descriptor_for(trip).slots if s.name == slot_name)
    return slot.prompt


def _askable_optionals(trip: TripType) -> set[str]:
    return {s.name for s in descriptor_for(trip).slots if s.askable and not s.required}


def _value_for(slot: SlotSpec) -> object:
    if slot.rule is SlotRule.PASSENGERS:
        return Passengers(adults=2, minors_mentioned=False)
    if slot.rule is SlotRule.BUDGET:
        return Budget(amount="2000-3000 USD")
    return f"valor-{slot.name}"


async def _seed(
    redis: FakeAsyncRedis,
    trip: TripType,
    slots: dict[str, object],
    last_asked: str,
    asked: set[str] | None = None,
    attempts: dict[str, int] | None = None,
    pending: set[str] | None = None,
) -> None:
    state = ConversationState(
        trip_type=trip,
        slots=slots,
        phase=Phase.COLLECTING,
        last_asked=last_asked,
        asked=asked if asked is not None else set(),
        attempts=attempts if attempts is not None else {},
        pending=pending if pending is not None else set(),
    )
    await save_state(redis, SENDER, state)


# --- Asking in flow order ---------------------------------------------------


async def test_asks_first_required_slot_on_empty_state() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    llm = ScriptedLLM({})  # extracted nothing

    reply = await handle_message(
        _msg("hola"), llm, redis, descriptor_for(TripType.CRUISE)
    )

    assert reply == _prompt_of(TripType.CRUISE, "nombre_cliente")


async def test_state_and_asked_persist_between_turns() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    llm = ScriptedLLM({"nombre_cliente": "Ana"})

    reply = await handle_message(
        _msg("soy Ana"), llm, redis, descriptor_for(TripType.CRUISE)
    )

    assert reply == _prompt_of(TripType.CRUISE, "ruta_crucero")
    state = await load_state(redis, SENDER, TripType.CRUISE)
    assert state.slots["nombre_cliente"] == "Ana"
    assert state.last_asked == "ruta_crucero"
    # Only slots the bot actually asked are tracked; nombre was volunteered.
    assert state.asked == {"ruta_crucero"}


async def test_happy_path_asks_every_slot_in_order_then_hands_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay = AsyncMock()
    monkeypatch.setattr(orch, "relay_to_human", relay)
    redis = FakeAsyncRedis(decode_responses=True)
    descriptor = descriptor_for(TripType.CRUISE)
    askable = [s for s in descriptor.slots if s.askable]

    # Turn 1 extracts nothing; each later turn answers the slot just asked.
    presets: list[dict[str, object]] = [{}]
    presets += [{slot.name: _value_for(slot)} for slot in askable]
    llm = ScriptedLLM(*presets)

    replies = [
        await handle_message(_msg("..."), llm, redis, descriptor)
        for _ in range(len(askable) + 1)
    ]

    # Every askable slot (requireds + optionals) was asked in flow order...
    assert replies[:-1] == [s.prompt for s in askable]
    # ...including the cruise experience optional, then a goodbye.
    assert _prompt_of(TripType.CRUISE, "experiencia_crucero") in replies
    assert replies[-1] == FAREWELL
    assert await is_handed_off(redis, SENDER)
    # Optionals were captured and ride along in the handoff event.
    event = relay.await_args.args[1]
    assert event.reason is HandoffReason.COMPLETE
    assert "cabinas_crucero" in event.slots


async def test_experiencia_crucero_is_asked_as_a_step() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(
        redis,
        TripType.CRUISE,
        dict(_REQUIRED_ALL),
        last_asked="tipo_cabina",
        asked={"cabinas_crucero", "tipo_cabina"},
    )
    llm = ScriptedLLM({})

    reply = await handle_message(
        _msg("..."), llm, redis, descriptor_for(TripType.CRUISE)
    )

    assert reply == _prompt_of(TripType.CRUISE, "experiencia_crucero")


# --- Optionals asked once ---------------------------------------------------


async def test_optional_asked_once_then_skipped_even_if_unanswered() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(
        redis, TripType.CRUISE, dict(_REQUIRED_ALL), last_asked="pasajeros_crucero"
    )
    descriptor = descriptor_for(TripType.CRUISE)
    llm = ScriptedLLM({}, {})  # the user ignores the optional both turns

    reply1 = await handle_message(_msg("ok"), llm, redis, descriptor)
    reply2 = await handle_message(_msg("ok"), llm, redis, descriptor)

    assert reply1 == _prompt_of(TripType.CRUISE, "cabinas_crucero")
    # cabinas was asked once and is not repeated; we advance.
    assert reply2 == _prompt_of(TripType.CRUISE, "tipo_cabina")
    state = await load_state(redis, SENDER, TripType.CRUISE)
    assert "cabinas_crucero" in state.asked


async def test_completion_waits_until_optionals_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay = AsyncMock()
    monkeypatch.setattr(orch, "relay_to_human", relay)
    redis = FakeAsyncRedis(decode_responses=True)
    # All requireds satisfied, but no optional asked yet.
    await _seed(
        redis, TripType.CRUISE, dict(_REQUIRED_ALL), last_asked="presupuesto_crucero"
    )
    llm = ScriptedLLM({})

    reply = await handle_message(
        _msg("ok"), llm, redis, descriptor_for(TripType.CRUISE)
    )

    # Requireds done is not enough: it asks an optional, no handoff yet.
    assert reply == _prompt_of(TripType.CRUISE, "cabinas_crucero")
    assert reply != FAREWELL
    assert not await is_handed_off(redis, SENDER)
    relay.assert_not_awaited()


# --- Budget escape ----------------------------------------------------------


async def test_budget_defer_to_advisor_satisfies_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orch, "relay_to_human", AsyncMock())
    redis = FakeAsyncRedis(decode_responses=True)
    # Requireds-minus-budget done; all askable optionals already asked.
    await _seed(
        redis,
        TripType.CRUISE,
        dict(_REQUIRED_NO_BUDGET),
        last_asked="presupuesto_crucero",
        asked=_askable_optionals(TripType.CRUISE),
    )
    llm = ScriptedLLM({"presupuesto_crucero": Budget(defer_to_advisor=True)})

    reply = await handle_message(
        _msg("prefiero revisarlo con un asesor"),
        llm,
        redis,
        descriptor_for(TripType.CRUISE),
    )

    # Budget satisfied by the advisor escape -> nothing askable left -> farewell.
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

    # Destination requirement satisfied; we move on without re-asking it, and
    # never ask the passive experience escape itself.
    assert reply != _prompt_of(TripType.EUROPE, "paises_europa")
    assert reply != _prompt_of(TripType.EUROPE, "experiencia_europa")
    assert reply == _prompt_of(TripType.EUROPE, "servicios_europa")


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

    # Turn 2: ages given; merge keeps adults=2 -> satisfied -> advance to the
    # next askable slot (the first optional).
    reply2 = await handle_message(_msg("tiene 8 años"), llm, redis, descriptor)
    assert reply2 == _prompt_of(TripType.CRUISE, "cabinas_crucero")


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
    # Resolved passengers satisfy the slot -> advance to the next askable slot.
    assert reply == _prompt_of(TripType.CRUISE, "cabinas_crucero")


# --- Out-of-order answers ---------------------------------------------------


async def test_out_of_order_answer_is_accumulated_and_slot_skipped() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(
        redis,
        TripType.CRUISE,
        dict(_REQUIRED_ALL),
        last_asked="cabinas_crucero",
        asked={"cabinas_crucero"},
    )
    # Answering cabinas while also volunteering tipo_cabina (asked next).
    llm = ScriptedLLM({"cabinas_crucero": "1 balcón", "tipo_cabina": "balcón"})

    reply = await handle_message(
        _msg("una cabina con balcón"), llm, redis, descriptor_for(TripType.CRUISE)
    )

    # tipo_cabina was volunteered out of order -> skipped when its turn comes.
    assert reply == _prompt_of(TripType.CRUISE, "experiencia_crucero")
    state = await load_state(redis, SENDER, TripType.CRUISE)
    assert state.slots["tipo_cabina"] == "balcón"


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
        dict(_REQUIRED_NO_BUDGET),
        last_asked="presupuesto_crucero",
        asked=_askable_optionals(TripType.CRUISE),
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


async def test_bot_is_silent_after_handoff() -> None:
    # Once handed off, the production webhook short-circuits before the loop;
    # the orchestrator's guarantee is simply that the flag is set.
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(
        redis,
        TripType.CRUISE,
        dict(_REQUIRED_NO_BUDGET),
        last_asked="presupuesto_crucero",
        asked=_askable_optionals(TripType.CRUISE),
    )
    llm = ScriptedLLM({"presupuesto_crucero": Budget(defer_to_advisor=True)})

    await handle_message(
        _msg("con asesor"), llm, redis, descriptor_for(TripType.CRUISE)
    )

    assert await is_handed_off(redis, SENDER)


# --- Retry counter / stuck (4a-extra-2) ------------------------------------


async def _seed_only_budget_left(
    redis: FakeAsyncRedis, attempts: dict[str, int] | None = None
) -> None:
    """Seed a state where the only askable thing left is the budget required."""
    await _seed(
        redis,
        TripType.CRUISE,
        dict(_REQUIRED_NO_BUDGET),
        last_asked="presupuesto_crucero",
        asked=_askable_optionals(TripType.CRUISE),
        attempts=attempts,
    )


async def test_three_unusable_answers_mark_pending_and_hand_off_atorado(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay = AsyncMock()
    monkeypatch.setattr(orch, "relay_to_human", relay)
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed_only_budget_left(redis)
    descriptor = descriptor_for(TripType.CRUISE)
    llm = ScriptedLLM({}, {}, {})  # three genuinely unusable turns

    await handle_message(_msg("no sé"), llm, redis, descriptor)
    await handle_message(_msg("mmm"), llm, redis, descriptor)
    reply = await handle_message(_msg("ni idea"), llm, redis, descriptor)

    # 3rd failure -> slot given up on -> stuck handoff carrying the pending slot.
    assert reply == orch._FAREWELL_BY_REASON[HandoffReason.STUCK]
    assert await is_handed_off(redis, SENDER)
    event = relay.await_args.args[1]
    assert event.reason is HandoffReason.STUCK
    assert "presupuesto_crucero" in event.pending
    assert event.slots["nombre_cliente"] == "Ana"


async def test_attempts_persist_between_turns() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed_only_budget_left(redis)
    descriptor = descriptor_for(TripType.CRUISE)
    llm = ScriptedLLM({}, {})

    await handle_message(_msg("no sé"), llm, redis, descriptor)
    await handle_message(_msg("mmm"), llm, redis, descriptor)

    state = await load_state(redis, SENDER, TripType.CRUISE)
    assert state.attempts["presupuesto_crucero"] == 2
    assert "presupuesto_crucero" not in state.pending


async def test_question_digression_does_not_count_attempt() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed_only_budget_left(redis)
    llm = ScriptedLLM({"question": "¿aceptan tarjeta de crédito?"})

    reply = await handle_message(
        _msg("¿aceptan tarjeta?"), llm, redis, descriptor_for(TripType.CRUISE)
    )

    state = await load_state(redis, SENDER, TripType.CRUISE)
    assert state.attempts == {}  # a question is not a failed attempt
    assert not await is_handed_off(redis, SENDER)
    # Asked again literally (no failed attempt -> no reformulation).
    assert reply == _prompt_of(TripType.CRUISE, "presupuesto_crucero")


async def test_out_of_order_data_does_not_count_attempt() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed_only_budget_left(redis)
    # Asked budget; the user answers with a different slot instead.
    llm = ScriptedLLM({"pasaporte_crucero": "vigente"})

    await handle_message(
        _msg("sí tengo pasaporte"), llm, redis, descriptor_for(TripType.CRUISE)
    )

    state = await load_state(redis, SENDER, TripType.CRUISE)
    assert state.attempts == {}  # data for another slot is not a failed attempt


async def test_valid_value_after_failure_satisfies_without_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay = AsyncMock()
    monkeypatch.setattr(orch, "relay_to_human", relay)
    redis = FakeAsyncRedis(decode_responses=True)
    # Two prior failures already recorded for budget.
    await _seed_only_budget_left(redis, attempts={"presupuesto_crucero": 2})
    llm = ScriptedLLM({"presupuesto_crucero": Budget(amount="2500 USD")})

    reply = await handle_message(
        _msg("unos 2500"), llm, redis, descriptor_for(TripType.CRUISE)
    )

    # The value lands -> satisfied -> complete, never pending.
    assert reply == FAREWELL
    event = relay.await_args.args[1]
    assert event.reason is HandoffReason.COMPLETE
    assert event.pending == ()
    state = await load_state(redis, SENDER, TripType.CRUISE)
    assert state.pending == set()


async def test_reask_after_failure_is_reformulated_not_literal() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed_only_budget_left(redis)
    llm = ScriptedLLM({})  # one unusable turn

    reply = await handle_message(
        _msg("no sé"), llm, redis, descriptor_for(TripType.CRUISE)
    )

    literal = _prompt_of(TripType.CRUISE, "presupuesto_crucero")
    assert reply != literal  # a retry never repeats the question literally
    assert literal in reply  # but still asks for the same thing


# --- pidió_humano (immediate) ----------------------------------------------


async def test_wants_human_hands_off_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay = AsyncMock()
    monkeypatch.setattr(orch, "relay_to_human", relay)
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(
        redis,
        TripType.CRUISE,
        {"nombre_cliente": "Ana", "ruta_crucero": "Caribe"},
        last_asked="fechas_crucero",
    )
    llm = ScriptedLLM({"wants_human": True})

    reply = await handle_message(
        _msg("mejor quiero hablar con una persona"),
        llm,
        redis,
        descriptor_for(TripType.CRUISE),
    )

    assert reply == orch._FAREWELL_BY_REASON[HandoffReason.HUMAN_REQUESTED]
    assert await is_handed_off(redis, SENDER)
    event = relay.await_args.args[1]
    assert event.reason is HandoffReason.HUMAN_REQUESTED
    # Carries what was captured so far.
    assert event.slots["nombre_cliente"] == "Ana"
    assert event.slots["ruta_crucero"] == "Caribe"
