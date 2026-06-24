"""Tests for the per-turn orchestrator loop (mocked LLM, fakeredis).

Covers the 4a-extra-1 backbone: asking every askable slot (requireds +
optionals) in flow order, optionals asked once, the budget/destination escapes,
the minors-need-ages condition, terse answers resolved via context, out-of-order
answers, persistent state + `asked` tracking, and the `completa` handoff once
nothing askable remains.
"""

from __future__ import annotations

import weakref
from unittest.mock import AsyncMock

import pytest
from fakeredis import FakeAsyncRedis
from pydantic import BaseModel

import app.domain.orchestrator as orch
from app.concurrency.handoff import clear_handoff, is_handed_off
from app.crm.kommo_crm import KommoCrmError
from app.domain.models import HandoffReason, IncomingMessage, Referral
from app.domain.orchestrator import FAREWELL, handle_message
from app.domain.state import ConversationState, Phase
from app.routing.campaign import RoutingConfig
from app.understanding.schemas import (
    Budget,
    Passengers,
    SlotRule,
    SlotSpec,
    TripType,
    descriptor_for,
)
from tests.fakes import InMemoryStateStore

SENDER = "whatsapp:+5215512345678"
_CORPUS = "CORPUS DE PRUEBA TOPVIAJES"
_ROUTING = RoutingConfig(prefill_crucero=None, prefill_europa=None, prefill_asia=None)

# State now lives in a StateStore (not Redis). Each test's redis gets its own
# in-memory store, so the existing `_seed`/`_handle` call sites stay unchanged.
_STORES: weakref.WeakKeyDictionary[object, InMemoryStateStore] = (
    weakref.WeakKeyDictionary()
)


def _store_for(redis: object) -> InMemoryStateStore:
    store = _STORES.get(redis)
    if store is None:
        store = InMemoryStateStore()
        _STORES[redis] = store
    return store


async def _load(redis: FakeAsyncRedis) -> ConversationState:
    state = await _store_for(redis).load(SENDER)
    assert state is not None
    return state


class _FakeHandoffRunner:
    """No-op handoff runner that succeeds; records each ``run`` call's kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def run(self, **kwargs: object) -> int:
        self.calls.append(kwargs)
        return 1


class _FailingHandoffRunner:
    """A handoff runner whose CRM sequence fails (to test the no-flip path)."""

    async def run(self, **kwargs: object) -> int:
        raise KommoCrmError("crm down")


async def _handle(
    msg: IncomingMessage,
    llm: object,
    redis: FakeAsyncRedis,
    runner: object | None = None,
) -> str | None:
    # Conversations here are seeded with a trip_type, so the routing gate is
    # skipped and the schema comes from state; routing uses an empty config.
    return await handle_message(
        msg, llm, redis, _store_for(redis), _ROUTING, _CORPUS,
        runner or _FakeHandoffRunner(),
    )


async def _route_turn(
    redis: FakeAsyncRedis, text: str, routing: RoutingConfig = _ROUTING
) -> str | None:
    """A routing pre-phase turn (fresh conversation, no slot data)."""
    return await handle_message(
        _msg(text), ScriptedLLM(), redis, _store_for(redis), routing, _CORPUS,
        _FakeHandoffRunner(),
    )

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
    last_asked: str | None = None,
    asked: set[str] | None = None,
    attempts: dict[str, int] | None = None,
    pending: set[str] | None = None,
    last_bot_message: str | None = None,
) -> None:
    state = ConversationState(
        trip_type=trip,
        slots=slots,
        phase=Phase.COLLECTING,
        last_asked=last_asked,
        asked=asked if asked is not None else set(),
        attempts=attempts if attempts is not None else {},
        pending=pending if pending is not None else set(),
        last_bot_message=last_bot_message,
    )
    await _store_for(redis).save(SENDER, state)


# --- Asking in flow order ---------------------------------------------------


async def test_asks_first_required_slot_on_empty_state() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(redis, TripType.CRUISE, {})  # already routed, nothing asked yet
    llm = ScriptedLLM({})  # extracted nothing

    reply = await _handle(
        _msg("hola"), llm, redis
    )

    assert reply == _prompt_of(TripType.CRUISE, "nombre_cliente")


async def test_state_and_asked_persist_between_turns() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(redis, TripType.CRUISE, {})  # already routed
    llm = ScriptedLLM({"nombre_cliente": "Ana"})

    reply = await _handle(
        _msg("soy Ana"), llm, redis
    )

    assert reply == _prompt_of(TripType.CRUISE, "ruta_crucero")
    state = await _load(redis)
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
    await _seed(redis, TripType.CRUISE, {})  # already routed
    descriptor = descriptor_for(TripType.CRUISE)
    askable = [s for s in descriptor.slots if s.askable]

    # Turn 1 extracts nothing; each later turn answers the slot just asked.
    presets: list[dict[str, object]] = [{}]
    presets += [{slot.name: _value_for(slot)} for slot in askable]
    llm = ScriptedLLM(*presets)

    replies = [
        await _handle(_msg("..."), llm, redis)
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

    reply = await _handle(
        _msg("..."), llm, redis
    )

    assert reply == _prompt_of(TripType.CRUISE, "experiencia_crucero")


# --- Optionals asked once ---------------------------------------------------


async def test_optional_asked_once_then_skipped_even_if_unanswered() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(
        redis, TripType.CRUISE, dict(_REQUIRED_ALL), last_asked="pasajeros_crucero"
    )
    llm = ScriptedLLM({}, {})  # the user ignores the optional both turns

    reply1 = await _handle(_msg("ok"), llm, redis)
    reply2 = await _handle(_msg("ok"), llm, redis)

    assert reply1 == _prompt_of(TripType.CRUISE, "cabinas_crucero")
    # cabinas was asked once and is not repeated; we advance.
    assert reply2 == _prompt_of(TripType.CRUISE, "tipo_cabina")
    state = await _load(redis)
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

    reply = await _handle(
        _msg("ok"), llm, redis
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

    reply = await _handle(
        _msg("prefiero revisarlo con un asesor"),
        llm,
        redis,    )

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

    reply = await _handle(
        _msg("algo romántico y tranquilo"),
        llm,
        redis,    )

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

    # Turn 1: minors mentioned but no ages -> keep asking passengers.
    reply1 = await _handle(_msg("somos 2 y un niño"), llm, redis)
    assert reply1 == _prompt_of(TripType.CRUISE, "pasajeros_crucero")

    # Turn 2: ages given; merge keeps adults=2 -> satisfied -> advance to the
    # next askable slot (the first optional).
    reply2 = await _handle(_msg("tiene 8 años"), llm, redis)
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

    reply = await _handle(
        _msg("4"), llm, redis
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

    reply = await _handle(
        _msg("una cabina con balcón"), llm, redis
    )

    # tipo_cabina was volunteered out of order -> skipped when its turn comes.
    assert reply == _prompt_of(TripType.CRUISE, "experiencia_crucero")
    state = await _load(redis)
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

    reply = await _handle(
        _msg("entre 2000 y 3000"), llm, redis
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
    state = await _load(redis)
    assert state.phase is Phase.HANDED_OFF


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

    await _handle(
        _msg("con asesor"), llm, redis
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
    llm = ScriptedLLM({}, {}, {})  # three genuinely unusable turns

    await _handle(_msg("no sé"), llm, redis)
    await _handle(_msg("mmm"), llm, redis)
    reply = await _handle(_msg("ni idea"), llm, redis)

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
    llm = ScriptedLLM({}, {})

    await _handle(_msg("no sé"), llm, redis)
    await _handle(_msg("mmm"), llm, redis)

    state = await _load(redis)
    assert state.attempts["presupuesto_crucero"] == 2
    assert "presupuesto_crucero" not in state.pending


async def test_question_digression_does_not_count_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orch, "answer_question", AsyncMock(return_value="Sí. "))
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed_only_budget_left(redis)
    llm = ScriptedLLM({"question": "¿aceptan tarjeta de crédito?"})

    reply = await _handle(
        _msg("¿aceptan tarjeta?"), llm, redis
    )

    state = await _load(redis)
    assert state.attempts == {}  # a question is not a failed attempt
    assert not await is_handed_off(redis, SENDER)
    # The answer is prepended, then the slot is re-asked literally (no failed
    # attempt -> no reformulation).
    assert reply.endswith(_prompt_of(TripType.CRUISE, "presupuesto_crucero"))


async def test_out_of_order_data_does_not_count_attempt() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed_only_budget_left(redis)
    # Asked budget; the user answers with a different slot instead.
    llm = ScriptedLLM({"pasaporte_crucero": "vigente"})

    await _handle(
        _msg("sí tengo pasaporte"), llm, redis
    )

    state = await _load(redis)
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

    reply = await _handle(
        _msg("unos 2500"), llm, redis
    )

    # The value lands -> satisfied -> complete, never pending.
    assert reply == FAREWELL
    event = relay.await_args.args[1]
    assert event.reason is HandoffReason.COMPLETE
    assert event.pending == ()
    state = await _load(redis)
    assert state.pending == set()


async def test_reask_after_failure_is_reformulated_not_literal() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed_only_budget_left(redis)
    llm = ScriptedLLM({})  # one unusable turn

    reply = await _handle(
        _msg("no sé"), llm, redis
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

    reply = await _handle(
        _msg("mejor quiero hablar con una persona"),
        llm,
        redis,    )

    assert reply == orch._FAREWELL_BY_REASON[HandoffReason.HUMAN_REQUESTED]
    assert await is_handed_off(redis, SENDER)
    event = relay.await_args.args[1]
    assert event.reason is HandoffReason.HUMAN_REQUESTED
    # Carries what was captured so far.
    assert event.slots["nombre_cliente"] == "Ana"
    assert event.slots["ruta_crucero"] == "Caribe"


# --- Campaign routing (4c) -------------------------------------------------


def _referral(headline: str, body: str) -> Referral:
    return Referral(source_id="s", headline=headline, body=body, ctwa_clid="c")


async def test_first_message_prefill_routes_and_asks_first_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    understand_spy = AsyncMock()
    monkeypatch.setattr(orch, "understand_turn", understand_spy)
    redis = FakeAsyncRedis(decode_responses=True)
    routing = RoutingConfig(
        prefill_crucero="mediterraneo magico", prefill_europa=None, prefill_asia=None
    )

    reply = await _route_turn(redis, "Vi su anuncio del Mediterráneo Mágico", routing)

    # Routed to cruise and emitted the first schema question; no understanding.
    assert reply == _prompt_of(TripType.CRUISE, "nombre_cliente")
    understand_spy.assert_not_awaited()
    state = await _load(redis)
    assert state.trip_type is TripType.CRUISE
    assert state.last_bot_message == reply


async def test_first_message_referral_routes_without_text_keyword() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    msg = IncomingMessage(
        sender=SENDER,
        text="hola, me interesa",
        message_id="SM1",
        referral=_referral("Cruceros por el Caribe", "Reserva tu lugar"),
    )

    reply = await handle_message(
        msg, ScriptedLLM(), redis, _store_for(redis), _ROUTING, _CORPUS,
        _FakeHandoffRunner(),
    )

    state = await _load(redis)
    assert state.trip_type is TripType.CRUISE
    assert reply == _prompt_of(TripType.CRUISE, "nombre_cliente")


async def test_indeterminate_first_message_asks_disambiguation() -> None:
    redis = FakeAsyncRedis(decode_responses=True)

    reply = await _route_turn(redis, "hola buenas tardes")

    assert reply == orch._DISAMBIGUATION_QUESTION
    state = await _load(redis)
    assert state.trip_type is None  # still not routed
    assert state.last_bot_message == orch._DISAMBIGUATION_QUESTION


# --- Handoff wiring (the CRM orchestration runs before the phase/flag flip) ---


async def test_handoff_runs_crm_sequence_then_flips() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(redis, TripType.CRUISE, {"nombre_cliente": "Ana"})
    runner = _FakeHandoffRunner()

    # An explicit request for a human triggers the handoff this turn.
    await _handle(_msg("quiero hablar con un asesor"),
                  ScriptedLLM({"wants_human": True}), redis, runner)

    assert await is_handed_off(redis, SENDER)
    state = await _load(redis)
    assert state.phase is Phase.HANDED_OFF
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["reason"] is HandoffReason.HUMAN_REQUESTED
    assert call["phone"] == "+5215512345678"  # the whatsapp: prefix is stripped
    assert call["customer_name"] == "Ana"


async def test_handoff_crm_failure_does_not_flip_and_propagates() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(redis, TripType.CRUISE, {"nombre_cliente": "Ana"})

    with pytest.raises(KommoCrmError):
        await _handle(_msg("quiero un humano"),
                      ScriptedLLM({"wants_human": True}), redis,
                      _FailingHandoffRunner())

    # Not handed off -> the next turn can retry; phase stays COLLECTING.
    assert not await is_handed_off(redis, SENDER)
    state = await _load(redis)
    assert state.phase is Phase.COLLECTING


async def test_disambiguation_reply_routes_and_starts_flow() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    # Turn 1: indeterminate -> disambiguation.
    await _route_turn(redis, "hola")

    # Turn 2: the user clarifies; still trip_type=None so routing runs again.
    reply = await _route_turn(redis, "sería un crucero")

    state = await _load(redis)
    assert state.trip_type is TripType.CRUISE
    assert reply == _prompt_of(TripType.CRUISE, "nombre_cliente")


async def test_unclear_disambiguation_reply_reasks() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    await _route_turn(redis, "hola")

    reply = await _route_turn(redis, "pues no estoy seguro")

    assert reply == orch._DISAMBIGUATION_QUESTION
    state = await _load(redis)
    assert state.trip_type is None


async def test_already_routed_conversation_skips_routing() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(
        redis,
        TripType.CRUISE,
        {"nombre_cliente": "Ana"},
        last_asked="ruta_crucero",
    )
    # The normal flow runs (understanding extracts the route), not routing.
    llm = ScriptedLLM({"ruta_crucero": "Caribe"})

    reply = await _handle(_msg("Caribe"), llm, redis)

    assert reply == _prompt_of(TripType.CRUISE, "fechas_crucero")


# --- Answering questions (CAG, 4b) -----------------------------------------


async def test_question_mid_collection_answers_then_reasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answerer = AsyncMock(return_value="Todo incluye vuelos y hospedaje.")
    monkeypatch.setattr(orch, "answer_question", answerer)
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(
        redis,
        TripType.CRUISE,
        {"nombre_cliente": "Ana", "ruta_crucero": "Caribe"},
        last_asked="fechas_crucero",
        last_bot_message="¿En qué fechas te gustaría viajar?",
    )
    # The turn is a question (no slot data, no human request).
    llm = ScriptedLLM({"question": "¿qué incluye la cotización?"})

    reply = await _handle(
        _msg("¿qué incluye?"), llm, redis
    )

    reask = _prompt_of(TripType.CRUISE, "fechas_crucero")
    # The answer is prepended to the slot re-ask, in a single message.
    assert reply == f"Todo incluye vuelos y hospedaje.\n\n{reask}"
    # The answerer was called with the corpus, trip type, question and context.
    answerer.assert_awaited_once_with(
        llm,
        _CORPUS,
        TripType.CRUISE,
        "¿qué incluye la cotización?",
        "¿En qué fechas te gustaría viajar?",
    )
    # State is preserved and the failed-attempt counter did not move.
    state = await _load(redis)
    assert state.slots["nombre_cliente"] == "Ana"
    assert state.attempts == {}
    assert not await is_handed_off(redis, SENDER)


async def test_out_of_corpus_question_defer_text_is_included(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The answerer (mocked) defers; the orchestrator just relays its text.
    defer = "Esa la puede revisar contigo un asesor. "
    monkeypatch.setattr(orch, "answer_question", AsyncMock(return_value=defer))
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(
        redis,
        TripType.CRUISE,
        {"nombre_cliente": "Ana", "ruta_crucero": "Caribe"},
        last_asked="fechas_crucero",
    )
    llm = ScriptedLLM({"question": "¿tienen seguro contra cancelación de aerolínea?"})

    reply = await _handle(
        _msg("¿seguro?"), llm, redis
    )

    assert defer in reply


async def test_question_on_completing_turn_answers_then_farewell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orch, "relay_to_human", AsyncMock())
    monkeypatch.setattr(
        orch, "answer_question", AsyncMock(return_value="Aceptamos transferencia.")
    )
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(
        redis,
        TripType.CRUISE,
        dict(_REQUIRED_NO_BUDGET),
        last_asked="presupuesto_crucero",
        asked=_askable_optionals(TripType.CRUISE),
    )
    # Same turn supplies the last required AND asks a question.
    llm = ScriptedLLM(
        {
            "presupuesto_crucero": Budget(amount="2000-3000 USD"),
            "question": "¿cómo puedo pagar?",
        }
    )

    reply = await _handle(
        _msg("unos 2500, ¿cómo pago?"), llm, redis
    )

    assert reply == f"Aceptamos transferencia.\n\n{FAREWELL}"
    assert await is_handed_off(redis, SENDER)


async def test_wants_human_skips_answerer_even_with_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answerer = AsyncMock()
    monkeypatch.setattr(orch, "answer_question", answerer)
    monkeypatch.setattr(orch, "relay_to_human", AsyncMock())
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(
        redis,
        TripType.CRUISE,
        {"nombre_cliente": "Ana"},
        last_asked="ruta_crucero",
    )
    # Both a question and a human request: human wins, no answerer call.
    llm = ScriptedLLM({"question": "¿precio?", "wants_human": True})

    reply = await _handle(
        _msg("mejor con un humano"), llm, redis
    )

    answerer.assert_not_awaited()
    assert reply == orch._FAREWELL_BY_REASON[HandoffReason.HUMAN_REQUESTED]


async def test_turn_without_question_does_not_call_answerer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answerer = AsyncMock()
    monkeypatch.setattr(orch, "answer_question", answerer)
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(redis, TripType.CRUISE, {})  # already routed: exercises the flow
    llm = ScriptedLLM({"nombre_cliente": "Ana"})

    await _handle(_msg("soy Ana"), llm, redis)

    answerer.assert_not_awaited()


async def test_last_bot_message_stores_full_message_for_followups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A follow-up ("¿y eso?") must see the previous full message, INCLUDING the
    # answer that was prepended last turn — not just the slot re-ask.
    answerer = AsyncMock(side_effect=["Incluye vuelos y hospedaje.", "El desayuno."])
    monkeypatch.setattr(orch, "answer_question", answerer)
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(
        redis,
        TripType.CRUISE,
        {"nombre_cliente": "Ana", "ruta_crucero": "Caribe"},
        last_asked="fechas_crucero",
        last_bot_message="¿En qué fechas te gustaría viajar?",
    )
    reask = _prompt_of(TripType.CRUISE, "fechas_crucero")
    llm = ScriptedLLM({"question": "¿qué incluye?"}, {"question": "¿y eso?"})

    reply1 = await _handle(_msg("¿qué incluye?"), llm, redis)

    # After turn 1, the PERSISTED message is the FULL reply (answer + re-ask).
    after_turn_1 = await _load(redis)
    assert reply1 == f"Incluye vuelos y hospedaje.\n\n{reask}"
    assert after_turn_1.last_bot_message == reply1

    await _handle(_msg("¿y eso?"), llm, redis)

    # The 2nd answerer call received that full message as `last_bot_message`,
    # so "¿y eso?" can refer back to the previous answer.
    second_call_last_bot_message = answerer.await_args_list[1].args[4]
    assert "Incluye vuelos y hospedaje." in second_call_last_bot_message


# --- Handoff idempotency backstop (inc 5) ----------------------------------


async def _assert_silent_after_flag_loss(
    redis: FakeAsyncRedis, relay: AsyncMock
) -> None:
    """After a real handoff (turn 1), simulate a lost Redis flag and assert the
    durable backstop keeps the bot silent on the next message."""
    # The handoff really happened on turn 1.
    relay.assert_awaited_once()
    # Simulate a Redis restart that lost the fast-path flag.
    await clear_handoff(redis, SENDER)
    assert not await is_handed_off(redis, SENDER)
    relay.reset_mock()

    # Turn 2: any message. A fresh ScriptedLLM has no turns, so reaching
    # understanding would raise — proving the backstop returns before it.
    reply = await _handle(_msg("¿siguen ahí?"), ScriptedLLM(), redis)

    assert reply is None  # silent: nothing is sent
    relay.assert_not_awaited()  # NOT relayed again
    assert await is_handed_off(redis, SENDER)  # backstop restored the flag


async def test_handoff_completa_is_idempotent_after_flag_loss(
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
    # Turn 1: fill the last required slot -> `completa` handoff (driven, not seeded).
    llm = ScriptedLLM({"presupuesto_crucero": Budget(amount="2000-3000 USD")})
    farewell = await _handle(_msg("entre 2000 y 3000"), llm, redis)
    assert farewell == FAREWELL

    await _assert_silent_after_flag_loss(redis, relay)


async def test_handoff_atorado_is_idempotent_after_flag_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay = AsyncMock()
    monkeypatch.setattr(orch, "relay_to_human", relay)
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed_only_budget_left(redis)
    # Turns 1-3: three unusable answers -> 3rd gives up -> `atorado` handoff.
    llm = ScriptedLLM({}, {}, {})
    await _handle(_msg("no sé"), llm, redis)
    await _handle(_msg("mmm"), llm, redis)
    farewell = await _handle(_msg("ni idea"), llm, redis)
    assert farewell == orch._FAREWELL_BY_REASON[HandoffReason.STUCK]

    await _assert_silent_after_flag_loss(redis, relay)


async def test_handoff_pidio_humano_is_idempotent_after_flag_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay = AsyncMock()
    monkeypatch.setattr(orch, "relay_to_human", relay)
    redis = FakeAsyncRedis(decode_responses=True)
    await _seed(
        redis,
        TripType.CRUISE,
        {"nombre_cliente": "Ana"},
        last_asked="ruta_crucero",
    )
    # Turn 1: the user asks for a human -> `pidió_humano` handoff.
    llm = ScriptedLLM({"wants_human": True})
    farewell = await _handle(_msg("mejor con un humano"), llm, redis)
    assert farewell == orch._FAREWELL_BY_REASON[HandoffReason.HUMAN_REQUESTED]

    await _assert_silent_after_flag_loss(redis, relay)
