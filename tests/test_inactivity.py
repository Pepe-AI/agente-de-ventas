"""Offline tests for the inactivity timer (4th handoff reason, "No respondió").

Covers the durable deadline field's sweep query, the headless handoff, and the
sweeper (per-tick + per-sender locks). Kommo/Twilio are faked throughout.
"""

from __future__ import annotations

import time

from fakeredis import FakeAsyncRedis
from structlog.testing import capture_logs

from app.concurrency import lock
from app.concurrency.handoff import is_handed_off
from app.concurrency.inactivity import _SWEEP_LOCK_ID, _sweep_one, sweep_once
from app.domain.handoff_orchestration import HandoffResult
from app.domain.inactivity import run_inactivity_handoff
from app.domain.models import HandoffReason, IncomingMessage
from app.domain.orchestrator import INACTIVITY_DEADLINE_S, handle_message
from app.domain.state import ConversationState, Phase
from app.routing.campaign import RoutingConfig
from app.understanding.schemas import TripType
from tests.fakes import InMemoryStateStore

SENDER = "whatsapp:+5215512345678"
PHONE = "+5215512345678"

_ROUTING = RoutingConfig(prefill_crucero=None, prefill_europa=None, prefill_asia=None)
_CORPUS = "CORPUS DE PRUEBA"


def _state(
    *,
    deadline: float | None,
    phase: Phase = Phase.COLLECTING,
    name: str = "Ana",
) -> ConversationState:
    return ConversationState(
        trip_type=TripType.CRUISE,
        slots={"nombre_cliente": name},
        phase=phase,
        inactivity_deadline=deadline,
    )


# --- find_expired_deadlines -------------------------------------------------


async def test_find_expired_returns_only_past_collecting() -> None:
    store = InMemoryStateStore()
    now = 1_000_000.0
    await store.save("whatsapp:+1", _state(deadline=now - 1))  # expired + collecting
    await store.save("whatsapp:+2", _state(deadline=now + 100))  # future
    await store.save("whatsapp:+3", _state(deadline=None))  # never armed
    await store.save(
        "whatsapp:+4", _state(deadline=now - 1, phase=Phase.HANDED_OFF)
    )  # already handed off

    expired = await store.find_expired_deadlines(now)

    assert {cid for cid, _ in expired} == {"whatsapp:+1"}


async def test_find_expired_includes_deadline_exactly_now() -> None:
    store = InMemoryStateStore()
    now = 1_000_000.0
    await store.save(SENDER, _state(deadline=now))  # <= now counts as due

    expired = await store.find_expired_deadlines(now)

    assert [cid for cid, _ in expired] == [SENDER]


# --- arming / clearing in the orchestrator ----------------------------------


class _Runner:
    """Handoff runner that records each run's kwargs; can fail for given phones."""

    def __init__(self, fail_phones: set[str] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._fail_phones = fail_phones or set()

    async def run(self, **kwargs: object) -> HandoffResult:
        self.calls.append(kwargs)
        if kwargs.get("phone") in self._fail_phones:
            raise RuntimeError("crm boom")
        return HandoffResult(lead_id=1, contact_id=1)


class _FakeConnector:
    """Records create_chat/link; returns a fixed chat_id (mirrors ChatConnector)."""

    def __init__(self, chat_id: str = "chat-uuid") -> None:
        self.chat_id = chat_id
        self.created: list[tuple[str, object]] = []
        self.links: list[tuple[int, str]] = []

    async def create_chat(self, conversation_id: str, user: object) -> str:
        self.created.append((conversation_id, user))
        return self.chat_id

    async def link(self, contact_id: int, chat_id: str) -> None:
        self.links.append((contact_id, chat_id))


class _ScriptedLLM:
    """Returns a preset extraction per turn (same shape as the orchestrator tests)."""

    def __init__(self, *turns: dict[str, object]) -> None:
        self._turns = list(turns)
        self.calls = 0

    async def complete_structured(self, prompt: str, schema: type) -> object:
        preset = self._turns[self.calls]
        self.calls += 1
        return schema(**preset)


async def _drive(
    store: InMemoryStateStore, llm: _ScriptedLLM, runner: _Runner | None = None
) -> str | None:
    return await handle_message(
        IncomingMessage(sender=SENDER, text="hola", message_id="SM1"),
        llm,
        FakeAsyncRedis(decode_responses=True),
        store,
        _ROUTING,
        _CORPUS,
        runner or _Runner(),
        None,  # chat_connector degraded -> chat skipped
    )


async def test_deadline_armed_when_name_present_on_send() -> None:
    store = InMemoryStateStore()
    await store.save(
        SENDER,
        ConversationState(trip_type=TripType.CRUISE, slots={"nombre_cliente": "Ana"}),
    )

    before = time.time()
    await _drive(store, _ScriptedLLM({}))  # empty extraction -> asks next slot
    after = time.time()

    state = await store.load(SENDER)
    assert state is not None and state.inactivity_deadline is not None
    assert before + INACTIVITY_DEADLINE_S <= state.inactivity_deadline
    assert state.inactivity_deadline <= after + INACTIVITY_DEADLINE_S + 1


async def test_deadline_not_armed_before_name_captured() -> None:
    store = InMemoryStateStore()
    await store.save(SENDER, ConversationState(trip_type=TripType.CRUISE, slots={}))

    await _drive(store, _ScriptedLLM({}))  # asks for the name; name not captured yet

    state = await store.load(SENDER)
    assert state is not None and state.inactivity_deadline is None


async def test_deadline_repushed_each_turn() -> None:
    store = InMemoryStateStore()
    await store.save(
        SENDER,
        ConversationState(
            trip_type=TripType.CRUISE,
            slots={"nombre_cliente": "Ana"},
            inactivity_deadline=5000.0,  # an old, stale deadline
        ),
    )

    await _drive(store, _ScriptedLLM({}))

    state = await store.load(SENDER)
    assert state is not None and state.inactivity_deadline is not None
    assert state.inactivity_deadline > 1_000_000  # moved far forward from 5000


async def test_handoff_clears_deadline() -> None:
    store = InMemoryStateStore()
    await store.save(
        SENDER,
        ConversationState(
            trip_type=TripType.CRUISE,
            slots={"nombre_cliente": "Ana"},
            inactivity_deadline=5000.0,
        ),
    )

    # wants_human -> immediate handoff -> the deadline must be cleared.
    await _drive(store, _ScriptedLLM({"wants_human": True}), _Runner())

    state = await store.load(SENDER)
    assert state is not None
    assert state.phase is Phase.HANDED_OFF
    assert state.inactivity_deadline is None


# --- headless handoff -------------------------------------------------------


async def test_headless_transfers_to_no_response_connects_chat_and_clears() -> None:
    store = InMemoryStateStore()
    state = ConversationState(
        trip_type=TripType.CRUISE,
        slots={"nombre_cliente": "Ana"},
        inactivity_deadline=5000.0,
    )
    await store.save(SENDER, state)
    redis = FakeAsyncRedis(decode_responses=True)
    runner = _Runner()
    connector = _FakeConnector()

    await run_inactivity_handoff(
        SENDER, state, redis=redis, store=store,
        handoff_runner=runner, chat_connector=connector,
    )

    assert runner.calls[0]["reason"] is HandoffReason.NO_RESPONSE
    assert runner.calls[0]["phone"] == PHONE
    assert connector.created == [(PHONE, connector.created[0][1])]  # chat created
    assert connector.links == [(1, "chat-uuid")]  # linked to the contact
    saved = await store.load(SENDER)
    assert saved is not None
    assert saved.phase is Phase.HANDED_OFF
    assert saved.inactivity_deadline is None
    assert saved.chat_id == "chat-uuid"
    assert await is_handed_off(redis, SENDER)


async def test_headless_degraded_channel_skips_chat_but_still_flips() -> None:
    store = InMemoryStateStore()
    state = ConversationState(
        trip_type=TripType.CRUISE,
        slots={"nombre_cliente": "Ana"},
        inactivity_deadline=5000.0,
    )
    await store.save(SENDER, state)
    redis = FakeAsyncRedis(decode_responses=True)
    runner = _Runner()

    await run_inactivity_handoff(
        SENDER, state, redis=redis, store=store,
        handoff_runner=runner, chat_connector=None,  # degraded
    )

    saved = await store.load(SENDER)
    assert saved is not None
    assert saved.phase is Phase.HANDED_OFF
    assert saved.inactivity_deadline is None
    assert saved.chat_id is None  # no chat connected
    assert await is_handed_off(redis, SENDER)




# --- the sweeper (per-tick + per-sender locks) ------------------------------

SENDER2 = "whatsapp:+5215599999999"
PHONE2 = "+5215599999999"


async def test_sweep_hands_off_an_expired_conversation() -> None:
    store = InMemoryStateStore()
    now = 2_000_000.0
    await store.save(SENDER, _state(deadline=now - 1))
    redis = FakeAsyncRedis(decode_responses=True)
    runner = _Runner()

    await sweep_once(
        now, redis=redis, store=store, handoff_runner=runner,
        chat_connector=_FakeConnector(),
    )

    assert runner.calls and runner.calls[0]["reason"] is HandoffReason.NO_RESPONSE
    saved = await store.load(SENDER)
    assert saved is not None and saved.phase is Phase.HANDED_OFF
    assert await is_handed_off(redis, SENDER)


async def test_sweep_is_noop_when_nothing_expired() -> None:
    store = InMemoryStateStore()
    now = 2_000_000.0
    await store.save(SENDER, _state(deadline=now + 1000))  # future
    redis = FakeAsyncRedis(decode_responses=True)
    runner = _Runner()

    await sweep_once(
        now, redis=redis, store=store, handoff_runner=runner,
        chat_connector=_FakeConnector(),
    )

    assert runner.calls == []
    saved = await store.load(SENDER)
    assert saved is not None and saved.phase is Phase.COLLECTING


async def test_sweep_isolates_a_per_conversation_failure() -> None:
    store = InMemoryStateStore()
    now = 2_000_000.0
    await store.save(SENDER, _state(deadline=now - 1))  # PHONE -> will fail
    await store.save(SENDER2, _state(deadline=now - 1))  # PHONE2 -> succeeds
    redis = FakeAsyncRedis(decode_responses=True)
    runner = _Runner(fail_phones={PHONE})

    with capture_logs() as logs:
        await sweep_once(
            now, redis=redis, store=store, handoff_runner=runner,
            chat_connector=_FakeConnector(),
        )

    assert any(e["event"] == "inactivity_handoff_failed" for e in logs)
    first = await store.load(SENDER)
    second = await store.load(SENDER2)
    assert first is not None and first.phase is Phase.COLLECTING  # failed -> no flip
    assert second is not None and second.phase is Phase.HANDED_OFF  # loop continued


async def test_sweep_skips_when_tick_lock_is_held() -> None:
    store = InMemoryStateStore()
    now = 2_000_000.0
    await store.save(SENDER, _state(deadline=now - 1))
    redis = FakeAsyncRedis(decode_responses=True)
    await lock.acquire(redis, _SWEEP_LOCK_ID, "someone-else", 100)  # tick lock taken
    runner = _Runner()

    await sweep_once(
        now, redis=redis, store=store, handoff_runner=runner,
        chat_connector=_FakeConnector(),
    )

    assert runner.calls == []  # another sweeper owns this tick
    saved = await store.load(SENDER)
    assert saved is not None and saved.phase is Phase.COLLECTING


async def test_sweep_skips_a_sender_whose_lock_is_held_by_a_flush() -> None:
    store = InMemoryStateStore()
    now = 2_000_000.0
    await store.save(SENDER, _state(deadline=now - 1))
    redis = FakeAsyncRedis(decode_responses=True)
    await lock.acquire(redis, SENDER, "flush-token", 100)  # a flush holds the sender
    runner = _Runner()

    await sweep_once(
        now, redis=redis, store=store, handoff_runner=runner,
        chat_connector=_FakeConnector(),
    )

    assert runner.calls == []  # the message wins; the sweeper backs off
    saved = await store.load(SENDER)
    assert saved is not None and saved.phase is Phase.COLLECTING


async def test_sweep_one_rechecks_and_skips_if_deadline_repushed() -> None:
    store = InMemoryStateStore()
    now = 2_000_000.0
    # The candidate's stored deadline is now in the FUTURE (a flush re-pushed it
    # between the query and the lock): the reload-under-lock must skip it.
    await store.save(SENDER, _state(deadline=now + 9999))
    redis = FakeAsyncRedis(decode_responses=True)
    runner = _Runner()

    await _sweep_one(
        SENDER, now, redis=redis, store=store, handoff_runner=runner,
        chat_connector=_FakeConnector(),
    )

    assert runner.calls == []
    saved = await store.load(SENDER)
    assert saved is not None and saved.phase is Phase.COLLECTING
