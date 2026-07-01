"""Concurrency-layer tests with fakeredis (dedupe, debounce, flood, lock).

handle_message now runs the orchestrator, so these tests inject a FakeLLM and
assert on send count + the text that reached the engine, not on a fixed echo
string.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator

import pytest
from fakeredis import FakeAsyncRedis
from pydantic import BaseModel
from structlog.testing import capture_logs

from app.concurrency import buffer, dedup, lock, rate_limit
from app.concurrency import flush as flush_mod
from app.concurrency.config import ConcurrencyConfig
from app.concurrency.flush import _GENERIC_APOLOGY, _LLM_FALLBACK, _safe_send
from app.domain.state import ConversationState
from app.llm.base import LLMUnavailableError
from app.routing.campaign import RoutingConfig
from app.understanding.schemas import TripType
from tests.fakes import InMemoryStateStore

SENDER = "whatsapp:+5215512345678"
ROUTING = RoutingConfig(prefill_crucero=None, prefill_europa=None, prefill_asia=None)
CORPUS = "CORPUS DE PRUEBA"


async def _route_to_cruise(store: InMemoryStateStore) -> None:
    """Seed an already-routed conversation so the flush exercises the engine
    (not the routing pre-phase)."""
    await store.save(SENDER, ConversationState(trip_type=TripType.CRUISE))


class _FakeHandoffRunner:
    """No-op handoff runner (these flushes never reach the handoff step)."""

    async def run(self, **kwargs: object) -> int:
        return 1


_HANDOFF_RUNNER = _FakeHandoffRunner()


async def _flush(
    redis: FakeAsyncRedis,
    channel: FakeChannel,
    llm: FakeLLM,
    store: InMemoryStateStore,
    sid: str,
    config: ConcurrencyConfig,
) -> None:
    """Run a flush for SENDER with the test's fixed routing config + corpus."""
    await flush_mod.flush(
        redis, channel, llm, store, ROUTING, CORPUS, _HANDOFF_RUNNER,
        None, SENDER, sid, config,
    )


class FakeChannel:
    """Captures the messages a flush sends."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def send(self, to: str, text: str) -> None:
        self.sent.append((to, text))


class FakeLLM:
    """Records prompts; returns an empty understanding (all slots null)."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete_structured(
        self, prompt: str, schema: type[BaseModel]
    ) -> BaseModel:
        self.prompts.append(prompt)
        return schema()


@pytest.fixture
def redis_client() -> FakeAsyncRedis:
    return FakeAsyncRedis(decode_responses=True)


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def store() -> InMemoryStateStore:
    return InMemoryStateStore()


@pytest.fixture
def config() -> ConcurrencyConfig:
    # Small windows/thresholds; flush() itself does not sleep.
    return ConcurrencyConfig(
        debounce_window_s=0.0,
        max_buffer_wait_s=12.0,
        dedup_ttl_s=3600,
        lock_ttl_s=30,
        rate_window_s=10,
        rate_threshold=3,
        block_cooldown_s=600,
        buffer_max=5,
        inactivity_deadline_s=7200.0,
    )


@pytest.fixture(autouse=True)
def _clear_debounce_state() -> Iterator[None]:
    """Isolate the in-memory per-sender debounce dicts between tests."""
    flush_mod._pending_flushes.clear()
    flush_mod._burst_anchors.clear()
    yield
    flush_mod._pending_flushes.clear()
    flush_mod._burst_anchors.clear()


async def test_duplicate_message_processed_once(
    redis_client: FakeAsyncRedis,
    llm: FakeLLM,
    store: InMemoryStateStore,
    config: ConcurrencyConfig,
) -> None:
    sid = "SM1"
    await _route_to_cruise(store)

    assert not await dedup.is_duplicate(redis_client, sid, config.dedup_ttl_s)
    await buffer.append(redis_client, SENDER, "hi")

    # The retry is detected as duplicate and is not buffered again.
    assert await dedup.is_duplicate(redis_client, sid, config.dedup_ttl_s)

    channel = FakeChannel()
    await _flush(redis_client, channel, llm, store, sid, config)

    # Processed exactly once, over the single buffered message.
    assert len(channel.sent) == 1
    assert channel.sent[0][0] == SENDER
    assert len(llm.prompts) == 1
    assert "hi" in llm.prompts[0]


async def test_debounce_combines_messages(
    redis_client: FakeAsyncRedis,
    llm: FakeLLM,
    store: InMemoryStateStore,
    config: ConcurrencyConfig,
) -> None:
    await _route_to_cruise(store)
    messages = [("SM1", "a"), ("SM2", "b"), ("SM3", "c")]
    for sid, text in messages:
        assert not await dedup.is_duplicate(redis_client, sid, config.dedup_ttl_s)
        await buffer.append(redis_client, SENDER, text)

    channel = FakeChannel()
    # Several flushes race (as if one was scheduled per message): the lock admits
    # exactly one, which drains the WHOLE buffer at once; the rest find the lock
    # held / the buffer already empty and abort. Coalescing is by lock+drain now,
    # not by a token gate.
    await asyncio.gather(
        *(
            _flush(redis_client, channel, llm, store, sid, config)
            for sid, _ in messages
        )
    )

    # One reply, fed the combined text of all three buffered messages.
    assert len(channel.sent) == 1
    assert len(llm.prompts) == 1
    assert "a\nb\nc" in llm.prompts[0]


async def test_flood_blocks_and_discards(
    redis_client: FakeAsyncRedis,
    llm: FakeLLM,
    store: InMemoryStateStore,
    config: ConcurrencyConfig,
) -> None:
    hits = 0
    for _ in range(config.rate_threshold + 1):
        hits = await rate_limit.register_hit(redis_client, SENDER, config.rate_window_s)
    assert hits > config.rate_threshold

    await rate_limit.block(redis_client, SENDER, config.block_cooldown_s)
    assert await rate_limit.is_blocked(redis_client, SENDER)

    # A flush of pre-block messages must not reply once the sender is blocked.
    await buffer.append(redis_client, SENDER, "x")

    channel = FakeChannel()
    await _flush(redis_client, channel, llm, store, "SMx", config)

    assert channel.sent == []
    assert llm.prompts == []  # the engine was never invoked


async def test_lock_prevents_double_processing(
    redis_client: FakeAsyncRedis,
    llm: FakeLLM,
    store: InMemoryStateStore,
    config: ConcurrencyConfig,
) -> None:
    await buffer.append(redis_client, SENDER, "x")

    channel = FakeChannel()
    # Two concurrent flushes for the same sender; the lock admits only one.
    await asyncio.gather(
        _flush(redis_client, channel, llm, store, "SM1", config),
        _flush(redis_client, channel, llm, store, "SM1", config),
    )

    assert len(channel.sent) == 1
    assert channel.sent[0][0] == SENDER


# --- LLM-failure resilience -------------------------------------------------


class RaisingLLM:
    """Raises a fixed exception on every call (simulates a failing LLM)."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def complete_structured(
        self, prompt: str, schema: type[BaseModel]
    ) -> BaseModel:
        raise self._exc


class RaisingChannel(FakeChannel):
    """Channel whose send always fails (simulates Twilio being down)."""

    async def send(self, to: str, text: str) -> None:
        raise RuntimeError("transport down")


async def _buffer_one(redis: FakeAsyncRedis) -> None:
    await buffer.append(redis, SENDER, "hola")


async def test_flush_llm_unavailable_sends_fallback_and_preserves_state(
    redis_client: FakeAsyncRedis, store: InMemoryStateStore, config: ConcurrencyConfig
) -> None:
    seeded = ConversationState(
        trip_type=TripType.CRUISE, slots={"nombre_cliente": "Ana"}
    )
    await store.save(SENDER, seeded)  # routed, so understand_turn (the LLM) runs
    await _buffer_one(redis_client)
    channel = FakeChannel()
    llm = RaisingLLM(LLMUnavailableError("retries exhausted"))

    # The background task must NOT re-raise (no "exception never retrieved").
    await _flush(redis_client, channel, llm, store, "SM1", config)

    # Graceful fallback sent instead of silence.
    assert channel.sent == [(SENDER, _LLM_FALLBACK)]
    # State is not mutated (a resend will work).
    assert await store.load(SENDER) == seeded
    # Buffer drained and lock released (re-acquirable).
    assert await buffer.drain(redis_client, SENDER) == []
    assert await lock.acquire(redis_client, SENDER, "other-token", config.lock_ttl_s)


async def test_flush_unexpected_error_sends_apology(
    redis_client: FakeAsyncRedis, store: InMemoryStateStore, config: ConcurrencyConfig
) -> None:
    await _route_to_cruise(store)
    await _buffer_one(redis_client)
    channel = FakeChannel()
    llm = RaisingLLM(RuntimeError("unexpected bug"))  # not an LLM-unavailable signal

    await _flush(redis_client, channel, llm, store, "SM1", config)

    assert channel.sent == [(SENDER, _GENERIC_APOLOGY)]
    assert await lock.acquire(redis_client, SENDER, "other-token", config.lock_ttl_s)


async def test_flush_does_not_crash_when_fallback_send_fails(
    redis_client: FakeAsyncRedis, store: InMemoryStateStore, config: ConcurrencyConfig
) -> None:
    await _route_to_cruise(store)
    await _buffer_one(redis_client)
    channel = RaisingChannel()
    llm = RaisingLLM(LLMUnavailableError("down"))

    # Even the fallback send fails -> the task still must not re-raise.
    await _flush(redis_client, channel, llm, store, "SM1", config)

    # Cleanup still happened: the lock is released.
    assert await lock.acquire(redis_client, SENDER, "other-token", config.lock_ttl_s)


async def test_safe_send_logs_transport_failure_and_does_not_propagate() -> None:
    # "safe" means: never crashes the task AND leaves a trace (a Twilio outage
    # must not be invisible).
    channel = RaisingChannel()

    with capture_logs() as logs:
        await _safe_send(channel, SENDER, "hola")  # must not raise

    events = [e for e in logs if e["event"] == "flush_send_failed"]
    assert len(events) == 1
    assert events[0]["log_level"] == "error"


# --- Real debounce: one per-sender timer (cancel+reschedule) with a max cap ---


def _debounce_config(debounce: float, cap: float) -> ConcurrencyConfig:
    return ConcurrencyConfig(
        debounce_window_s=debounce,
        max_buffer_wait_s=cap,
        dedup_ttl_s=3600,
        lock_ttl_s=30,
        rate_window_s=10,
        rate_threshold=3,
        block_cooldown_s=600,
        buffer_max=5,
        inactivity_deadline_s=7200.0,
    )


def _schedule(
    redis: FakeAsyncRedis,
    channel: FakeChannel,
    llm: FakeLLM,
    store: InMemoryStateStore,
    sid: str,
    config: ConcurrencyConfig,
) -> None:
    flush_mod.schedule_flush(
        redis, channel, llm, store, ROUTING, CORPUS, _HANDOFF_RUNNER,
        None, SENDER, sid, config,
    )


async def test_burst_within_window_fires_once_and_cancels_prior(
    monkeypatch: pytest.MonkeyPatch,
    redis_client: FakeAsyncRedis,
    llm: FakeLLM,
    store: InMemoryStateStore,
) -> None:
    # Bug #1 regression: a second message within the window CANCELS the first
    # timer and reschedules, so exactly ONE flush fires over the combined buffer
    # (the old per-message model left >=2 surviving partial flushes).
    await _route_to_cruise(store)
    config = _debounce_config(debounce=3.0, cap=12.0)
    channel = FakeChannel()

    release = asyncio.Event()

    async def fake_sleep(_delay: float) -> None:
        await release.wait()  # the test controls when the timer "fires"

    monkeypatch.setattr(flush_mod, "_sleep", fake_sleep)
    monkeypatch.setattr(flush_mod, "_now", lambda: 0.0)  # both arrive within window

    await buffer.append(redis_client, SENDER, "hola")
    _schedule(redis_client, channel, llm, store, "SM1", config)
    task1 = flush_mod._pending_flushes[SENDER]

    await buffer.append(redis_client, SENDER, "mundo")
    _schedule(redis_client, channel, llm, store, "SM2", config)
    task2 = flush_mod._pending_flushes[SENDER]

    # The first timer was superseded by the second message.
    with contextlib.suppress(asyncio.CancelledError):
        await task1
    assert task1.cancelled()
    assert task1 is not task2

    release.set()  # fire the single surviving timer
    await task2

    assert len(channel.sent) == 1  # ONE reply, not two
    assert len(llm.prompts) == 1
    assert "hola\nmundo" in llm.prompts[0]  # over BOTH messages
    # Burst tracking is cleared once the flush fired.
    assert SENDER not in flush_mod._pending_flushes
    assert SENDER not in flush_mod._burst_anchors


async def test_cap_clamps_delay_for_a_nonstop_typer(
    monkeypatch: pytest.MonkeyPatch,
    redis_client: FakeAsyncRedis,
    llm: FakeLLM,
    store: InMemoryStateStore,
) -> None:
    # Cap regression: messages keep arriving with gaps < debounce; the cap, anchored
    # at the first message, clamps the scheduled delay so the flush fires at
    # anchor + cap rather than waiting forever.
    config = _debounce_config(debounce=3.0, cap=12.0)
    channel = FakeChannel()

    release = asyncio.Event()
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)
        await release.wait()

    clock = {"t": 0.0}
    monkeypatch.setattr(flush_mod, "_sleep", fake_sleep)
    monkeypatch.setattr(flush_mod, "_now", lambda: clock["t"])

    clock["t"] = 0.0  # first message anchors the cap at t=0
    _schedule(redis_client, channel, llm, store, "SM1", config)
    await asyncio.sleep(0)  # let the timer task run to _sleep
    assert delays[-1] == 3.0  # debounce binds: min(3, 12 - 0)

    clock["t"] = 10.0  # still typing; 12 - 10 = 2 < debounce -> cap binds
    _schedule(redis_client, channel, llm, store, "SM2", config)
    await asyncio.sleep(0)
    assert delays[-1] == 2.0

    clock["t"] = 12.0  # at the cap deadline -> immediate flush
    _schedule(redis_client, channel, llm, store, "SM3", config)
    await asyncio.sleep(0)
    assert delays[-1] == 0.0

    # Cleanup: cancel the surviving (never-released) timer.
    pending = flush_mod._pending_flushes.get(SENDER)
    if pending is not None:
        pending.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pending


async def test_concurrent_drains_do_not_duplicate(
    redis_client: FakeAsyncRedis,
) -> None:
    # CAPA 2: the drain is one MULTI/EXEC transaction, so two concurrent drains
    # never both see the items -- one gets them all, the other an empty list (no
    # duplication, no loss). NOTE: true cross-client atomicity is a Redis server
    # guarantee; fakeredis approximates it by running each pipeline.execute()
    # without interleaving, which is enough to exercise the invariant here.
    for text in ("a", "b", "c"):
        await buffer.append(redis_client, SENDER, text)

    r1, r2 = await asyncio.gather(
        buffer.drain(redis_client, SENDER),
        buffer.drain(redis_client, SENDER),
    )

    assert sorted(r1 + r2) == ["a", "b", "c"]  # every item exactly once
    assert [] in (r1, r2)  # one drain got nothing
    assert await buffer.drain(redis_client, SENDER) == []  # buffer now empty
