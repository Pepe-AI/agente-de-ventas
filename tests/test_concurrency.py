"""Concurrency-layer tests with fakeredis (dedupe, debounce, flood, lock).

handle_message now runs the understanding engine, so these tests inject a
FakeLLM and assert on send count + the text that reached the engine, not on a
fixed echo string.
"""

from __future__ import annotations

import asyncio

import pytest
from fakeredis import FakeAsyncRedis

from app.concurrency import buffer, debounce, dedup, rate_limit
from app.concurrency.config import ConcurrencyConfig
from app.concurrency.flush import flush
from app.understanding.schemas import DummyReservation

SENDER = "whatsapp:+5215512345678"


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
        self, prompt: str, schema: type[DummyReservation]
    ) -> DummyReservation:
        self.prompts.append(prompt)
        return DummyReservation()


@pytest.fixture
def redis_client() -> FakeAsyncRedis:
    return FakeAsyncRedis(decode_responses=True)


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def config() -> ConcurrencyConfig:
    # Small windows/thresholds; flush() itself does not sleep.
    return ConcurrencyConfig(
        debounce_window_s=0.0,
        dedup_ttl_s=3600,
        lock_ttl_s=30,
        rate_window_s=10,
        rate_threshold=3,
        block_cooldown_s=600,
        buffer_max=5,
    )


async def test_duplicate_message_processed_once(
    redis_client: FakeAsyncRedis, llm: FakeLLM, config: ConcurrencyConfig
) -> None:
    sid = "SM1"

    assert not await dedup.is_duplicate(redis_client, sid, config.dedup_ttl_s)
    await buffer.append(redis_client, SENDER, "hi")
    await debounce.set_token(redis_client, SENDER, sid)

    # The retry is detected as duplicate and is not buffered again.
    assert await dedup.is_duplicate(redis_client, sid, config.dedup_ttl_s)

    channel = FakeChannel()
    await flush(redis_client, channel, llm, SENDER, sid, config)

    # Processed exactly once, over the single buffered message.
    assert len(channel.sent) == 1
    assert channel.sent[0][0] == SENDER
    assert len(llm.prompts) == 1
    assert "hi" in llm.prompts[0]


async def test_debounce_combines_messages(
    redis_client: FakeAsyncRedis, llm: FakeLLM, config: ConcurrencyConfig
) -> None:
    messages = [("SM1", "a"), ("SM2", "b"), ("SM3", "c")]
    for sid, text in messages:
        assert not await dedup.is_duplicate(redis_client, sid, config.dedup_ttl_s)
        await buffer.append(redis_client, SENDER, text)
        await debounce.set_token(redis_client, SENDER, sid)

    channel = FakeChannel()
    # Only the latest token (SM3) wins; the others abort.
    await asyncio.gather(
        *(flush(redis_client, channel, llm, SENDER, sid, config) for sid, _ in messages)
    )

    # One flush, fed the combined text of all three messages.
    assert len(channel.sent) == 1
    assert len(llm.prompts) == 1
    assert "a\nb\nc" in llm.prompts[0]


async def test_flood_blocks_and_discards(
    redis_client: FakeAsyncRedis, llm: FakeLLM, config: ConcurrencyConfig
) -> None:
    hits = 0
    for _ in range(config.rate_threshold + 1):
        hits = await rate_limit.register_hit(redis_client, SENDER, config.rate_window_s)
    assert hits > config.rate_threshold

    await rate_limit.block(redis_client, SENDER, config.block_cooldown_s)
    assert await rate_limit.is_blocked(redis_client, SENDER)

    # A flush of pre-block messages must not reply once the sender is blocked.
    await buffer.append(redis_client, SENDER, "x")
    await debounce.set_token(redis_client, SENDER, "SMx")

    channel = FakeChannel()
    await flush(redis_client, channel, llm, SENDER, "SMx", config)

    assert channel.sent == []
    assert llm.prompts == []  # the engine was never invoked


async def test_lock_prevents_double_processing(
    redis_client: FakeAsyncRedis, llm: FakeLLM, config: ConcurrencyConfig
) -> None:
    await buffer.append(redis_client, SENDER, "x")
    await debounce.set_token(redis_client, SENDER, "SM1")

    channel = FakeChannel()
    # Two concurrent flushes for the same sender; the lock admits only one.
    await asyncio.gather(
        flush(redis_client, channel, llm, SENDER, "SM1", config),
        flush(redis_client, channel, llm, SENDER, "SM1", config),
    )

    assert len(channel.sent) == 1
    assert channel.sent[0][0] == SENDER
