"""Offline tests for the B3 inbound drain (advisor reply -> customer via Twilio).

The queue is a real FakeAsyncRedis (exercises FIFO lpop/rpush semantics); the
channel is a minimal fake that records sends or raises to simulate a Twilio
failure. The drain loops lpop until empty and try/excepts each send.
"""

from __future__ import annotations

import json

from fakeredis import FakeAsyncRedis
from structlog.testing import capture_logs

from app.concurrency.keys import KeyPrefix, make_key
from app.crm.kommo_inbound import _drain

SCOPE_ID = "scope-1"
PHONE = "+5215512345678"
QUEUE = make_key(KeyPrefix.KOMMO_INBOUND, SCOPE_ID)


class _FakeChannel:
    """Minimal Channel double: records sends, optionally raises for some texts."""

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.sent: list[tuple[str, str]] = []
        self._fail_on = fail_on or set()

    async def send(self, to: str, text: str) -> None:
        if text in self._fail_on:
            raise RuntimeError("twilio boom")  # stand-in for TwilioRestException
        self.sent.append((to, text))


def _v1(
    text: str = "Hola, soy tu asesor de Top Viajes",
    *,
    conversation_id: str = PHONE,
    type_: str = "text",
) -> str:
    """Build a REAL-shape v1 inbound webhook payload as a JSON string."""
    return json.dumps(
        {
            "receiver": f"wa-{conversation_id}",
            "conversation_id": conversation_id,
            "msec_timestamp": 1700000000000,
            "type": type_,
            "text": text,
            "markup": None,
            "tag": None,
            "media": "",
            "thumbnail": "",
            "file_name": "",
            "file_size": 0,
            "media_group_id": "",
        }
    )


async def _redis_with(*payloads: str) -> FakeAsyncRedis:
    redis = FakeAsyncRedis(decode_responses=True)
    for payload in payloads:
        await redis.rpush(QUEUE, payload)
    return redis


async def test_drain_relays_valid_text() -> None:
    redis = await _redis_with(_v1("¿seguimos por aquí?"))
    channel = _FakeChannel()

    await _drain(redis, channel, SCOPE_ID)

    assert channel.sent == [(f"whatsapp:{PHONE}", "¿seguimos por aquí?")]
    assert await redis.lpop(QUEUE) is None  # queue fully drained


async def test_drain_skips_non_text() -> None:
    redis = await _redis_with(_v1("", type_="picture"))
    channel = _FakeChannel()

    with capture_logs() as logs:
        await _drain(redis, channel, SCOPE_ID)

    assert channel.sent == []
    assert any(e["event"] == "drain_skipped_non_text" for e in logs)


async def test_drain_skips_empty_text() -> None:
    redis = await _redis_with(_v1("   "))  # text present but blank
    channel = _FakeChannel()

    with capture_logs() as logs:
        await _drain(redis, channel, SCOPE_ID)

    assert channel.sent == []
    assert any(e["event"] == "drain_skipped_empty_text" for e in logs)


async def test_drain_skips_malformed_non_json() -> None:
    redis = await _redis_with("{not valid json")
    channel = _FakeChannel()

    with capture_logs() as logs:
        await _drain(redis, channel, SCOPE_ID)

    assert channel.sent == []
    assert any(e["event"] == "drain_parse_failed" for e in logs)


async def test_drain_skips_missing_conversation_id() -> None:
    redis = await _redis_with(json.dumps({"type": "text", "text": "hola"}))
    channel = _FakeChannel()

    with capture_logs() as logs:
        await _drain(redis, channel, SCOPE_ID)

    assert channel.sent == []
    assert any(e["event"] == "drain_parse_failed" for e in logs)


async def test_drain_empty_queue_is_noop() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    channel = _FakeChannel()

    await _drain(redis, channel, SCOPE_ID)  # must not raise

    assert channel.sent == []


async def test_drain_catches_send_failure_and_does_not_propagate() -> None:
    redis = await _redis_with(_v1("esto va a fallar"))
    channel = _FakeChannel(fail_on={"esto va a fallar"})

    with capture_logs() as logs:
        await _drain(redis, channel, SCOPE_ID)  # must NOT raise

    assert channel.sent == []
    assert any(e["event"] == "drain_failed" for e in logs)
    assert await redis.lpop(QUEUE) is None  # popped despite the failure


async def test_drain_loops_until_empty_fifo() -> None:
    redis = await _redis_with(_v1("primero"), _v1("segundo"))
    channel = _FakeChannel()

    await _drain(redis, channel, SCOPE_ID)  # ONE drain clears BOTH

    assert channel.sent == [
        (f"whatsapp:{PHONE}", "primero"),
        (f"whatsapp:{PHONE}", "segundo"),
    ]
    assert await redis.lpop(QUEUE) is None


async def test_drain_continues_after_a_send_failure() -> None:
    redis = await _redis_with(_v1("primero"), _v1("segundo"))
    channel = _FakeChannel(fail_on={"primero"})

    with capture_logs() as logs:
        await _drain(redis, channel, SCOPE_ID)

    # The 1st send fails but the loop continues and delivers the 2nd.
    assert channel.sent == [(f"whatsapp:{PHONE}", "segundo")]
    assert any(e["event"] == "drain_failed" for e in logs)
    assert await redis.lpop(QUEUE) is None
