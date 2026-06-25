"""Tests for the human-handoff flag and endpoint short-circuit."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from unittest.mock import AsyncMock, Mock

import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient
from pydantic import BaseModel

import app.main as main_module
from app.channels.twilio import InvalidPayloadError, TwilioField
from app.concurrency import handoff
from app.concurrency.config import ConcurrencyConfig
from app.domain.models import IncomingMessage
from app.main import (
    WEBHOOK_PATH,
    app,
    get_channel,
    get_chat_connector,
    get_concurrency_config,
    get_corpus,
    get_handoff_runner,
    get_llm,
    get_redis,
    get_routing_config,
    get_store,
)
from app.routing.campaign import RoutingConfig
from tests.fakes import InMemoryStateStore

SENDER = "whatsapp:+5215512345678"
VALID_FORM = {"From": SENDER, "Body": "hola", "MessageSid": "SM123"}

TEST_CONFIG = ConcurrencyConfig(
    debounce_window_s=60.0,
    dedup_ttl_s=3600,
    lock_ttl_s=30,
    rate_window_s=10,
    rate_threshold=15,
    block_cooldown_s=600,
    buffer_max=10,
)


# --- Primitives ------------------------------------------------------------


async def test_set_then_is_handed_off_true() -> None:
    redis = FakeAsyncRedis(decode_responses=True)

    assert not await handoff.is_handed_off(redis, SENDER)
    await handoff.set_handoff(redis, SENDER)
    assert await handoff.is_handed_off(redis, SENDER)


async def test_clear_handoff_returns_to_bot() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    await handoff.set_handoff(redis, SENDER)

    await handoff.clear_handoff(redis, SENDER)

    assert not await handoff.is_handed_off(redis, SENDER)


# --- Endpoint short-circuit -----------------------------------------------


class StubLLM:
    async def complete_structured(
        self, prompt: str, schema: type[BaseModel]
    ) -> BaseModel:
        return schema()


class FakeChannel:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def verify_signature(
        self, url: str, params: Mapping[str, str], signature: str
    ) -> bool:
        return True

    def parse_incoming(self, form: Mapping[str, str]) -> IncomingMessage:
        try:
            return IncomingMessage(
                sender=form[TwilioField.FROM],
                text=form[TwilioField.BODY],
                message_id=form[TwilioField.MESSAGE_SID],
            )
        except KeyError as exc:
            raise InvalidPayloadError(str(exc)) from exc

    async def send(self, to: str, text: str) -> None:
        self.sent.append((to, text))


def _client_with(channel: FakeChannel) -> TestClient:
    app.dependency_overrides[get_channel] = lambda: channel
    app.dependency_overrides[get_redis] = lambda: FakeAsyncRedis(decode_responses=True)
    app.dependency_overrides[get_llm] = lambda: StubLLM()
    app.dependency_overrides[get_concurrency_config] = lambda: TEST_CONFIG
    app.dependency_overrides[get_routing_config] = lambda: RoutingConfig(
        prefill_crucero=None, prefill_europa=None, prefill_asia=None
    )
    app.dependency_overrides[get_store] = lambda: InMemoryStateStore()
    app.dependency_overrides[get_corpus] = lambda: "CORPUS DE PRUEBA"
    # The real runner needs the CRM client (token/base_url); these endpoint tests
    # mock schedule_flush, so a placeholder is enough.
    app.dependency_overrides[get_handoff_runner] = lambda: object()
    # schedule_flush is mocked in these tests, so the connector is never used.
    app.dependency_overrides[get_chat_connector] = lambda: None
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def test_handed_off_relays_and_stays_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    channel = FakeChannel()
    relay = AsyncMock()
    flush_spy = Mock()
    monkeypatch.setattr(handoff, "is_handed_off", AsyncMock(return_value=True))
    monkeypatch.setattr(main_module, "relay_to_human", relay)
    monkeypatch.setattr(main_module, "schedule_flush", flush_spy)

    response = _client_with(channel).post(WEBHOOK_PATH, data=VALID_FORM)

    assert response.status_code == 200
    relay.assert_awaited_once()
    # Bot is silent and the inc-2 flow (buffer/debounce/orchestrator) is skipped.
    assert channel.sent == []
    flush_spy.assert_not_called()


def test_not_handed_off_runs_normal_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    channel = FakeChannel()
    relay = AsyncMock()
    flush_spy = Mock()
    monkeypatch.setattr(main_module, "relay_to_human", relay)
    monkeypatch.setattr(main_module, "schedule_flush", flush_spy)

    response = _client_with(channel).post(WEBHOOK_PATH, data=VALID_FORM)

    assert response.status_code == 200
    relay.assert_not_awaited()
    # The normal inc-2 path schedules the background flush.
    flush_spy.assert_called_once()
