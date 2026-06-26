"""Tests for the webhook HTTP behavior with mocked transport + fakeredis.

The endpoint now fast-acks: a valid request is buffered and the reply is sent by
the background flush, not within the request. These tests assert the HTTP
contract (status codes, no synchronous send); flush behavior is covered in
test_concurrency.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.channels.twilio import InvalidPayloadError, TwilioField
from app.concurrency.config import ConcurrencyConfig
from app.domain.models import IncomingMessage
from app.main import (
    WEBHOOK_PATH,
    app,
    get_channel,
    get_chat_connector,
    get_chat_mirror,
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

VALID_FORM = {
    "From": "whatsapp:+5215512345678",
    "Body": "hola",
    "MessageSid": "SM123",
}

TEST_CONFIG = ConcurrencyConfig(
    debounce_window_s=60.0,  # long window so the background flush never fires in-test
    dedup_ttl_s=3600,
    lock_ttl_s=30,
    rate_window_s=10,
    rate_threshold=15,
    block_cooldown_s=600,
    buffer_max=10,
    inactivity_deadline_s=7200.0,
)


class StubLLM:
    """Never invoked in these tests (the flush does not run); satisfies get_llm."""

    async def complete_structured(
        self, prompt: str, schema: type[BaseModel]
    ) -> BaseModel:
        return schema()


class FakeChannel:
    """In-memory Channel implementation for assertions."""

    def __init__(self, *, signature_valid: bool) -> None:
        self._signature_valid = signature_valid
        self.sent: list[tuple[str, str]] = []

    def verify_signature(
        self, url: str, params: Mapping[str, str], signature: str
    ) -> bool:
        return self._signature_valid

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
    # The flush never runs here; a placeholder runner avoids building the real CRM
    # client (which would need token/base_url config).
    app.dependency_overrides[get_handoff_runner] = lambda: object()
    app.dependency_overrides[get_chat_connector] = lambda: None
    app.dependency_overrides[get_chat_mirror] = lambda: None
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def test_invalid_signature_returns_403_and_does_not_send() -> None:
    channel = FakeChannel(signature_valid=False)
    client = _client_with(channel)

    response = client.post(WEBHOOK_PATH, data=VALID_FORM)

    assert response.status_code == 403
    assert channel.sent == []


def test_valid_signature_fast_acks_without_sending_synchronously() -> None:
    channel = FakeChannel(signature_valid=True)
    client = _client_with(channel)

    response = client.post(WEBHOOK_PATH, data=VALID_FORM)

    # Fast-ack: 200 returns immediately; the reply is the background flush's job.
    assert response.status_code == 200
    assert channel.sent == []
