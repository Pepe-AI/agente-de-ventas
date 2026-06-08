"""Tests for the webhook HTTP behavior with a mocked transport channel."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from fastapi.testclient import TestClient

from app.channels.twilio import InvalidPayloadError, TwilioField
from app.domain.models import IncomingMessage
from app.main import WEBHOOK_PATH, app, get_channel

VALID_FORM = {
    "From": "whatsapp:+5215512345678",
    "Body": "hola",
    "MessageSid": "SM123",
}


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
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides() -> None:
    yield
    app.dependency_overrides.clear()


def test_invalid_signature_returns_403_and_does_not_send() -> None:
    channel = FakeChannel(signature_valid=False)
    client = _client_with(channel)

    response = client.post(WEBHOOK_PATH, data=VALID_FORM)

    assert response.status_code == 403
    assert channel.sent == []


def test_valid_signature_returns_200_and_echoes() -> None:
    channel = FakeChannel(signature_valid=True)
    client = _client_with(channel)

    response = client.post(WEBHOOK_PATH, data=VALID_FORM)

    assert response.status_code == 200
    assert channel.sent == [("whatsapp:+5215512345678", "Echo: hola")]
