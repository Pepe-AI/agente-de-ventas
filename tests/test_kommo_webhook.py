"""Tests for the inbound Kommo Chats webhook (offline; httpx/Kommo not real).

Rule under test: verify the HMAC signature over the RAW bytes -> fast-ack ->
durable enqueue. Invalid/absent signature is rejected before any processing.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fakeredis import FakeAsyncRedis
from fastapi import HTTPException
from fastapi.testclient import TestClient

import app.main as main_module
from app.concurrency.keys import KeyPrefix, make_key
from app.crm.kommo_inbound import enqueue_inbound
from app.crm.kommo_signing import KommoSigner
from app.main import app, get_kommo_signer, get_redis

_SECRET = "test-channel-secret"
_SCOPE = "ch123_acc456"
_SIGNER = KommoSigner(_SECRET)
_PATH = f"/kommo/chats/webhook/{_SCOPE}"


def _client_with_enqueue_spy(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, AsyncMock]:
    enqueue = AsyncMock()
    monkeypatch.setattr(main_module, "enqueue_inbound", enqueue)
    app.dependency_overrides[get_kommo_signer] = lambda: KommoSigner(_SECRET)
    app.dependency_overrides[get_redis] = lambda: FakeAsyncRedis(decode_responses=True)
    return TestClient(app), enqueue


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def test_valid_signature_acks_200_and_enqueues_raw_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, enqueue = _client_with_enqueue_spy(monkeypatch)
    # Unusual spacing/key-order: a re-serialization would change these bytes.
    body = b'{ "message" : { "text": "hola agente" }, "b": 2, "a": 1 }'
    signature = _SIGNER.sign(body)

    response = client.post(_PATH, content=body, headers={"X-Signature": signature})

    assert response.status_code == 200
    enqueue.assert_awaited_once()
    _redis, scope_id, enqueued_body = enqueue.await_args.args
    assert scope_id == _SCOPE
    assert enqueued_body == body  # exact raw bytes, never re-serialized


def test_signature_header_is_case_insensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, enqueue = _client_with_enqueue_spy(monkeypatch)
    body = b'{"message":{"text":"hola"}}'
    signature = _SIGNER.sign(body)

    response = client.post(_PATH, content=body, headers={"x-signature": signature})

    assert response.status_code == 200
    enqueue.assert_awaited_once()


def test_invalid_signature_returns_401_and_does_not_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, enqueue = _client_with_enqueue_spy(monkeypatch)
    body = b'{"message":{"text":"hola"}}'

    response = client.post(_PATH, content=body, headers={"X-Signature": "deadbeef"})

    assert response.status_code == 401
    enqueue.assert_not_awaited()


def test_missing_signature_header_returns_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, enqueue = _client_with_enqueue_spy(monkeypatch)
    body = b'{"message":{"text":"hola"}}'

    response = client.post(_PATH, content=body)  # no X-Signature header

    assert response.status_code == 401
    enqueue.assert_not_awaited()


def test_enqueue_failure_returns_500(monkeypatch: pytest.MonkeyPatch) -> None:
    client, enqueue = _client_with_enqueue_spy(monkeypatch)
    enqueue.side_effect = RuntimeError("redis down")
    body = b'{"message":{"text":"hola"}}'
    signature = _SIGNER.sign(body)

    response = client.post(_PATH, content=body, headers={"X-Signature": signature})

    assert response.status_code == 500


def test_get_kommo_signer_raises_503_when_secret_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_module, "get_settings", lambda: SimpleNamespace(kommo_channel_secret=None)
    )

    with pytest.raises(HTTPException) as exc_info:
        main_module.get_kommo_signer()

    assert exc_info.value.status_code == 503


async def test_enqueue_inbound_durably_stores_payload() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    body = b'{"x":1}'

    await enqueue_inbound(redis, _SCOPE, body)

    key = make_key(KeyPrefix.KOMMO_INBOUND, _SCOPE)
    assert await redis.lrange(key, 0, -1) == ['{"x":1}']


async def test_web_app_boot_fails_fast_without_kommo_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The web app refuses to boot without the Kommo secret — and fails BEFORE any
    # DB connection (the check is first in the lifespan).
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: SimpleNamespace(
            kommo_channel_secret=None, database_url="postgresql://unused"
        ),
    )

    with pytest.raises(RuntimeError, match="KOMMO_CHANNEL_SECRET"):
        async with main_module.lifespan(main_module.app):
            pass


def test_kommo_secret_is_optional_so_migrate_is_unaffected() -> None:
    # The migration runner calls get_settings(); the fail-fast lives only in the
    # web app's lifespan. The secret must stay OPTIONAL in Settings so migrate
    # (and get_settings) never require it.
    from app.config import Settings

    assert Settings.model_fields["kommo_channel_secret"].is_required() is False
