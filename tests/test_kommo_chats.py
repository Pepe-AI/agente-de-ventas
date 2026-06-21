"""Offline tests for KommoChatsClient.connect (httpx mocked via MockTransport).

The real connect is validated by scripts/connect_kommo_channel.py, not here.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from app.crm.kommo_chats import KOMMO_BASE_URL, KommoChatsClient, KommoChatsError
from app.crm.kommo_signing import KommoHeader, KommoSigner

_SECRET = "channel-secret"
_CHANNEL_ID = "chan-123"
_AMOJO_ID = "amojo-xyz"
_SCOPE_ID = f"{_CHANNEL_ID}_{_AMOJO_ID}"

_Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: _Handler) -> KommoChatsClient:
    return KommoChatsClient(
        KommoSigner(_SECRET), _CHANNEL_ID, transport=httpx.MockTransport(handler)
    )


async def test_connect_posts_signed_request_and_returns_scope_id() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["req"] = request
        return httpx.Response(
            200, json={"account_id": _AMOJO_ID, "scope_id": _SCOPE_ID}
        )

    async with _client(handler) as client:
        scope_id = await client.connect(_AMOJO_ID)

    assert scope_id == _SCOPE_ID
    req = seen["req"]
    assert req.method == "POST"
    assert str(req.url) == f"{KOMMO_BASE_URL}/v2/origin/custom/{_CHANNEL_ID}/connect"
    # Body sent as the EXACT bytes that were signed (content=, not json=).
    expected_body = json.dumps({"account_id": _AMOJO_ID}).encode("utf-8")
    assert req.content == expected_body
    # Signed headers computed over those exact bytes are present and correct.
    signer = KommoSigner(_SECRET)
    assert req.headers[KommoHeader.SIGNATURE] == signer.sign(expected_body)
    assert req.headers[KommoHeader.CONTENT_TYPE] == "application/json"
    assert KommoHeader.CONTENT_MD5 in req.headers
    assert KommoHeader.DATE in req.headers


async def test_connect_raises_typed_error_on_non_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    async with _client(handler) as client:
        with pytest.raises(KommoChatsError) as exc_info:
            await client.connect(_AMOJO_ID)

    assert exc_info.value.status == 403
    assert exc_info.value.body == "forbidden"


async def test_connect_raises_when_scope_id_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"account_id": _AMOJO_ID})  # no scope_id

    async with _client(handler) as client:
        with pytest.raises(KommoChatsError):
            await client.connect(_AMOJO_ID)


async def test_connect_wraps_network_error_as_typed_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async with _client(handler) as client:
        with pytest.raises(KommoChatsError) as exc_info:
            await client.connect(_AMOJO_ID)

    assert exc_info.value.status is None  # network failure, no HTTP status
