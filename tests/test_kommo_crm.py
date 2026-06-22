"""Offline tests for KommoCrmClient (CRM API v4) + the long-lived-token wiring.

The real CRM calls are validated by scripts/add_test_note.py, not here.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from types import SimpleNamespace

import httpx
import pytest

import app.main as main_module
from app.config import Settings
from app.crm.kommo_crm import KommoCrmClient, KommoCrmError

_TOKEN = "long-lived-token-xyz"
_BASE_URL = "https://asuareztopviajescommx.kommo.com"
_LEAD_ID = 123456

_Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: _Handler) -> KommoCrmClient:
    return KommoCrmClient(_TOKEN, _BASE_URL, transport=httpx.MockTransport(handler))


async def test_get_account_sends_bearer_get_and_returns_body() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["req"] = request
        return httpx.Response(200, json={"id": 36614339, "name": "Top Viajes"})

    async with _client(handler) as client:
        account = await client.get_account()

    assert account == {"id": 36614339, "name": "Top Viajes"}
    req = seen["req"]
    assert req.method == "GET"
    assert str(req.url) == f"{_BASE_URL}/api/v4/account"
    assert req.headers["Authorization"] == f"Bearer {_TOKEN}"


async def test_add_note_posts_common_note_to_lead() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["req"] = request
        return httpx.Response(200, json={"_embedded": {"notes": [{"id": 1}]}})

    async with _client(handler) as client:
        result = await client.add_note(_LEAD_ID, "Resumen del lead")

    assert result == {"_embedded": {"notes": [{"id": 1}]}}
    req = seen["req"]
    assert req.method == "POST"
    assert str(req.url) == f"{_BASE_URL}/api/v4/leads/notes"
    assert req.headers["Authorization"] == f"Bearer {_TOKEN}"
    # Confirmed CRM-v4 shape: lead_id goes in the body as entity_id, note_type
    # "common", text under params.
    assert json.loads(req.content) == [
        {
            "entity_id": _LEAD_ID,
            "note_type": "common",
            "params": {"text": "Resumen del lead"},
        }
    ]


async def test_crm_raises_typed_error_on_non_2xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="Unauthorized")

    async with _client(handler) as client:
        with pytest.raises(KommoCrmError) as exc_info:
            await client.get_account()

    assert exc_info.value.status == 401
    assert exc_info.value.body == "Unauthorized"


async def test_crm_wraps_network_error_as_typed_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async with _client(handler) as client:
        with pytest.raises(KommoCrmError) as exc_info:
            await client.add_note(_LEAD_ID, "x")

    assert exc_info.value.status is None  # network failure, no HTTP status


def test_kommo_long_lived_token_is_optional_so_migrate_is_unaffected() -> None:
    # migrate.py builds the full Settings; the token must NOT be required there.
    assert Settings.model_fields["kommo_long_lived_token"].is_required() is False


async def test_web_app_boot_fails_fast_without_kommo_long_lived_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Single defense layer this increment: the web app refuses to boot without
    # the token (before any DB connection). No request-time 503 yet (no caller).
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: SimpleNamespace(
            kommo_channel_secret="present",  # passes the first fail-fast check
            kommo_long_lived_token=None,
            database_url="postgresql://unused",
        ),
    )

    with pytest.raises(RuntimeError, match="KOMMO_LONG_LIVED_TOKEN"):
        async with main_module.lifespan(main_module.app):
            pass
