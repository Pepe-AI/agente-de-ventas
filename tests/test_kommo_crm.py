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
from app.crm.kommo_crm import KommoContactMatch, KommoCrmClient, KommoCrmError

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


async def test_find_contact_by_phone_returns_matches_with_lead_ids() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["req"] = request
        return httpx.Response(
            200,
            json={
                "_embedded": {
                    "contacts": [
                        {"id": 111, "_embedded": {"leads": [{"id": 10}, {"id": 11}]}},
                        {"id": 222, "_embedded": {"leads": []}},
                    ]
                }
            },
        )

    async with _client(handler) as client:
        matches = await client.find_contact_by_phone("+5215512345678")

    assert matches == [
        KommoContactMatch(111, (10, 11)),
        KommoContactMatch(222, ()),
    ]
    req = seen["req"]
    assert req.method == "GET"
    assert req.url.path == "/api/v4/contacts"
    # Confirmed CRM-v4 shape: full-text query is the phone, with=leads embeds them.
    assert req.url.params["query"] == "+5215512345678"
    assert req.url.params["with"] == "leads"
    assert req.headers["Authorization"] == f"Bearer {_TOKEN}"


async def test_find_contact_by_phone_returns_empty_on_no_match_204() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(204)  # Kommo returns no content when nothing matches

    async with _client(handler) as client:
        matches = await client.find_contact_by_phone("+10000000000")

    assert matches == []


async def test_find_contact_by_phone_raises_typed_error_on_non_2xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="Forbidden")

    async with _client(handler) as client:
        with pytest.raises(KommoCrmError) as exc_info:
            await client.find_contact_by_phone("+5215512345678")

    assert exc_info.value.status == 403
    assert exc_info.value.body == "Forbidden"


async def test_create_lead_resolves_phone_field_then_posts_complex() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v4/contacts/custom_fields":
            return httpx.Response(
                200,
                json={
                    "_embedded": {
                        "custom_fields": [
                            {"id": 5, "code": "POSITION"},
                            {"id": 1698052, "code": "PHONE"},
                        ]
                    }
                },
            )
        return httpx.Response(200, json=[{"id": 999, "contact_id": 555}])

    async with _client(handler) as client:
        lead_id = await client.create_lead_with_contact(
            "Lead X", "Cliente Y", "+5215512345678"
        )

    assert lead_id == 999
    # The phone field id was looked up by code, then used in the complex payload.
    complex_req = next(
        r for r in requests if r.url.path == "/api/v4/leads/complex"
    )
    assert complex_req.method == "POST"
    assert complex_req.headers["Authorization"] == f"Bearer {_TOKEN}"
    assert json.loads(complex_req.content) == [
        {
            "name": "Lead X",
            "_embedded": {
                "contacts": [
                    {
                        "name": "Cliente Y",
                        "custom_fields_values": [
                            {
                                "field_id": 1698052,
                                "values": [{"value": "+5215512345678"}],
                            }
                        ],
                    }
                ]
            },
        }
    ]


async def test_create_lead_caches_phone_field_id_across_calls() -> None:
    custom_field_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal custom_field_calls
        if request.url.path == "/api/v4/contacts/custom_fields":
            custom_field_calls += 1
            return httpx.Response(
                200,
                json={"_embedded": {"custom_fields": [{"id": 7, "code": "PHONE"}]}},
            )
        return httpx.Response(200, json=[{"id": 1}])

    async with _client(handler) as client:
        await client.create_lead_with_contact("L1", "C1", "+111")
        await client.create_lead_with_contact("L2", "C2", "+222")

    assert custom_field_calls == 1  # resolved once, then cached


async def test_create_lead_raises_when_phone_field_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v4/contacts/custom_fields":
            return httpx.Response(
                200,
                json={"_embedded": {"custom_fields": [{"id": 5, "code": "POSITION"}]}},
            )
        return httpx.Response(200, json=[{"id": 1}])

    async with _client(handler) as client:
        with pytest.raises(KommoCrmError, match="PHONE"):
            await client.create_lead_with_contact("L", "C", "+1")


async def test_create_lead_raises_typed_error_on_non_2xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v4/contacts/custom_fields":
            return httpx.Response(
                200,
                json={"_embedded": {"custom_fields": [{"id": 7, "code": "PHONE"}]}},
            )
        return httpx.Response(400, text="Bad Request")

    async with _client(handler) as client:
        with pytest.raises(KommoCrmError) as exc_info:
            await client.create_lead_with_contact("L", "C", "+1")

    assert exc_info.value.status == 400
    assert exc_info.value.body == "Bad Request"


async def test_update_lead_patches_custom_fields_status_and_pipeline() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["req"] = request
        return httpx.Response(200, json={"id": _LEAD_ID, "status_id": 107566779})

    cfv = [{"field_id": 1112708, "values": [{"value": "Italia"}]}]
    async with _client(handler) as client:
        result = await client.update_lead(
            _LEAD_ID,
            custom_fields_values=cfv,
            status_id=107566779,
            pipeline_id=13937935,
        )

    assert result == {"id": _LEAD_ID, "status_id": 107566779}
    req = seen["req"]
    assert req.method == "PATCH"
    assert str(req.url) == f"{_BASE_URL}/api/v4/leads/{_LEAD_ID}"
    assert req.headers["Authorization"] == f"Bearer {_TOKEN}"
    # One PATCH carries custom fields AND the stage move (status + pipeline).
    assert json.loads(req.content) == {
        "custom_fields_values": cfv,
        "status_id": 107566779,
        "pipeline_id": 13937935,
    }


async def test_update_lead_sends_only_custom_fields_when_no_stage_move() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["req"] = request
        return httpx.Response(200, json={"id": _LEAD_ID})

    cfv = [{"field_id": 1112714, "values": [{"value": "Julio"}]}]
    async with _client(handler) as client:
        await client.update_lead(_LEAD_ID, custom_fields_values=cfv)

    assert json.loads(seen["req"].content) == {"custom_fields_values": cfv}


async def test_update_lead_omits_empty_custom_fields() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["req"] = request
        return httpx.Response(200, json={"id": _LEAD_ID})

    async with _client(handler) as client:
        await client.update_lead(
            _LEAD_ID, custom_fields_values=[], status_id=107566783, pipeline_id=13937935
        )

    # No field has a value -> custom_fields_values must NOT be in the body.
    assert json.loads(seen["req"].content) == {
        "status_id": 107566783,
        "pipeline_id": 13937935,
    }


async def test_update_lead_raises_typed_error_on_non_2xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="Bad Request")

    async with _client(handler) as client:
        with pytest.raises(KommoCrmError) as exc_info:
            await client.update_lead(_LEAD_ID, status_id=1, pipeline_id=2)

    assert exc_info.value.status == 400
    assert exc_info.value.body == "Bad Request"


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
