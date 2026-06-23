"""Minimal Kommo CRM API v4 client (async, httpx) — DISTINCT from the Chats API.

The CRM API (the account subdomain, e.g. ``https://<account>.kommo.com``) uses
Bearer auth with a long-lived token and plain JSON requests — NO HMAC body
signing. That signing belongs to the Chats API's ``KommoChatsClient`` (a different
subsystem, a different credential). Only the STYLE (structure, typed error) is
shared here, never the signing mechanism.

All calls funnel through one method (``_request``) so a rate limiter (the CRM API
caps ~7 req/s) can be added there later; none is implemented yet.

Endpoints (confirmed against Kommo API v4 docs):
* auth: ``Authorization: Bearer <token>``
* ``GET /api/v4/account`` (connectivity smoke test)
* add a text note: ``POST /api/v4/leads/notes`` body
  ``[{"entity_id": <lead_id>, "note_type": "common", "params": {"text": ...}}]``
* find a contact by phone: ``GET /api/v4/contacts?query={phone}&with=leads``
  (full-text query matches phone by its last 7 digits; returns no body / 204 when
  there are no matches)
* create a lead + embedded contact: ``POST /api/v4/leads/complex`` body
  ``[{"name", "_embedded": {"contacts": [{"name", "custom_fields_values": [...]}]}}]``
  -> response ``[{"id", "contact_id", ...}]``. The contact PHONE is a standard
  custom field referenced by its NUMERIC ``field_id`` (resolved once from
  ``GET /api/v4/contacts/custom_fields`` by ``code == "PHONE"``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import httpx

_TIMEOUT_S = 15.0
_ACCOUNT_PATH = "/api/v4/account"
_LEAD_NOTES_PATH = "/api/v4/leads/notes"
_CONTACTS_PATH = "/api/v4/contacts"
_CONTACTS_CUSTOM_FIELDS_PATH = "/api/v4/contacts/custom_fields"
_LEADS_COMPLEX_PATH = "/api/v4/leads/complex"
_NOTE_TYPE_COMMON = "common"  # Kommo's note_type for a plain text note
_PHONE_FIELD_CODE = "PHONE"  # code of the standard contact phone custom field


@dataclass(frozen=True, slots=True)
class KommoContactMatch:
    """A contact returned by a phone search, with its linked lead ids."""

    contact_id: int
    lead_ids: tuple[int, ...]


class KommoCrmError(RuntimeError):
    """A Kommo CRM API call failed (non-2xx status or a network error).

    Carries the HTTP ``status`` and response ``body`` when available; ``status``
    is ``None`` for a network/transport failure.
    """

    def __init__(
        self, message: str, *, status: int | None = None, body: str | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class KommoCrmClient:
    """Async Kommo CRM API v4 client (Bearer auth, plain JSON, no signing)."""

    def __init__(
        self,
        token: str,
        base_url: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # ``base_url`` is a parameter (testable); the caller provides it from
        # config. ``transport`` is injected only in tests (httpx.MockTransport).
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=_TIMEOUT_S,
            transport=transport,
            headers={"Authorization": f"Bearer {token}"},
        )
        self._phone_field_id: int | None = None  # resolved + cached on first use

    async def __aenter__(self) -> KommoCrmClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get_account(self) -> dict[str, object]:
        """GET the account info — a connectivity smoke test for the token."""
        result = await self._request("GET", _ACCOUNT_PATH)
        if not isinstance(result, dict):
            raise KommoCrmError("Kommo CRM account response was not a JSON object")
        return cast("dict[str, object]", result)

    async def add_note(self, lead_id: int, text: str) -> object:
        """Add a plain-text ('common') note to a lead; return the parsed body."""
        payload = [
            {
                "entity_id": lead_id,
                "note_type": _NOTE_TYPE_COMMON,
                "params": {"text": text},
            }
        ]
        return await self._request("POST", _LEAD_NOTES_PATH, json=payload)

    async def find_contact_by_phone(self, phone: str) -> list[KommoContactMatch]:
        """Return the contacts matching ``phone``, each with its linked lead ids.

        Pure read: it reports what exists (or an empty list) and decides nothing.
        """
        result = await self._request(
            "GET", _CONTACTS_PATH, params={"query": phone, "with": "leads"}
        )
        return _parse_contact_matches(result)

    async def create_lead_with_contact(
        self, lead_name: str, contact_name: str, phone: str
    ) -> int:
        """Create a lead with an embedded contact (phone set); return the lead id.

        The lead lands in the default pipeline/stage — moving it is out of scope.
        """
        phone_field_id = await self._resolve_phone_field_id()
        payload = [
            {
                "name": lead_name,
                "_embedded": {
                    "contacts": [
                        {
                            "name": contact_name,
                            "custom_fields_values": [
                                {
                                    "field_id": phone_field_id,
                                    "values": [{"value": phone}],
                                }
                            ],
                        }
                    ]
                },
            }
        ]
        result = await self._request("POST", _LEADS_COMPLEX_PATH, json=payload)
        return _first_lead_id(result)

    async def _resolve_phone_field_id(self) -> int:
        """Find the contact PHONE custom field's numeric id (cached per client)."""
        if self._phone_field_id is not None:
            return self._phone_field_id
        result = await self._request("GET", _CONTACTS_CUSTOM_FIELDS_PATH)
        for field in _embedded_list(result, "custom_fields"):
            if isinstance(field, dict):
                field_dict = cast("dict[str, object]", field)
                if field_dict.get("code") == _PHONE_FIELD_CODE:
                    field_id = field_dict.get("id")
                    if isinstance(field_id, int):
                        self._phone_field_id = field_id
                        return field_id
        raise KommoCrmError("Kommo CRM: PHONE contact custom field not found")

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        json: object = None,
    ) -> object | None:
        """The single request path — so a rate limiter can wrap it later."""
        try:
            response = await self._client.request(
                method, path, params=params, json=json
            )
        except httpx.RequestError as exc:
            raise KommoCrmError(f"Kommo CRM request failed: {exc}") from exc

        if not 200 <= response.status_code < 300:
            raise KommoCrmError(
                "Kommo CRM returned a non-2xx status",
                status=response.status_code,
                body=response.text,
            )
        if not response.content:
            return None  # e.g. 204 No Content (a search with no matches)
        try:
            return response.json()
        except ValueError as exc:  # JSONDecodeError
            raise KommoCrmError(
                "Kommo CRM response was not valid JSON",
                status=response.status_code,
                body=response.text,
            ) from exc


def _embedded_list(obj: object, key: str) -> list[object]:
    """Return ``obj["_embedded"][key]`` if it is a list, else an empty list."""
    if not isinstance(obj, dict):
        return []
    embedded = cast("dict[str, object]", obj).get("_embedded")
    if not isinstance(embedded, dict):
        return []
    value = cast("dict[str, object]", embedded).get(key)
    return cast("list[object]", value) if isinstance(value, list) else []


def _parse_contact_matches(result: object) -> list[KommoContactMatch]:
    matches: list[KommoContactMatch] = []
    for contact in _embedded_list(result, "contacts"):
        if not isinstance(contact, dict):
            continue
        contact_dict = cast("dict[str, object]", contact)
        contact_id = contact_dict.get("id")
        if isinstance(contact_id, int):
            matches.append(KommoContactMatch(contact_id, _lead_ids(contact_dict)))
    return matches


def _lead_ids(contact: object) -> tuple[int, ...]:
    ids: list[int] = []
    for lead in _embedded_list(contact, "leads"):
        if isinstance(lead, dict):
            lead_id = cast("dict[str, object]", lead).get("id")
            if isinstance(lead_id, int):
                ids.append(lead_id)
    return tuple(ids)


def _first_lead_id(result: object) -> int:
    if isinstance(result, list):
        items = cast("list[object]", result)
        if items and isinstance(items[0], dict):
            lead_id = cast("dict[str, object]", items[0]).get("id")
            if isinstance(lead_id, int):
                return lead_id
    raise KommoCrmError("Kommo CRM complex lead response missing the lead id")
