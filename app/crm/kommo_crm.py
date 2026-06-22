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
* add a text note: ``POST /api/v4/leads/notes`` with body
  ``[{"entity_id": <lead_id>, "note_type": "common", "params": {"text": ...}}]``
"""

from __future__ import annotations

from typing import cast

import httpx

_TIMEOUT_S = 15.0
_ACCOUNT_PATH = "/api/v4/account"
_LEAD_NOTES_PATH = "/api/v4/leads/notes"
_NOTE_TYPE_COMMON = "common"  # Kommo's note_type for a plain text note


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

    async def _request(self, method: str, path: str, *, json: object = None) -> object:
        """The single request path — so a rate limiter can wrap it later."""
        try:
            response = await self._client.request(method, path, json=json)
        except httpx.RequestError as exc:
            raise KommoCrmError(f"Kommo CRM request failed: {exc}") from exc

        if not 200 <= response.status_code < 300:
            raise KommoCrmError(
                "Kommo CRM returned a non-2xx status",
                status=response.status_code,
                body=response.text,
            )
        try:
            return response.json()
        except ValueError as exc:  # JSONDecodeError
            raise KommoCrmError(
                "Kommo CRM response was not valid JSON",
                status=response.status_code,
                body=response.text,
            ) from exc
