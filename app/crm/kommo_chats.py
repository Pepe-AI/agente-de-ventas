"""Minimal Kommo Chats API client (async, httpx).

One method for now — ``connect`` — which links the Kommo account to this custom
channel and returns the ``scope_id`` used later to send/receive messages. It is
also the real acceptance test of :class:`~app.crm.kommo_signing.KommoSigner`: a
200 + scope_id confirms our signature is correct.

CRITICAL: the request body is serialized to bytes ONCE, those exact bytes are
signed, and they are sent verbatim via httpx ``content=`` (NEVER ``json=``, which
re-serializes and changes the bytes -> broken signature).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

import httpx

from app.crm.kommo_signing import KommoSigner

KOMMO_BASE_URL = "https://amojo.kommo.com"
_TIMEOUT_S = 15.0
_HTTP_OK = 200


@dataclass(frozen=True, slots=True)
class KommoChatSender:
    """The sender of a chat message (all fields provided by the caller).

    ``avatar`` must be a URL publicly reachable by Kommo (not localhost).
    """

    id: str
    avatar: str
    name: str
    phone: str


@dataclass(frozen=True, slots=True)
class KommoChatMessage:
    """A chat message to push to Kommo (a transport value object).

    Every field is supplied by the caller — including ``timestamp`` (Unix epoch
    seconds) and ``msgid`` (e.g. the Twilio SID) — so the client invents nothing
    and tests stay deterministic.
    """

    conversation_id: str
    msgid: str
    timestamp: int
    sender: KommoChatSender
    text: str


class KommoChatsError(RuntimeError):
    """A Kommo Chats API call failed (non-200 status or a network error).

    Carries the HTTP ``status`` and response ``body`` when available, for
    diagnosis. ``status`` is ``None`` for a network/transport failure.
    """

    def __init__(
        self, message: str, *, status: int | None = None, body: str | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class KommoChatsClient:
    """Async client for the Kommo Chats API (signs every outbound request)."""

    def __init__(
        self,
        signer: KommoSigner,
        channel_id: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._signer = signer
        self._channel_id = channel_id
        # ``transport`` is injected only in tests (httpx.MockTransport); in
        # production it is None and httpx uses its real network transport.
        self._client = httpx.AsyncClient(
            base_url=KOMMO_BASE_URL, timeout=_TIMEOUT_S, transport=transport
        )

    async def __aenter__(self) -> KommoChatsClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def connect(self, amojo_id: str) -> str:
        """Connect ``amojo_id`` to this channel; return the ``scope_id``."""
        body = json.dumps({"account_id": amojo_id}).encode("utf-8")  # serialize ONCE
        path = f"/v2/origin/custom/{self._channel_id}/connect"
        try:
            response = await self._client.post(
                path, content=body, headers=self._signer.outbound_headers(body)
            )
        except httpx.RequestError as exc:
            raise KommoChatsError(f"Kommo connect request failed: {exc}") from exc

        if response.status_code != _HTTP_OK:
            raise KommoChatsError(
                "Kommo connect returned a non-200 status",
                status=response.status_code,
                body=response.text,
            )
        return self._scope_id_of(response)

    async def send_message(
        self, scope_id: str, message: KommoChatMessage
    ) -> dict[str, object]:
        """Push a ``new_message`` to the chat at ``scope_id``; return the 2xx body.

        Transport primitive, agnostic to policy: it builds / signs / posts exactly
        what the caller passes and derives nothing (conversation_id, avatar,
        direction, timestamp, mirror-vs-import). A ``new_message`` whose ``sender``
        is the external party represents an INBOUND (customer) message.
        """
        body = self._new_message_body(message)  # serialize ONCE
        path = f"/v2/origin/custom/{scope_id}"
        try:
            response = await self._client.post(
                path, content=body, headers=self._signer.outbound_headers(body)
            )
        except httpx.RequestError as exc:
            raise KommoChatsError(f"Kommo send_message request failed: {exc}") from exc

        if not 200 <= response.status_code < 300:
            raise KommoChatsError(
                "Kommo send_message returned a non-2xx status",
                status=response.status_code,
                body=response.text,
            )
        return self._parsed_object(response)

    @staticmethod
    def _new_message_body(message: KommoChatMessage) -> bytes:
        """Serialize the ``new_message`` payload to bytes exactly once."""
        payload: dict[str, object] = {
            "event_type": "new_message",
            "payload": {
                "timestamp": message.timestamp,
                "msgid": message.msgid,
                "conversation_id": message.conversation_id,
                "sender": {
                    "id": message.sender.id,
                    "avatar": message.sender.avatar,
                    "name": message.sender.name,
                    "profile": {"phone": message.sender.phone},
                },
                "message": {"type": "text", "text": message.text},
            },
        }
        return json.dumps(payload).encode("utf-8")

    @staticmethod
    def _parsed_object(response: httpx.Response) -> dict[str, object]:
        try:
            payload: object = response.json()
        except ValueError as exc:  # JSONDecodeError
            raise KommoChatsError(
                "Kommo response was not valid JSON",
                status=response.status_code,
                body=response.text,
            ) from exc
        if not isinstance(payload, dict):
            raise KommoChatsError(
                "Kommo response was not a JSON object",
                status=response.status_code,
                body=response.text,
            )
        return cast("dict[str, object]", payload)

    @staticmethod
    def _scope_id_of(response: httpx.Response) -> str:
        try:
            payload: object = response.json()
        except ValueError as exc:  # JSONDecodeError
            raise KommoChatsError(
                "Kommo connect response was not valid JSON",
                status=response.status_code,
                body=response.text,
            ) from exc
        if isinstance(payload, dict):
            scope_id = cast("dict[str, object]", payload).get("scope_id")
            if isinstance(scope_id, str):
                return scope_id
        raise KommoChatsError(
            "Kommo connect response missing scope_id",
            status=response.status_code,
            body=response.text,
        )
