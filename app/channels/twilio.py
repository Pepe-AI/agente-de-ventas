"""Twilio WhatsApp transport adapter.

Implements the :class:`~app.channels.base.Channel` port plus request-signature
verification. This is the only module that knows about Twilio's wire format and
SDK; the rest of the app depends on the abstraction.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

from anyio import to_thread
from twilio.request_validator import RequestValidator
from twilio.rest import Client

from app.domain.models import IncomingMessage


class TwilioField(StrEnum):
    """Field names in Twilio's inbound WhatsApp form payload."""

    FROM = "From"
    BODY = "Body"
    MESSAGE_SID = "MessageSid"


class InvalidPayloadError(ValueError):
    """Raised when an inbound payload is missing required Twilio fields."""


class TwilioChannel:
    """Adapter for the Twilio WhatsApp API.

    The Twilio SDK is synchronous, so :meth:`send` offloads the blocking call
    to a worker thread to keep the event loop responsive.
    """

    def __init__(
        self,
        validator: RequestValidator,
        client: Client,
        from_: str,
    ) -> None:
        self._validator = validator
        self._client = client
        self._from = from_

    def parse_incoming(self, form: Mapping[str, str]) -> IncomingMessage:
        """Map Twilio's inbound form to a neutral domain message.

        Raises :class:`InvalidPayloadError` if mandatory fields are absent.
        """
        try:
            return IncomingMessage(
                sender=form[TwilioField.FROM],
                text=form[TwilioField.BODY],
                message_id=form[TwilioField.MESSAGE_SID],
            )
        except KeyError as exc:
            raise InvalidPayloadError(f"missing field: {exc.args[0]}") from exc

    async def send(self, to: str, text: str) -> None:
        """Send a WhatsApp text via the Twilio REST API (off the event loop)."""
        await to_thread.run_sync(self._send_sync, to, text)

    def _send_sync(self, to: str, text: str) -> None:
        self._client.messages.create(from_=self._from, to=to, body=text)

    def verify_signature(
        self,
        url: str,
        params: Mapping[str, str],
        signature: str,
    ) -> bool:
        """Validate Twilio's request signature against the public URL."""
        # Twilio's validate() is only partially typed; coerce its result to bool.
        result = self._validator.validate(url, params, signature)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        return bool(result)  # pyright: ignore[reportUnknownArgumentType]
