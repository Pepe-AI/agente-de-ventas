"""Transport port (hexagonal architecture).

The core depends on this abstraction, never on a concrete provider. Swapping
WhatsApp providers (e.g. Twilio -> Meta Cloud API) means writing another
adapter that satisfies this Protocol, with no changes to the domain or the
HTTP layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from app.domain.models import IncomingMessage


@runtime_checkable
class Channel(Protocol):
    """A bidirectional messaging transport."""

    def verify_signature(
        self, url: str, params: Mapping[str, str], signature: str
    ) -> bool:
        """Verify an inbound webhook request is authentic for this transport.

        Kept on the port (not just the adapter) so the HTTP layer can stay
        provider-agnostic: swapping transports never edits the endpoint. The
        ``url``/``params``/``signature`` shape suits HMAC-over-request schemes
        like Twilio's; generalize to a neutral request DTO if a future provider
        signs differently (e.g. raw-body HMAC).
        """
        ...

    def parse_incoming(self, form: Mapping[str, str]) -> IncomingMessage:
        """Map a provider's inbound webhook payload to a domain message."""
        ...

    async def send(self, to: str, text: str) -> None:
        """Deliver a text message to a recipient."""
        ...
