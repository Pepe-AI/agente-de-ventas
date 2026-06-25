"""Kommo Chats API HMAC signing/verification (pure: no I/O, no Settings).

Symmetric scheme (verified against Kommo's docs): ``X-Signature`` is the
HMAC-SHA1 of the RAW request body, keyed by the channel secret, as lowercase
hex. The same computation signs outbound requests and verifies inbound webhooks.

CRITICAL: always sign/verify the RAW body bytes — never a re-serialized JSON
(re-serializing changes key order/spacing -> different bytes -> broken signature).

The channel secret is INJECTED (constructor); it is never imported from Settings
nor logged, keeping this module pure and testable. The wiring (a SecretStr
setting + composition root) and the Chats API client / inbound webhook come next:
``outbound_headers`` feeds the client (connect / send / status) and ``verify``
guards the inbound webhook.

Note on Kommo's official example: it compares the raw body against the header
(instead of the computed signature) with a plain ``==`` (timing-unsafe). We do
NOT replicate that — we compare the COMPUTED signature in constant time.
"""

from __future__ import annotations

import hashlib
import hmac
from email.utils import formatdate
from enum import StrEnum

_CONTENT_TYPE_JSON = "application/json"


class KommoHeader(StrEnum):
    """Header names of an outbound, signed Kommo Chats API request."""

    DATE = "Date"
    CONTENT_TYPE = "Content-Type"
    CONTENT_MD5 = "Content-MD5"
    SIGNATURE = "X-Signature"


class KommoSigner:
    """Signs/verifies Kommo Chats API traffic with the channel secret (HMAC-SHA1)."""

    def __init__(self, channel_secret: str) -> None:
        # Kept private as bytes (HMAC key); never logged or exposed in repr.
        self._secret = channel_secret.encode("utf-8")

    def sign(self, body: bytes) -> str:
        """Return the HMAC-SHA1 of the raw ``body`` as lowercase hex."""
        return hmac.new(self._secret, body, hashlib.sha1).hexdigest()

    def outbound_headers(self, body: bytes) -> dict[str, str]:
        """Build the full signed header set for an outbound request over ``body``."""
        content_md5 = hashlib.md5(body, usedforsecurity=False).hexdigest()
        headers: dict[str, str] = {
            KommoHeader.DATE: formatdate(usegmt=True),
            KommoHeader.CONTENT_TYPE: _CONTENT_TYPE_JSON,
            KommoHeader.CONTENT_MD5: content_md5,
            KommoHeader.SIGNATURE: self.sign(body),
        }
        return headers

    def verify(self, body: bytes, signature: str) -> bool:
        """Return whether ``signature`` matches the HMAC of the raw ``body``.

        Constant-time comparison; case-insensitive on the hex digest; never raises
        on malformed input (an invalid signature simply fails verification).
        """
        try:
            return hmac.compare_digest(self.sign(body), signature.lower())
        except TypeError:
            return False  # e.g. a non-ASCII signature value
