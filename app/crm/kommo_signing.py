"""Kommo Chats API HMAC signing/verification (pure: no I/O, no Settings).

``X-Signature`` is the HMAC-SHA1 of the request body, keyed by the channel
secret, as lowercase hex. OUTBOUND (``sign`` / ``outbound_headers``) and the
generic ``verify`` use the raw body bytes as-is. INBOUND is NOT symmetric: Kommo
transmits the webhook body with a trailing newline but signs it WITHOUT that
newline, so ``verify_inbound`` strips exactly one trailing ``b"\\n"`` first
(confirmed empirically against a real captured payload).

CRITICAL: always sign/verify the RAW body bytes — never a re-serialized JSON
(re-serializing changes key order/spacing -> different bytes -> broken signature).

The channel secret is INJECTED (constructor); it is never imported from Settings
nor logged, keeping this module pure and testable. ``outbound_headers`` feeds the
Chats API client (connect / send / status); ``verify_inbound`` guards the inbound
webhook.

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

    def verify_inbound(self, body: bytes, signature: str) -> bool:
        """Verify an INBOUND Kommo webhook signature (NOT symmetric with ``sign``).

        Kommo TRANSMITS the webhook body with a trailing newline but computes its
        ``X-Signature`` over the body WITHOUT it (confirmed empirically against a
        real captured payload). So strip exactly ONE trailing ``b"\\n"`` before
        verifying — never ``rstrip`` (which would drop every trailing newline and
        change the very bytes Kommo signed). A body without a trailing newline is
        verified as-is.
        """
        signed = body[:-1] if body.endswith(b"\n") else body
        return self.verify(signed, signature)
