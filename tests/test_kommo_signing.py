"""Tests for the Kommo HMAC signer (pure: no I/O, no network, no real Kommo)."""

from __future__ import annotations

import hashlib
import hmac

from app.crm.kommo_signing import KommoHeader, KommoSigner

_SECRET = "channel-secret-123"
_BODY = b'{"event":"new_message","payload":{"text":"hola"}}'


def test_sign_is_deterministic_hmac_sha1_lowercase_hex() -> None:
    signer = KommoSigner(_SECRET)
    expected = hmac.new(_SECRET.encode(), _BODY, hashlib.sha1).hexdigest()

    assert signer.sign(_BODY) == expected  # HMAC-SHA1 of the raw body, hex
    assert signer.sign(_BODY) == signer.sign(_BODY)  # deterministic
    assert signer.sign(_BODY) == signer.sign(_BODY).lower()  # lowercase


def test_verify_accepts_its_own_signature() -> None:
    signer = KommoSigner(_SECRET)

    assert signer.verify(_BODY, signer.sign(_BODY)) is True


def test_verify_is_case_insensitive_on_the_received_signature() -> None:
    signer = KommoSigner(_SECRET)

    assert signer.verify(_BODY, signer.sign(_BODY).upper()) is True


def test_verify_rejects_a_tampered_body() -> None:
    signer = KommoSigner(_SECRET)
    signature = signer.sign(_BODY)

    assert signer.verify(_BODY + b" ", signature) is False


def test_verify_rejects_a_wrong_secret() -> None:
    signature = KommoSigner(_SECRET).sign(_BODY)

    assert KommoSigner("a-different-secret").verify(_BODY, signature) is False


def test_verify_does_not_raise_on_malformed_signature() -> None:
    signer = KommoSigner(_SECRET)

    assert signer.verify(_BODY, "ñó-not-ascii-hex") is False


def test_outbound_headers_are_complete_and_consistent() -> None:
    signer = KommoSigner(_SECRET)

    headers = signer.outbound_headers(_BODY)

    assert headers[KommoHeader.CONTENT_TYPE] == "application/json"
    assert headers[KommoHeader.CONTENT_MD5] == hashlib.md5(_BODY).hexdigest()
    assert headers[KommoHeader.SIGNATURE] == signer.sign(_BODY)
    assert headers[KommoHeader.DATE]  # an RFC date is present
    # The header set verifies against the same raw body.
    assert signer.verify(_BODY, headers[KommoHeader.SIGNATURE]) is True
