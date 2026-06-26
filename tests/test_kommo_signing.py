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


# A REAL inbound webhook body Kommo sent: note the trailing "\n" it transmits but
# does NOT include in the X-Signature (empirically confirmed against the live
# payload — the captured signature matched the HMAC of this body WITHOUT the "\n").
_INBOUND_BODY = (
    b'{"receiver":"wa-+5215583232460","conversation_id":"+5215583232460",'
    b'"msec_timestamp":1782436669124,"type":"text","text":"going","markup":null,'
    b'"tag":"","media":"","thumbnail":"","file_name":"","file_size":0,'
    b'"media_group_id":""}\n'
)


def test_verify_inbound_accepts_signature_over_body_without_trailing_newline() -> None:
    signer = KommoSigner(_SECRET)
    assert _INBOUND_BODY.endswith(b"\n")
    # Kommo signs the body WITHOUT the trailing "\n" it transmits.
    signature = signer.sign(_INBOUND_BODY[:-1])

    # Inbound verify strips one "\n" -> matches. The plain verify (over the body
    # as received, WITH the "\n") would NOT — that asymmetry caused the 401.
    assert signer.verify_inbound(_INBOUND_BODY, signature) is True
    assert signer.verify(_INBOUND_BODY, signature) is False


def test_verify_inbound_also_accepts_a_body_without_a_trailing_newline() -> None:
    signer = KommoSigner(_SECRET)
    body = _INBOUND_BODY[:-1]  # no trailing "\n"
    signature = signer.sign(body)

    assert not body.endswith(b"\n")
    # body[:-1] only applies when it ends with "\n", so a clean body still verifies.
    assert signer.verify_inbound(body, signature) is True


def test_verify_inbound_rejects_a_wrong_signature() -> None:
    signer = KommoSigner(_SECRET)

    assert signer.verify_inbound(_INBOUND_BODY, "deadbeef") is False


def test_outbound_headers_are_complete_and_consistent() -> None:
    signer = KommoSigner(_SECRET)

    headers = signer.outbound_headers(_BODY)

    assert headers[KommoHeader.CONTENT_TYPE] == "application/json"
    assert headers[KommoHeader.CONTENT_MD5] == hashlib.md5(_BODY).hexdigest()
    assert headers[KommoHeader.SIGNATURE] == signer.sign(_BODY)
    assert headers[KommoHeader.DATE]  # an RFC date is present
    # The header set verifies against the same raw body.
    assert signer.verify(_BODY, headers[KommoHeader.SIGNATURE]) is True
