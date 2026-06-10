"""Tests for mapping Twilio's inbound form to the domain model."""

from __future__ import annotations

import pytest
from twilio.request_validator import RequestValidator
from twilio.rest import Client

from app.channels.twilio import InvalidPayloadError, TwilioChannel
from app.domain.models import IncomingMessage, Referral


def _make_channel() -> TwilioChannel:
    # parse_incoming uses none of these collaborators; construct lightweight ones.
    return TwilioChannel(
        validator=RequestValidator("test_token"),
        client=Client("ACtest", "test_token"),
        from_="whatsapp:+14155238886",
    )


def test_parse_incoming_maps_twilio_form() -> None:
    channel = _make_channel()
    form = {
        "From": "whatsapp:+5215512345678",
        "Body": "hola",
        "MessageSid": "SM123",
        "NumMedia": "0",  # extra Twilio fields are ignored
    }

    msg = channel.parse_incoming(form)

    assert msg == IncomingMessage(
        sender="whatsapp:+5215512345678",
        text="hola",
        message_id="SM123",
    )
    assert msg.referral is None


def test_parse_incoming_populates_referral_for_ctwa_ad() -> None:
    channel = _make_channel()
    form = {
        "From": "whatsapp:+5215512345678",
        "Body": "quiero info",
        "MessageSid": "SM123",
        "ReferralSourceId": "ad-987",
        "ReferralHeadline": "Crucero por el Mediterráneo",
        "ReferralBody": "7 noches desde $999",
        "ReferralCtwaClid": "ctwa-abc-123",
    }

    msg = channel.parse_incoming(form)

    assert msg.referral == Referral(
        source_id="ad-987",
        headline="Crucero por el Mediterráneo",
        body="7 noches desde $999",
        ctwa_clid="ctwa-abc-123",
    )


def test_parse_incoming_referral_none_without_source_id() -> None:
    channel = _make_channel()
    # Referral fields other than the source id are ignored without the indicator.
    form = {
        "From": "whatsapp:+5215512345678",
        "Body": "hola",
        "MessageSid": "SM123",
        "ReferralHeadline": "stray field",
    }

    msg = channel.parse_incoming(form)

    assert msg.referral is None


def test_parse_incoming_rejects_missing_field() -> None:
    channel = _make_channel()
    form = {"From": "whatsapp:+5215512345678", "Body": "hola"}  # no MessageSid

    with pytest.raises(InvalidPayloadError):
        channel.parse_incoming(form)
