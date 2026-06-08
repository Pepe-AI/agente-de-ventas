"""Tests for mapping Twilio's inbound form to the domain model."""

from __future__ import annotations

import pytest
from twilio.request_validator import RequestValidator
from twilio.rest import Client

from app.channels.twilio import InvalidPayloadError, TwilioChannel
from app.domain.models import IncomingMessage


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


def test_parse_incoming_rejects_missing_field() -> None:
    channel = _make_channel()
    form = {"From": "whatsapp:+5215512345678", "Body": "hola"}  # no MessageSid

    with pytest.raises(InvalidPayloadError):
        channel.parse_incoming(form)
