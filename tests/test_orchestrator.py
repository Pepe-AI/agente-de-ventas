"""Tests for the domain orchestration seam (handle_message)."""

from __future__ import annotations

from app.domain.models import IncomingMessage, Referral
from app.domain.orchestrator import handle_message


def test_handle_message_plain_echo_without_referral() -> None:
    msg = IncomingMessage(sender="whatsapp:+1", text="hola", message_id="SM1")

    assert handle_message(msg) == "Echo: hola"


def test_handle_message_includes_campaign_headline() -> None:
    msg = IncomingMessage(
        sender="whatsapp:+1",
        text="hola",
        message_id="SM1",
        referral=Referral(
            source_id="ad-987",
            headline="Crucero Mediterráneo",
            body="7 noches",
            ctwa_clid="ctwa-1",
        ),
    )

    assert handle_message(msg) == "Echo: hola [campaña: Crucero Mediterráneo]"


def test_handle_message_falls_back_to_source_id_without_headline() -> None:
    msg = IncomingMessage(
        sender="whatsapp:+1",
        text="hola",
        message_id="SM1",
        referral=Referral(source_id="ad-987", headline="", body="", ctwa_clid=""),
    )

    assert handle_message(msg) == "Echo: hola [campaña: ad-987]"
