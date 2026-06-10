"""Domain orchestration seam.

This is the single extension point where conversational logic (LLM, NLU,
slot-filling, ...) will be plugged in later. For this walking skeleton it only
echoes the inbound text back.
"""

from __future__ import annotations

from app.domain.models import IncomingMessage

ECHO_PREFIX = "Echo: "
CAMPAIGN_SUFFIX = " [campaña: {campaign}]"


def handle_message(msg: IncomingMessage) -> str:
    """Return the reply text for an inbound message.

    Currently a pure echo, annotated with the CTWA campaign when the message
    comes from an ad. Future increments replace the body with real
    conversational logic without changing the signature.
    """
    reply = f"{ECHO_PREFIX}{msg.text}"
    if msg.referral is not None:
        campaign = msg.referral.headline or msg.referral.source_id
        reply += CAMPAIGN_SUFFIX.format(campaign=campaign)
    return reply
