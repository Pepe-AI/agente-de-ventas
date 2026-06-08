"""Domain orchestration seam.

This is the single extension point where conversational logic (LLM, NLU,
slot-filling, ...) will be plugged in later. For this walking skeleton it only
echoes the inbound text back.
"""

from __future__ import annotations

from app.domain.models import IncomingMessage

ECHO_PREFIX = "Echo: "


def handle_message(msg: IncomingMessage) -> str:
    """Return the reply text for an inbound message.

    Currently a pure echo. Future increments replace the body with real
    conversational logic without changing the signature.
    """
    return f"{ECHO_PREFIX}{msg.text}"
