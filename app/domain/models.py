"""Neutral domain model for an inbound message.

Decoupled from any transport wire format (Twilio, Meta Cloud API, ...) so the
core logic never depends on a concrete provider.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    """A message received from an end user, normalized across channels."""

    sender: str
    text: str
    message_id: str
