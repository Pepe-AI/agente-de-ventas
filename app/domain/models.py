"""Neutral domain model for an inbound message.

Decoupled from any transport wire format (Twilio, Meta Cloud API, ...) so the
core logic never depends on a concrete provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class HandoffReason(StrEnum):
    """Why a conversation was handed to a human."""

    COMPLETE = "completa"  # all required slots captured
    STUCK = "atorado"  # gave up on a required slot after repeated failures
    HUMAN_REQUESTED = "pidió_humano"  # the user asked to talk to a person
    NO_RESPONSE = "no_respondió"  # went silent past the inactivity window (timer)


@dataclass(frozen=True, slots=True)
class Referral:
    """Click-to-WhatsApp (CTWA) ad referral attached to an inbound message.

    Neutral across providers; carries the ad's identifying metadata.
    """

    source_id: str
    headline: str
    body: str
    ctwa_clid: str


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    """A message received from an end user, normalized across channels."""

    sender: str
    text: str
    message_id: str
    referral: Referral | None = None
