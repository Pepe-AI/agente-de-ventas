"""Neutral domain model for an inbound message.

Decoupled from any transport wire format (Twilio, Meta Cloud API, ...) so the
core logic never depends on a concrete provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class HandoffReason(StrEnum):
    """Why a conversation was handed to a human.

    Only the happy-path reason exists today; retry/stuck/human-requested reasons
    arrive in later increments.
    """

    COMPLETE = "completa"


@dataclass(frozen=True, slots=True)
class HandoffEvent:
    """The payload of a handoff: why, which schema, and what was captured.

    Increment 8 maps this to a CRM funnel; for now it is passed to the relay
    stub. ``trip_type`` is the trip-type value (kept primitive to keep this
    neutral model free of schema imports).
    """

    reason: HandoffReason
    trip_type: str
    slots: dict[str, object]


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
