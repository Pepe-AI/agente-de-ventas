"""Campaign-based trip-type routing (pure, deterministic classifier).

A new conversation has no trip type yet; this picks one from the first message
(or the disambiguation reply) via a deterministic cascade, falling back to
``None`` (indeterminate) so the orchestrator can ask the user.

No LLM, no I/O: :func:`classify_trip_type` is a pure function and the unit of
truth for keyword matching (reused for both the first message and the
disambiguation reply).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.config import Settings
from app.domain.models import Referral
from app.understanding.schemas import TripType

# Keyword stems, already normalized (lowercase, no accents). Adjustable: tune
# these as real campaign copy lands. They are matched at a word boundary on the
# left (a stem/prefix), so a stem inside an unrelated word does not match
# (e.g. "asia" in "gimnasia", "barco" in "embarco").
_KEYWORD_STEMS: dict[TripType, tuple[str, ...]] = {
    TripType.CRUISE: ("crucer", "barco"),
    TripType.EUROPE: ("europ",),
    TripType.ASIA: ("asia", "asiat"),
}

# Precompiled once: each pattern matches any of its stems at a word start.
_KEYWORD_PATTERNS: dict[TripType, re.Pattern[str]] = {
    trip_type: re.compile(r"\b(?:" + "|".join(re.escape(s) for s in stems) + r")")
    for trip_type, stems in _KEYWORD_STEMS.items()
}


@dataclass(frozen=True, slots=True)
class RoutingConfig:
    """Configurable pre-fill phrases, one per trip type (placeholders until G1).

    Decoupled from pydantic ``Settings`` so the classifier stays trivially
    testable. The keyword stems are stable and live in this module, not here.
    """

    prefill_crucero: str | None
    prefill_europa: str | None
    prefill_asia: str | None

    @classmethod
    def from_settings(cls, settings: Settings) -> RoutingConfig:
        """Build the routing config from application settings."""
        return cls(
            prefill_crucero=settings.prefill_crucero,
            prefill_europa=settings.prefill_europa,
            prefill_asia=settings.prefill_asia,
        )


def _normalize(text: str) -> str:
    """Lowercase, strip accents, and trim — for substring matching."""
    lowered = text.strip().lower()
    decomposed = unicodedata.normalize("NFKD", lowered)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _match_keywords(normalized: str) -> TripType | None:
    for trip_type, pattern in _KEYWORD_PATTERNS.items():
        if pattern.search(normalized):
            return trip_type
    return None


def classify_trip_type(
    text: str, referral: Referral | None, config: RoutingConfig
) -> TripType | None:
    """Classify the trip type from a turn, or ``None`` if indeterminate.

    Cascade (first match wins): (a) a configured pre-fill phrase in the text,
    (b) a keyword stem in the text, (c) a keyword stem in the referral's
    headline/body. ``referral.source_id`` / ``ctwa_clid`` are NOT used for
    routing — they are kept for CRM attribution later (inc 8).
    """
    normalized = _normalize(text)

    # (a) campaign pre-fill phrase.
    prefill: dict[TripType, str | None] = {
        TripType.CRUISE: config.prefill_crucero,
        TripType.EUROPE: config.prefill_europa,
        TripType.ASIA: config.prefill_asia,
    }
    for trip_type, phrase in prefill.items():
        if phrase and _normalize(phrase) in normalized:
            return trip_type

    # (b) keyword stem in the message text.
    by_text = _match_keywords(normalized)
    if by_text is not None:
        return by_text

    # (c) keyword stem in the referral (ad headline/body) as a fallback.
    if referral is not None:
        ad_text = _normalize(f"{referral.headline} {referral.body}")
        by_referral = _match_keywords(ad_text)
        if by_referral is not None:
            return by_referral

    return None
