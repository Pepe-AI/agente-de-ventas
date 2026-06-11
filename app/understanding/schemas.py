"""Response schemas for the understanding engine.

PLACEHOLDER for increment 3. ``DummyReservation`` is a throwaway booking schema
used to exercise the engine end to end; the real per-trip schemas
(cruise / Europe / Asia), selected by campaign, arrive in increment 4 (G2).

Convention: every field except ``question`` is a *slot*. ``question`` holds a
user question detected in the turn (or ``None``).
"""

from __future__ import annotations

from pydantic import BaseModel


class DummyReservation(BaseModel):
    """Placeholder booking slots + detected question (replace in increment 4)."""

    num_people: int | None = None
    travel_date: str | None = None
    has_id: bool | None = None
    question: str | None = None
