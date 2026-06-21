"""Redis key namespace (no magic strings).

Every key is built through :func:`make_key` so prefixes stay centralized and
consistent across the concurrency primitives.
"""

from __future__ import annotations

from enum import StrEnum

_SEPARATOR = ":"


class KeyPrefix(StrEnum):
    """Prefixes for the Redis keys used by the concurrency layer."""

    BLOCKED = "blocked"
    DEDUP = "dedup"
    RATE = "rate"
    BUFFER = "buffer"
    DEBOUNCE = "debounce"
    LOCK = "lock"
    HANDOFF = "handoff"
    KOMMO_INBOUND = "kommo_inbound"


def make_key(prefix: KeyPrefix, identifier: str) -> str:
    """Build a namespaced Redis key, e.g. ``blocked:whatsapp:+123``."""
    return f"{prefix.value}{_SEPARATOR}{identifier}"
