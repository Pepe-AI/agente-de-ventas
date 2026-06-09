"""Plain config object for the concurrency layer.

Decouples the primitives/flush from pydantic ``Settings`` so they stay easy to
exercise in tests with small values.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings


@dataclass(frozen=True, slots=True)
class ConcurrencyConfig:
    """Thresholds, TTLs and windows for the concurrency layer."""

    debounce_window_s: float
    dedup_ttl_s: int
    lock_ttl_s: int
    rate_window_s: int
    rate_threshold: int
    block_cooldown_s: int
    buffer_max: int

    @classmethod
    def from_settings(cls, settings: Settings) -> ConcurrencyConfig:
        """Build the config from application settings."""
        return cls(
            debounce_window_s=settings.debounce_window_s,
            dedup_ttl_s=settings.dedup_ttl_s,
            lock_ttl_s=settings.lock_ttl_s,
            rate_window_s=settings.rate_window_s,
            rate_threshold=settings.rate_threshold,
            block_cooldown_s=settings.block_cooldown_s,
            buffer_max=settings.buffer_max,
        )
