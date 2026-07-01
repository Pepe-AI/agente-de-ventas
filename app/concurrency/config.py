"""Plain config object for the concurrency layer.

Decouples the primitives/flush from pydantic ``Settings`` so they stay easy to
exercise in tests with small values.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings


@dataclass(frozen=True, slots=True)
class ConcurrencyConfig:
    """Settings-derived knobs threaded to the message-processing path.

    Mostly concurrency thresholds/TTLs/windows; also carries the orchestrator's
    inactivity deadline (a per-turn timer armed in the same path) so flush can pass
    it to ``handle_message`` without a separate thread.
    """

    debounce_window_s: float
    max_buffer_wait_s: float
    dedup_ttl_s: int
    lock_ttl_s: int
    rate_window_s: int
    rate_threshold: int
    block_cooldown_s: int
    buffer_max: int
    inactivity_deadline_s: float

    @classmethod
    def from_settings(cls, settings: Settings) -> ConcurrencyConfig:
        """Build the config from application settings."""
        return cls(
            debounce_window_s=settings.debounce_window_s,
            max_buffer_wait_s=settings.max_buffer_wait_s,
            dedup_ttl_s=settings.dedup_ttl_s,
            lock_ttl_s=settings.lock_ttl_s,
            rate_window_s=settings.rate_window_s,
            rate_threshold=settings.rate_threshold,
            block_cooldown_s=settings.block_cooldown_s,
            buffer_max=settings.buffer_max,
            inactivity_deadline_s=settings.inactivity_deadline_s,
        )
