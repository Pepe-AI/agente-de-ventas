"""Retrying decorator for the LLM port (provider-agnostic, reusable).

Wraps any :class:`~app.llm.base.LLM` and retries ONLY transient failures with
exponential backoff + jitter, bounded by ``max_attempts`` and a capped per-try
delay (so a turn never hangs for minutes). When retries are exhausted it raises
:class:`~app.llm.base.LLMUnavailableError`; a non-transient error is re-raised
immediately, unretried.

What counts as "transient" is the only provider-specific bit and is injected as
a predicate (see ``is_transient_gemini_error`` in ``gemini.py``), keeping this
decorator agnostic — it fits the replicable-core goal.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable

from app.llm.base import LLM, LLMUnavailableError, SchemaT

# Tunable defaults (sensible for a transient 503/429 from a hosted model).
_MAX_ATTEMPTS = 4
_BASE_DELAY_S = 0.5
_MAX_DELAY_S = 8.0


class RetryingLLM:
    """An :class:`LLM` that retries transient failures of an inner LLM."""

    def __init__(
        self,
        inner: LLM,
        is_transient: Callable[[BaseException], bool],
        *,
        max_attempts: int = _MAX_ATTEMPTS,
        base_delay_s: float = _BASE_DELAY_S,
        max_delay_s: float = _MAX_DELAY_S,
    ) -> None:
        self._inner = inner
        self._is_transient = is_transient
        self._max_attempts = max_attempts
        self._base_delay_s = base_delay_s
        self._max_delay_s = max_delay_s

    async def complete_structured(self, prompt: str, schema: type[SchemaT]) -> SchemaT:
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await self._inner.complete_structured(prompt, schema)
            except Exception as exc:
                if not self._is_transient(exc):
                    raise  # permanent (e.g. 400/auth): do not retry
                if attempt >= self._max_attempts:
                    raise LLMUnavailableError(
                        f"LLM unavailable after {attempt} attempts"
                    ) from exc
                await asyncio.sleep(self._backoff(attempt))
        raise AssertionError("unreachable")  # the loop always returns or raises

    def _backoff(self, attempt: int) -> float:
        """Capped exponential delay with additive jitter."""
        delay = min(self._max_delay_s, self._base_delay_s * (2 ** (attempt - 1)))
        return delay + random.uniform(0.0, self._base_delay_s)
