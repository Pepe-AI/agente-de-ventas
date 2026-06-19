"""Tests for the RetryingLLM port decorator (no real LLM; a fake controls raises)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.llm.base import LLMUnavailableError
from app.llm.retry import RetryingLLM


class _Out(BaseModel):
    pass


class _Transient(Exception):
    pass


class _Permanent(Exception):
    pass


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, _Transient)


class FlakyLLM:
    """Raises the queued exceptions on successive calls, then succeeds."""

    def __init__(self, *raises: Exception) -> None:
        self._raises = list(raises)
        self.calls = 0

    async def complete_structured(
        self, prompt: str, schema: type[BaseModel]
    ) -> BaseModel:
        self.calls += 1
        if self._raises:
            raise self._raises.pop(0)
        return schema()


def _retrying(inner: FlakyLLM) -> RetryingLLM:
    # Zero delays keep the test fast (no real backoff sleeps).
    return RetryingLLM(inner, _is_transient, base_delay_s=0.0, max_delay_s=0.0)


async def test_retries_transient_then_succeeds() -> None:
    inner = FlakyLLM(_Transient(), _Transient())  # 2 transient failures, then ok

    result = await _retrying(inner).complete_structured("p", _Out)

    assert isinstance(result, _Out)
    assert inner.calls == 3


async def test_persistent_transient_raises_unavailable_after_max_attempts() -> None:
    inner = FlakyLLM(*[_Transient() for _ in range(10)])  # never recovers

    with pytest.raises(LLMUnavailableError):
        await _retrying(inner).complete_structured("p", _Out)

    assert inner.calls == 4  # default max_attempts; not infinite


async def test_permanent_error_is_not_retried() -> None:
    inner = FlakyLLM(_Permanent())

    with pytest.raises(_Permanent):
        await _retrying(inner).complete_structured("p", _Out)

    assert inner.calls == 1  # re-raised immediately, no retry
