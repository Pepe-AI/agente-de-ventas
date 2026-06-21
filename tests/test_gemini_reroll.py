"""Tests for GeminiLLM's one-shot re-roll on structured-output parse failure.

A non-conforming response leaves the SDK's ``.parsed`` as None; the adapter
re-rolls ONCE (stochastic, not load -> no backoff), then raises ``LLMError``.
No real LLM: the google-genai client is mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from app.llm.gemini import GeminiLLM, LLMError


class _Out(BaseModel):
    x: int = 0


class _Resp:
    """Stand-in for the SDK response; only ``.parsed`` matters here."""

    def __init__(self, parsed: object) -> None:
        self.parsed = parsed


def _gemini_returning(*responses: _Resp) -> tuple[GeminiLLM, AsyncMock]:
    client = MagicMock()
    gen = AsyncMock(side_effect=list(responses))
    client.aio.models.generate_content = gen
    return GeminiLLM(client, "model"), gen


async def test_success_on_first_try_does_not_reroll() -> None:
    llm, gen = _gemini_returning(_Resp(_Out(x=1)))

    result = await llm.complete_structured("p", _Out)

    assert result == _Out(x=1)
    assert gen.await_count == 1


async def test_reroll_once_on_parse_failure_then_succeeds() -> None:
    # First response does not conform (.parsed is None); the re-roll does.
    llm, gen = _gemini_returning(_Resp(None), _Resp(_Out(x=7)))

    result = await llm.complete_structured("p", _Out)

    assert result == _Out(x=7)
    assert gen.await_count == 2  # exactly one re-roll


async def test_parse_failure_twice_raises_and_does_not_loop() -> None:
    llm, gen = _gemini_returning(_Resp(None), _Resp(None))

    with pytest.raises(LLMError):
        await llm.complete_structured("p", _Out)

    assert gen.await_count == 2  # bounded: 1 try + 1 re-roll, no infinite loop
