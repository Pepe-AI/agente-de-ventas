"""Tests for the Gemini transient-error predicate (the provider-specific bit)."""

from __future__ import annotations

import httpx
from google.genai import errors

from app.llm.gemini import LLMError, is_transient_gemini_error


def test_server_error_5xx_is_transient() -> None:
    assert is_transient_gemini_error(errors.ServerError(503, {})) is True


def test_rate_limited_429_is_transient() -> None:
    assert is_transient_gemini_error(errors.ClientError(429, {})) is True


def test_timeout_is_transient() -> None:
    assert is_transient_gemini_error(httpx.TimeoutException("slow")) is True


def test_permanent_client_error_is_not_transient() -> None:
    assert is_transient_gemini_error(errors.ClientError(400, {})) is False


def test_unrelated_errors_are_not_transient() -> None:
    assert is_transient_gemini_error(LLMError("bad parse")) is False
    assert is_transient_gemini_error(ValueError("nope")) is False
