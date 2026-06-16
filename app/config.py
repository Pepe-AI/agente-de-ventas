"""Application configuration and shared constants.

Credentials are loaded exclusively from the environment / ``.env`` via
pydantic-settings. They never appear in source or logs.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.understanding.schemas import TripType


class HttpHeader(StrEnum):
    """HTTP header names used by the webhook (no magic strings)."""

    X_TWILIO_SIGNATURE = "X-Twilio-Signature"
    X_FORWARDED_PROTO = "X-Forwarded-Proto"
    HOST = "Host"


class Settings(BaseSettings):
    """Runtime settings sourced from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    twilio_account_sid: str
    twilio_auth_token: SecretStr
    twilio_whatsapp_from: str

    # Redis connection (Render internal URL, set in the dashboard).
    redis_url: str

    # LLM (Gemini) configuration.
    gemini_api_key: SecretStr
    llm_model: str = "gemini-3.5-flash"

    # Which trip schema a brand-new conversation starts on (campaign-based
    # selection arrives in a later increment).
    trip_type: TripType = TripType.CRUISE

    # Concurrency tunables (seconds / counts) with sensible defaults.
    debounce_window_s: float = 3.0
    dedup_ttl_s: int = 3600
    lock_ttl_s: int = 30
    rate_window_s: int = 10
    rate_threshold: int = 15
    block_cooldown_s: int = 600
    buffer_max: int = 10


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide singleton of the settings."""
    return Settings()  # type: ignore[call-arg]  # values come from the environment
