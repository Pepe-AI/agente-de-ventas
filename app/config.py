"""Application configuration and shared constants.

Credentials are loaded exclusively from the environment / ``.env`` via
pydantic-settings. They never appear in source or logs.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Postgres connection (Render's DATABASE_URL): durable state source of truth.
    database_url: str

    # LLM (Gemini) configuration. Default to a GA-stable model so a deploy
    # without the env var does not fall back to a fragile preview model.
    gemini_api_key: SecretStr
    llm_model: str = "gemini-2.5-flash"

    # Kommo Chats API channel secret (HMAC key for in/outbound signing). OPTIONAL
    # on purpose: a required field would force the migration runner (which calls
    # get_settings) to demand it too. The Kommo webhook DI validates presence at
    # request time instead.
    kommo_channel_secret: SecretStr | None = None

    # Campaign pre-fill phrases for trip-type routing, one per type. PLACEHOLDERS
    # until the client delivers the real campaign copy (G1); when set, a phrase
    # found in the first message routes to that trip type.
    prefill_crucero: str | None = None
    prefill_europa: str | None = None
    prefill_asia: str | None = None

    # Knowledge corpus for the CAG answerer, loaded once at startup.
    corpus_path: str = "app/corpus_topviajes.md"

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
