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


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide singleton of the settings."""
    return Settings()  # type: ignore[call-arg]  # values come from the environment
