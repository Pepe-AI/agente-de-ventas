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

    # Kommo CRM API v4 long-lived Bearer token — a DIFFERENT credential from the
    # Chats channel secret above. OPTIONAL for the same reason (migrate.py builds
    # full Settings); the web app's lifespan fail-fast enforces presence at boot.
    kommo_long_lived_token: SecretStr | None = None

    # Kommo CRM API v4 base URL (the account subdomain, e.g.
    # https://<account>.kommo.com). OPTIONAL like the token; the web app's lifespan
    # fail-fast enforces presence at boot. Not a secret, so a plain str.
    kommo_crm_base_url: str | None = None

    # Kommo Chats API ids for OUTBOUND chat connection (B1): the custom channel id,
    # and the account's amoCRM (amojo) id that connect() uses to derive the scope_id
    # at boot. OPTIONAL like the rest (not required -> migrate.py keeps working); the
    # lifespan fail-fast enforces presence, and scope_id is derived (never stored).
    kommo_channel_id: str | None = None
    kommo_amojo_id: str | None = None

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
    # Hard ceiling on how long one burst is buffered. The flush fires at
    # min(last_message + debounce_window_s, first_message + max_buffer_wait_s), so
    # a nonstop typer is still processed at the cap. Anchored to the first message.
    max_buffer_wait_s: float = 12.0
    dedup_ttl_s: int = 3600
    lock_ttl_s: int = 30
    rate_window_s: int = 10
    rate_threshold: int = 15
    block_cooldown_s: int = 600
    buffer_max: int = 10

    # Inactivity timer (handoff 4th reason "No respondió"). Defaults = production;
    # overridable per deployment via env to validate the timer live (e.g. lower
    # them in the Render dashboard) without committing test values. Read once per
    # process (get_settings is cached) -> a change needs a restart/redeploy.
    inactivity_deadline_s: float = 7200.0  # 2h of customer silence -> auto-handoff
    sweep_interval_s: float = 300.0  # seconds between inactivity sweeps


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide singleton of the settings."""
    return Settings()  # type: ignore[call-arg]  # values come from the environment
