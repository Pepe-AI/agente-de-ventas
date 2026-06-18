"""Postgres integration tests for the StateStore + migration runner.

SKIPPED unless ``TEST_DATABASE_URL`` is set. Point it at a DEDICATED test
Postgres (never production). Not run in the default suite (no DB available).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import asyncpg
import pytest
import pytest_asyncio

from app.domain.state import ConversationState
from app.storage.migrate import run_migrations
from app.storage.postgres import PostgresStateStore, create_pool
from app.understanding.schemas import TripType

_DSN = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    _DSN is None,
    reason="set TEST_DATABASE_URL (a dedicated TEST Postgres, never prod) to run",
)

CID = "whatsapp:+5215599999999"


@pytest_asyncio.fixture
async def pool() -> AsyncIterator[asyncpg.Pool]:
    assert _DSN is not None
    created = await create_pool(_DSN)
    await run_migrations(created)
    async with created.acquire() as conn:
        await conn.execute(
            "DELETE FROM conversation_state WHERE conversation_id = $1", CID
        )
    try:
        yield created
    finally:
        await created.close()


async def test_round_trip(pool: asyncpg.Pool) -> None:
    store = PostgresStateStore(pool)
    state = ConversationState(
        trip_type=TripType.CRUISE, slots={"nombre_cliente": "Ana", "n": {"a": 1}}
    )

    await store.save(CID, state)

    assert await store.load(CID) == state


async def test_load_absent_returns_none(pool: asyncpg.Pool) -> None:
    store = PostgresStateStore(pool)

    assert await store.load("whatsapp:+absent") is None


async def test_upsert_overwrites_and_keeps_one_row(pool: asyncpg.Pool) -> None:
    store = PostgresStateStore(pool)
    await store.save(CID, ConversationState(trip_type=TripType.CRUISE))

    await store.save(CID, ConversationState(trip_type=TripType.EUROPE))

    loaded = await store.load(CID)
    assert loaded is not None and loaded.trip_type is TripType.EUROPE
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT count(*) FROM conversation_state WHERE conversation_id = $1", CID
        )
    assert count == 1  # upsert, not a second insert


async def test_update_bumps_updated_at(pool: asyncpg.Pool) -> None:
    store = PostgresStateStore(pool)
    await store.save(CID, ConversationState(trip_type=TripType.CRUISE))
    async with pool.acquire() as conn:
        first = await conn.fetchval(
            "SELECT updated_at FROM conversation_state WHERE conversation_id = $1", CID
        )

    await store.save(CID, ConversationState(trip_type=TripType.EUROPE))

    async with pool.acquire() as conn:
        second = await conn.fetchval(
            "SELECT updated_at FROM conversation_state WHERE conversation_id = $1", CID
        )
    assert second >= first


async def test_state_column_is_jsonb(pool: asyncpg.Pool) -> None:
    store = PostgresStateStore(pool)
    await store.save(CID, ConversationState(trip_type=TripType.ASIA))

    async with pool.acquire() as conn:
        typename = await conn.fetchval(
            "SELECT pg_typeof(state)::text FROM conversation_state "
            "WHERE conversation_id = $1",
            CID,
        )
    assert typename == "jsonb"


async def test_runner_is_idempotent_and_records_versions(
    pool: asyncpg.Pool,
) -> None:
    # The fixture already applied migrations; a second run applies nothing.
    applied = await run_migrations(pool)

    assert applied == []
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT version FROM schema_migrations")
    assert any(str(row["version"]).startswith("0001") for row in rows)
