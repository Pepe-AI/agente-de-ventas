"""Postgres-backed StateStore (asyncpg, no ORM) — the durable source of truth.

State is stored as a single JSONB blob (no extracted columns). asyncpg does not
adapt ``dict`` <-> ``jsonb`` automatically, so (de)serialization is explicit:
``json.dumps`` on write with a ``::jsonb`` cast, ``json.loads`` on read (asyncpg
returns ``jsonb`` as its raw JSON text). The serialization itself is reused from
the domain (:func:`~app.domain.state.to_payload` / ``from_payload``).
"""

# asyncpg is the (loosely-typed) driver boundary for this module; relax the
# "unknown type" reports here, like the Twilio/Gemini adapters do at their edges.
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import json

import asyncpg

from app.domain.state import ConversationState, from_payload, to_payload

_UPSERT = """
INSERT INTO conversation_state (conversation_id, state, updated_at)
VALUES ($1, $2::jsonb, now())
ON CONFLICT (conversation_id)
DO UPDATE SET state = EXCLUDED.state, updated_at = now()
"""

_SELECT = "SELECT state FROM conversation_state WHERE conversation_id = $1"


class PostgresStateStore:
    """Durable conversation-state store backed by a JSONB column."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def load(self, conversation_id: str) -> ConversationState | None:
        # asyncpg returns a jsonb column as its raw JSON text (or None).
        raw = await self._pool.fetchval(_SELECT, conversation_id)
        if raw is None:
            return None
        return from_payload(json.loads(raw))

    async def save(self, conversation_id: str, state: ConversationState) -> None:
        await self._pool.execute(
            _UPSERT, conversation_id, json.dumps(to_payload(state))
        )


async def create_pool(dsn: str) -> asyncpg.Pool:
    """Create an asyncpg connection pool from ``dsn`` (Render's DATABASE_URL).

    SSL parameters in the DSN are honored by asyncpg as-is.
    """
    return await asyncpg.create_pool(dsn)
