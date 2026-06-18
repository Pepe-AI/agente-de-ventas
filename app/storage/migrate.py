"""Idempotent SQL migration runner.

Applies pending ``migrations/*.sql`` in filename order and records each in
``schema_migrations``. Idempotent: re-running applies nothing already recorded.

NOT auto-run at startup (that would race across instances). It is invoked as a
command — ``python -m app.storage.migrate`` — and HOW Render runs it on deploy
is increment 9. The initial Render apply is done by hand via DBeaver with the
``.sql`` files (which are directly runnable).
"""

# asyncpg driver boundary: relax the "unknown type" reports (see postgres.py).
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from pathlib import Path

import asyncpg

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_ENSURE_LEDGER = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


async def run_migrations(pool: asyncpg.Pool) -> list[str]:
    """Apply pending migrations and return the versions newly applied."""
    newly_applied: list[str] = []
    async with pool.acquire() as conn:
        await conn.execute(_ENSURE_LEDGER)
        rows = await conn.fetch("SELECT version FROM schema_migrations")
        applied = {str(row["version"]) for row in rows}

        for path in sorted(_MIGRATIONS_DIR.glob("*.sql")):
            version = path.stem
            if version in applied:
                continue
            sql = path.read_text(encoding="utf-8")
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES ($1)", version
                )
            newly_applied.append(version)
    return newly_applied


async def _main() -> None:
    from app.config import get_settings

    pool = await asyncpg.create_pool(get_settings().database_url)
    try:
        applied = await run_migrations(pool)
        print(f"migrations applied: {applied or 'none (up to date)'}")
    finally:
        await pool.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(_main())
