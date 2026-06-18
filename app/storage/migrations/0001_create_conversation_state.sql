-- Durable conversation state: the source of truth for per-conversation state.
-- The whole ConversationState lives as a JSONB blob (no extracted columns;
-- trip_type etc. live inside the blob — YAGNI). Idempotent and directly
-- runnable in DBeaver for the initial Render apply.

CREATE TABLE IF NOT EXISTS conversation_state (
    conversation_id TEXT PRIMARY KEY,
    state           JSONB       NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
