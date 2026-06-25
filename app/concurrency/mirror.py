"""Background mirror of post-handoff customer messages into the Kommo chat (B2).

When a handed-off sender keeps writing, the webhook fast-acks and fires this off the
request path (preserving the inc-2 fast-ack: send_message is a real network call). It
loads the durable state, and ONLY if a chat was connected (``state.chat_id`` set, from
B1) posts the message INTO that chat as inbound; otherwise it skips with a warning
(it never reconnects the chat — that is B1's job at handoff).
"""

from __future__ import annotations

import asyncio

import structlog

from app.crm.kommo_chats import KommoChatsError
from app.domain.chat_mirror import ChatMirror
from app.domain.handoff_orchestration import phone_from_sender
from app.domain.models import IncomingMessage
from app.domain.state import StateStore

log = structlog.get_logger()

# Strong refs so fire-and-forget background tasks are not GC'd mid-flight.
_background_tasks: set[asyncio.Task[None]] = set()


def schedule_mirror(
    chat_mirror: ChatMirror | None, store: StateStore, msg: IncomingMessage
) -> None:
    """Fire-and-forget the inbound mirror (keeps the webhook ack fast)."""
    task = asyncio.create_task(_mirror(chat_mirror, store, msg))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _mirror(
    chat_mirror: ChatMirror | None, store: StateStore, msg: IncomingMessage
) -> None:
    sender = msg.sender
    # Degraded channel (no scope_id at boot): no chat to mirror into -> skip.
    if chat_mirror is None:
        log.warning(
            "mirror_skipped_no_chat", sender=sender, reason="channel_unavailable"
        )
        return
    # Everything below runs in a fire-and-forget task: any escaping exception (incl.
    # store.load) would become an orphaned "Task exception was never retrieved", so the
    # whole worker is guarded — expected CRM errors warn, anything else logs+traceback.
    try:
        state = await store.load(sender)
        # No chat connected for this sender (B1 degraded / not handed off) -> skip.
        if state is None or state.chat_id is None:
            log.warning("mirror_skipped_no_chat", sender=sender)
            return
        name = state.slots.get("nombre_cliente")
        phone = phone_from_sender(sender)
        customer_name = name if isinstance(name, str) and name.strip() else phone
        await chat_mirror.mirror_inbound(
            conversation_id=phone,  # B1's routing key (NOT chat_id)
            name=customer_name,
            phone=phone,
            text=msg.text,
            msgid=msg.message_id,  # the Twilio id -> natural idempotency on Kommo
        )
        log.info("mirror_sent", sender=sender)
    except KommoChatsError as exc:
        # Expected CRM/transport failure: the message just isn't forwarded (a Twilio
        # retry is deduped upstream, so no double-post). Warn, don't crash the task.
        log.warning("mirror_failed", sender=sender, error=str(exc))
    except Exception:
        # Unexpected bug: never let a background task die with an orphaned exception.
        log.exception("mirror_error", sender=sender)
