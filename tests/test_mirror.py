"""Offline tests for the B2 background mirror worker (_mirror): load state, gate on
state.chat_id, post (or skip+warn). The ChatMirror itself is mocked here."""

from __future__ import annotations

from unittest.mock import AsyncMock

from structlog.testing import capture_logs

from app.concurrency.mirror import _mirror
from app.domain.models import IncomingMessage
from app.domain.state import ConversationState
from app.understanding.schemas import TripType
from tests.fakes import InMemoryStateStore

SENDER = "whatsapp:+5215512345678"
PHONE = "+5215512345678"


def _msg(text: str = "¿siguen ahí?") -> IncomingMessage:
    return IncomingMessage(sender=SENDER, text=text, message_id="SM1")


async def _store_with(state: ConversationState | None) -> InMemoryStateStore:
    store = InMemoryStateStore()
    if state is not None:
        await store.save(SENDER, state)
    return store


async def test_mirror_posts_inbound_when_chat_id_present() -> None:
    store = await _store_with(
        ConversationState(
            trip_type=TripType.CRUISE,
            slots={"nombre_cliente": "Ana"},
            chat_id="chat-uuid",
        )
    )
    mirror = AsyncMock()

    await _mirror(mirror, store, _msg("hola de nuevo"))

    mirror.mirror_inbound.assert_awaited_once_with(
        conversation_id=PHONE,  # = phone (B1's key), NOT chat_id
        name="Ana",
        phone=PHONE,
        text="hola de nuevo",
        msgid="SM1",
    )


async def test_mirror_skips_and_warns_when_no_chat_id() -> None:
    store = await _store_with(
        ConversationState(trip_type=TripType.CRUISE, slots={"nombre_cliente": "Ana"})
    )  # chat_id is None (B1 degraded for this sender)
    mirror = AsyncMock()

    with capture_logs() as logs:
        await _mirror(mirror, store, _msg())

    mirror.mirror_inbound.assert_not_awaited()
    assert any(e["event"] == "mirror_skipped_no_chat" for e in logs)


async def test_mirror_skips_when_state_absent() -> None:
    store = await _store_with(None)
    mirror = AsyncMock()

    await _mirror(mirror, store, _msg())

    mirror.mirror_inbound.assert_not_awaited()


async def test_mirror_skips_and_warns_when_channel_degraded() -> None:
    store = await _store_with(
        ConversationState(trip_type=TripType.CRUISE, slots={}, chat_id="chat-uuid")
    )

    with capture_logs() as logs:
        await _mirror(None, store, _msg())  # chat_mirror None = channel unavailable

    assert any(e["event"] == "mirror_skipped_no_chat" for e in logs)


async def test_mirror_swallows_unexpected_error_and_logs() -> None:
    store = await _store_with(
        ConversationState(trip_type=TripType.CRUISE, slots={}, chat_id="chat-uuid")
    )
    mirror = AsyncMock()
    mirror.mirror_inbound.side_effect = RuntimeError("boom")  # not a KommoChatsError

    with capture_logs() as logs:
        await _mirror(mirror, store, _msg())  # must NOT raise out of the bg task

    assert any(e["event"] == "mirror_error" for e in logs)


async def test_mirror_name_falls_back_to_phone_when_unknown() -> None:
    store = await _store_with(
        ConversationState(trip_type=TripType.CRUISE, slots={}, chat_id="chat-uuid")
    )  # no nombre_cliente
    mirror = AsyncMock()

    await _mirror(mirror, store, _msg())

    _, kwargs = mirror.mirror_inbound.await_args
    assert kwargs["name"] == PHONE
