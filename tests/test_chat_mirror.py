"""Offline tests for ChatMirror (the post-handoff inbound mirror, send_message mocked).

It posts the customer's message INTO the existing chat, routed by conversation_id
(= the customer phone, B1's key), with the sender = the CUSTOMER (inbound). The
sender.id MUST match the chat user id B1 used at create_chat.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from app.crm.kommo_chats import KommoChatMessage
from app.domain.chat_mirror import _MIRROR_AVATAR, ChatMirror

_PHONE = "+5215512345678"


async def test_mirror_inbound_posts_customer_message_by_conversation_id() -> None:
    client = AsyncMock()
    mirror = ChatMirror(client, "scope-1")

    await mirror.mirror_inbound(
        conversation_id=_PHONE,
        name="Ana",
        phone=_PHONE,
        text="hola, ¿siguen ahí?",
        msgid="SM999",
    )

    client.send_message.assert_awaited_once()
    scope_id, message = client.send_message.await_args.args
    assert isinstance(message, KommoChatMessage)
    assert scope_id == "scope-1"
    assert message.conversation_id == _PHONE  # B1's routing key (phone), NOT chat_id
    assert message.msgid == "SM999"  # the Twilio id -> natural idempotency on Kommo
    assert message.text == "hola, ¿siguen ahí?"
    assert isinstance(message.timestamp, int)
    # The sender is the CUSTOMER (so Kommo shows it inbound); its id MUST equal the
    # chat user id B1 created the chat with (orchestrator: KommoChatUser id wa-{phone}).
    assert message.sender.id == f"wa-{_PHONE}"
    assert message.sender.name == "Ana"
    assert message.sender.phone == _PHONE
    assert message.sender.avatar == _MIRROR_AVATAR
