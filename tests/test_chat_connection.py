"""Offline tests for ChatConnector (create_chat + link, injected clients mocked).

The two methods are SEPARATE on purpose: the caller persists the chat_id between
create and link, so a link failure never loses the chat_id (no recreation on retry).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from app.crm.kommo_chats import KommoChatUser
from app.domain.chat_connection import ChatConnector

_USER = KommoChatUser(id="wa-+1", name="Ana", phone="+1")


async def test_create_chat_delegates_with_injected_scope_id() -> None:
    creator = AsyncMock()
    creator.create_chat.return_value = "chat-uuid"
    linker = AsyncMock()
    connector = ChatConnector(creator, linker, "scope-1")

    chat_id = await connector.create_chat("+1", _USER)

    assert chat_id == "chat-uuid"
    creator.create_chat.assert_awaited_once_with("scope-1", "+1", _USER)
    linker.link_chat_to_contact.assert_not_awaited()  # link is a SEPARATE step


async def test_link_delegates_to_the_crm_linker() -> None:
    creator = AsyncMock()
    linker = AsyncMock()
    connector = ChatConnector(creator, linker, "scope-1")

    await connector.link(555, "chat-uuid")

    linker.link_chat_to_contact.assert_awaited_once_with(555, "chat-uuid")
    creator.create_chat.assert_not_awaited()
