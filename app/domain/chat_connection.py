"""Chat connection: create a Chats API chat and link it to the existing contact.

Isolated + testable with the clients INJECTED, mirroring ``HandoffRunner``. The two
steps are SEPARATE methods (NOT one atomic call) on purpose: the caller (_handoff)
persists the ``chat_id`` BETWEEN them, so a link failure after a successful create
never loses the chat_id — a retry then re-links instead of recreating the chat.

Linking the chat to the EXISTING contact (CRM API) is what stops Kommo from creating
a duplicate Incoming lead (the contact already exists; the lead is in Calificado).
"""

from __future__ import annotations

from typing import Protocol

from app.crm.kommo_chats import KommoChatUser


class ChatCreator(Protocol):
    """The Chats-API create_chat primitive (KommoChatsClient satisfies it)."""

    async def create_chat(
        self, scope_id: str, conversation_id: str, user: KommoChatUser
    ) -> str: ...


class ChatLinker(Protocol):
    """The CRM link_chat_to_contact primitive (KommoCrmClient satisfies it)."""

    async def link_chat_to_contact(self, contact_id: int, chat_id: str) -> object: ...


class ChatConnector:
    """Creates a chat (Chats API) and links it to a contact (CRM API).

    Holds the per-channel ``scope_id`` (resolved once at boot) so the caller does
    not thread it. Clients are injected — offline-testable with mocks.
    """

    def __init__(
        self, creator: ChatCreator, linker: ChatLinker, scope_id: str
    ) -> None:
        self._creator = creator
        self._linker = linker
        self._scope_id = scope_id

    async def create_chat(self, conversation_id: str, user: KommoChatUser) -> str:
        """Create the chat (Chats API); return its ``chat_id``."""
        return await self._creator.create_chat(self._scope_id, conversation_id, user)

    async def link(self, contact_id: int, chat_id: str) -> None:
        """Link the chat to the EXISTING contact (CRM API).

        ASSUMED idempotent — re-linking the same chat<->contact must be a safe no-op
        (it runs on every retried turn after the first). Verified by the live
        validation's double-link assertion. NOTE for B3: the inbound webhook will map
        back to the customer by ``conversation_id`` OR ``chat_id`` (whichever the v2
        payload carries — confirmed in B3); that is why both are persisted.
        """
        await self._linker.link_chat_to_contact(contact_id, chat_id)
