"""Mirror an inbound (post-handoff) customer message into the linked Kommo chat (B2).

After handoff the bot is silent (the handoff flag), but the customer may keep writing
on WhatsApp; B2 forwards each such message INTO the Chats API chat that B1
created+linked, so the advisor sees it. It posts via ``send_message`` routed by the
``conversation_id`` (= the customer phone, B1's key — NOT the chat_id, which is B3's
key). The sender is the CUSTOMER (the external party) so Kommo shows it as INBOUND.

Isolated + testable: the message-sender client is injected, mirroring ChatConnector.
"""

from __future__ import annotations

import time
from typing import Protocol

from app.crm.kommo_chats import KommoChatMessage, KommoChatSender

# KommoChatSender requires an avatar; the live-validated create_chat body (B1) OMITTED
# it, so this value is UNVERIFIED for send_message — the Chats docs only show URL
# examples. Confirm in the live validation that Kommo accepts it (and renders sanely).
_MIRROR_AVATAR = "https://www.gravatar.com/avatar?d=mp"


class MessageSender(Protocol):
    """The Chats-API send_message primitive (KommoChatsClient satisfies it)."""

    async def send_message(
        self, scope_id: str, message: KommoChatMessage
    ) -> dict[str, object]: ...


class ChatMirror:
    """Posts a post-handoff customer message into the linked chat (inbound)."""

    def __init__(self, sender: MessageSender, scope_id: str) -> None:
        self._sender = sender
        self._scope_id = scope_id

    async def mirror_inbound(
        self, conversation_id: str, name: str, phone: str, text: str, msgid: str
    ) -> None:
        """Post the customer's ``text`` into the chat as an INBOUND message."""
        # sender.id MUST match the chat user id B1 used at create_chat
        # (orchestrator._handoff: KommoChatUser(id=f"wa-{phone}", ...)) — otherwise
        # Kommo attributes the message to a different participant in the same chat.
        sender = KommoChatSender(
            id=f"wa-{phone}", avatar=_MIRROR_AVATAR, name=name, phone=phone
        )
        message = KommoChatMessage(
            conversation_id=conversation_id,
            msgid=msgid,
            timestamp=int(time.time()),
            sender=sender,
            text=text,
        )
        await self._sender.send_message(self._scope_id, message)
