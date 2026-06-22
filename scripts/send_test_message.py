"""One-off: push a test message to a Kommo chat, print the API result.

Mirror of scripts/connect_kommo_channel.py. Reads everything from os.environ (NOT
Settings — the channel secret stays injected, avoiding coupling to get_settings /
migrate.py). It makes the REAL HTTPS call to Kommo: run it manually once and watch
the message appear in the Kommo inbox — this also CREATES the chat.

NOTE: KOMMO_SENDER_AVATAR must be a URL publicly reachable by Kommo (not localhost).

    python -m scripts.send_test_message
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid

from app.crm.kommo_chats import (
    KommoChatMessage,
    KommoChatsClient,
    KommoChatSender,
    KommoChatsError,
)
from app.crm.kommo_signing import KommoSigner

_REQUIRED_ENV = (
    "KOMMO_CHANNEL_ID",
    "KOMMO_CHANNEL_SECRET",
    "KOMMO_SCOPE_ID",
    "KOMMO_CONVERSATION_ID",
    "KOMMO_SENDER_ID",
    "KOMMO_SENDER_AVATAR",
    "KOMMO_SENDER_PHONE",
)


async def _run() -> int:
    missing = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        print(f"missing env vars: {', '.join(missing)}", file=sys.stderr)
        return 2

    # The caller (this script) provides every field — including a fresh msgid and
    # the current timestamp — so the client invents nothing.
    message = KommoChatMessage(
        conversation_id=os.environ["KOMMO_CONVERSATION_ID"],
        msgid=uuid.uuid4().hex,
        timestamp=int(time.time()),
        sender=KommoChatSender(
            id=os.environ["KOMMO_SENDER_ID"],
            avatar=os.environ["KOMMO_SENDER_AVATAR"],
            name=os.environ.get("KOMMO_SENDER_NAME", "Cliente de prueba"),
            phone=os.environ["KOMMO_SENDER_PHONE"],
        ),
        text=os.environ.get("KOMMO_TEXT", "Hola desde el script de prueba"),
    )

    signer = KommoSigner(os.environ["KOMMO_CHANNEL_SECRET"])
    async with KommoChatsClient(signer, os.environ["KOMMO_CHANNEL_ID"]) as client:
        try:
            result = await client.send_message(os.environ["KOMMO_SCOPE_ID"], message)
        except KommoChatsError as exc:
            print(
                f"send_message failed: {exc} (status={exc.status}, body={exc.body})",
                file=sys.stderr,
            )
            return 1

    print(result)  # the parsed Kommo response (new_message info)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
