"""One-off: connect this app's custom channel to a Kommo account, print scope_id.

Reads KOMMO_CHANNEL_ID, KOMMO_AMOJO_ID and KOMMO_CHANNEL_SECRET from the
environment (NOT Settings — the channel secret stays injected, so this avoids
coupling to get_settings / migrate.py). It makes the REAL HTTPS call to Kommo;
run it manually once. It is not part of the request path.

    python -m scripts.connect_kommo_channel
"""

from __future__ import annotations

import asyncio
import os
import sys

from app.crm.kommo_chats import KommoChatsClient, KommoChatsError
from app.crm.kommo_signing import KommoSigner

_REQUIRED_ENV = ("KOMMO_CHANNEL_ID", "KOMMO_AMOJO_ID", "KOMMO_CHANNEL_SECRET")


async def _run() -> int:
    missing = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        print(f"missing env vars: {', '.join(missing)}", file=sys.stderr)
        return 2

    signer = KommoSigner(os.environ["KOMMO_CHANNEL_SECRET"])
    async with KommoChatsClient(signer, os.environ["KOMMO_CHANNEL_ID"]) as client:
        try:
            scope_id = await client.connect(os.environ["KOMMO_AMOJO_ID"])
        except KommoChatsError as exc:
            print(
                f"connect failed: {exc} (status={exc.status}, body={exc.body})",
                file=sys.stderr,
            )
            return 1

    print(scope_id)  # the scope_id used later to send/receive messages
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
