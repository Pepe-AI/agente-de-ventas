"""One-off: add a TEST note to a real Top Viajes lead, print the API result.

Mirror of scripts/send_test_message.py. Reads everything from os.environ (NOT
Settings — the long-lived token stays injected, avoiding coupling to get_settings
/ migrate.py). It makes the REAL HTTPS call to the Kommo CRM API: run it manually
once and see the note appear on the lead in the Kommo UI.

KOMMO_CRM_BASE_URL is the account subdomain, e.g. https://<account>.kommo.com.
KOMMO_TEST_LEAD_ID is the numeric lead id from any lead URL in the pipeline.

    python -m scripts.add_test_note
"""

from __future__ import annotations

import asyncio
import os
import sys

from app.crm.kommo_crm import KommoCrmClient, KommoCrmError

_REQUIRED_ENV = ("KOMMO_LONG_LIVED_TOKEN", "KOMMO_CRM_BASE_URL", "KOMMO_TEST_LEAD_ID")


async def _run() -> int:
    missing = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        print(f"missing env vars: {', '.join(missing)}", file=sys.stderr)
        return 2

    try:
        lead_id = int(os.environ["KOMMO_TEST_LEAD_ID"])
    except ValueError:
        print("KOMMO_TEST_LEAD_ID must be an integer", file=sys.stderr)
        return 2

    text = os.environ.get("KOMMO_NOTE_TEXT", "Nota de prueba desde el script")
    token = os.environ["KOMMO_LONG_LIVED_TOKEN"]
    base_url = os.environ["KOMMO_CRM_BASE_URL"]

    async with KommoCrmClient(token, base_url) as client:
        try:
            account = await client.get_account()  # smoke-test the token first
            print(f"account ok: id={account.get('id')}")
            result = await client.add_note(lead_id, text)
        except KommoCrmError as exc:
            print(
                f"CRM call failed: {exc} (status={exc.status}, body={exc.body})",
                file=sys.stderr,
            )
            return 1

    print(result)  # the parsed Kommo response (created note)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
