"""One-off: find a contact by phone, then create a test lead+contact, for the UI.

Mirror of scripts/add_test_note.py. Reads everything from os.environ (NOT Settings
— the long-lived token stays injected). Makes REAL HTTPS calls to the Kommo CRM
API: run it manually once and verify in the Kommo UI.
  (a) searches a phone and prints the matching contacts + their lead ids;
  (b) creates a TEST lead + contact (phone set) and prints the new lead id.

KOMMO_CRM_BASE_URL is the account subdomain, e.g. https://<account>.kommo.com.

    python -m scripts.find_create_lead
"""

from __future__ import annotations

import asyncio
import os
import sys

from app.crm.kommo_crm import KommoCrmClient, KommoCrmError

_REQUIRED_ENV = ("KOMMO_LONG_LIVED_TOKEN", "KOMMO_CRM_BASE_URL", "KOMMO_TEST_PHONE")


async def _run() -> int:
    missing = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        print(f"missing env vars: {', '.join(missing)}", file=sys.stderr)
        return 2

    phone = os.environ["KOMMO_TEST_PHONE"]
    lead_name = os.environ.get("KOMMO_TEST_LEAD_NAME", "Lead de prueba (API)")
    contact_name = os.environ.get("KOMMO_TEST_CONTACT_NAME", "Cliente de prueba")
    token = os.environ["KOMMO_LONG_LIVED_TOKEN"]
    base_url = os.environ["KOMMO_CRM_BASE_URL"]

    async with KommoCrmClient(token, base_url) as client:
        try:
            matches = await client.find_contact_by_phone(phone)
            print(f"matches for {phone}: {matches}")
            lead_id = await client.create_lead_with_contact(
                lead_name, contact_name, phone
            )
        except KommoCrmError as exc:
            print(
                f"CRM call failed: {exc} (status={exc.status}, body={exc.body})",
                file=sys.stderr,
            )
            return 1

    print(f"created lead_id={lead_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
