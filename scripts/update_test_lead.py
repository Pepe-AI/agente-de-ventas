"""One-off: write custom fields to a REAL Top Viajes lead AND move its stage.

Mirror of scripts/add_test_note.py. Reads everything from os.environ (NOT Settings
— the long-lived token stays injected). Makes the REAL HTTPS PATCH to the Kommo
CRM API: run it manually once and verify in the Kommo UI that the lead shows the
custom fields written and has moved to the "Calificado" stage.

It exercises the full path: a sample slot state -> build_custom_fields_values (with
the Top Viajes mapping) -> update_lead (custom fields + status + pipeline in ONE
PATCH).

KOMMO_CRM_BASE_URL is the account subdomain, e.g. https://<account>.kommo.com.
KOMMO_TEST_LEAD_ID is the numeric lead id from any lead URL in the pipeline.

    python -m scripts.update_test_lead
"""

from __future__ import annotations

import asyncio
import os
import sys

from app.crm import kommo_mapping_topviajes as mapping
from app.crm.kommo_crm import KommoCrmClient, KommoCrmError
from app.crm.lead_payload import build_custom_fields_values
from app.domain.models import HandoffReason
from app.understanding.schemas import escape_slot_names

_REQUIRED_ENV = ("KOMMO_LONG_LIVED_TOKEN", "KOMMO_CRM_BASE_URL", "KOMMO_TEST_LEAD_ID")

# A sample Europe conversation state (incl. a structured Budget + Passengers).
_SAMPLE_SLOTS: dict[str, object] = {
    "nombre_cliente": "Cliente de prueba",  # excluded — lives on the contact
    "paises_europa": "Italia y Francia",
    "fechas_europa": "Septiembre 2026",
    "duracion_europa": "12 días",
    "pasajeros_europa": {"adults": 2, "minors_mentioned": True, "minor_ages": [10]},
    "presupuesto_europa": {"defer_to_advisor": True},
    "ciudad_salida_europa": "CDMX",
}


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

    token = os.environ["KOMMO_LONG_LIVED_TOKEN"]
    base_url = os.environ["KOMMO_CRM_BASE_URL"]

    custom_fields_values = build_custom_fields_values(
        _SAMPLE_SLOTS,
        slot_concepts=mapping.SLOT_CONCEPTS,
        concept_field_ids=mapping.CONCEPT_FIELD_IDS,
        escape_slots=escape_slot_names(),
    )
    status_id = mapping.REASON_STATUS_IDS[HandoffReason.COMPLETE]
    pipeline_id = mapping.PIPELINE_ID
    print(f"custom_fields_values: {custom_fields_values}")
    print(f"moving lead {lead_id} -> status {status_id} / pipeline {pipeline_id}")

    async with KommoCrmClient(token, base_url) as client:
        try:
            result = await client.update_lead(
                lead_id,
                custom_fields_values=custom_fields_values,
                status_id=status_id,
                pipeline_id=pipeline_id,
            )
        except KommoCrmError as exc:
            print(
                f"CRM call failed: {exc} (status={exc.status}, body={exc.body})",
                file=sys.stderr,
            )
            return 1

    print(result)  # the parsed Kommo response (updated lead)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
