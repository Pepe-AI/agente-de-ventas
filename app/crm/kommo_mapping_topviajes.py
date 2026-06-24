"""Top Viajes' Kommo CRM mapping — the PER-CLIENT data artifact.

This module holds ONLY data, no logic: the numeric IDs are specific to the Top
Viajes Kommo account. Another client gets its own copy of this file (the same
``_topviajes`` per-client convention as ``corpus_topviajes.md``); the logic that
consumes it (``app/crm/lead_payload.py``, the client, the orchestration) is core
and shared. Keeping the IDs in typed Python — not JSON/env — lets pyright and the
mapping-integrity test catch a missing concept or a fat-fingered id at check time.

``Concept`` and the slot->concept normalization (``SLOT_CONCEPTS``) are the shared,
schema-derived grouping and live in the domain (``app.domain.concepts``); only the
account-specific IDs below are per-client (CRM -> domain, never the reverse).
"""

from __future__ import annotations

from app.domain.concepts import Concept
from app.domain.models import HandoffReason

# Concept -> Kommo lead custom field id (all type "text"). The native price
# ("Presupuesto") field is the advisor's and is NOT here; the client's stated
# budget goes to INVERSION (1114102). The duplicate "Fecha" field 1112710 is
# intentionally ignored.
CONCEPT_FIELD_IDS: dict[Concept, int] = {
    Concept.DESTINO: 1112708,
    Concept.FECHA: 1112714,
    Concept.DURACION: 1112716,
    Concept.PASAJEROS: 1112718,
    Concept.INVERSION: 1114102,
    Concept.CIUDAD_SALIDA: 1112720,
    Concept.SERVICIOS: 1112712,
    Concept.NIVEL_HOSPEDAJE: 1112722,
    Concept.VUELOS: 1112724,
    Concept.OCASION: 1112726,
    Concept.DOCUMENTACION: 1112730,
    Concept.CABINAS: 1114110,
    Concept.TIPO_CABINA: 1114112,
    Concept.EXPERIENCIA_CRUCERO: 1114114,
}


# Pipeline "Embudo de ventas" and the stage a handoff moves the lead to.
PIPELINE_ID = 13937935

# First stage "Incoming leads" of PIPELINE_ID: where a lead created without an
# explicit status lands. A REUSED lead still sitting here is treated as new
# (unpublished) and gets moved; a reused lead in any other stage is left where the
# advisor placed it. (Assumes API-created-without-status leads land here.)
INCOMING_STATUS_ID = 107559931

# Handoff reason -> status_id (stage) within PIPELINE_ID. atorado and pidió_humano
# share the "Atención 1 a 1" stage.
REASON_STATUS_IDS: dict[HandoffReason, int] = {
    HandoffReason.COMPLETE: 107566779,  # Calificado
    HandoffReason.STUCK: 107566783,  # Atención 1 a 1
    HandoffReason.HUMAN_REQUESTED: 107566783,  # Atención 1 a 1
}

# Stage for a lead that went silent past the inactivity window. Parked here as
# per-client config; there is no HandoffReason for it yet (the inactivity timer
# is a future increment), so it is not in REASON_STATUS_IDS.
STATUS_NO_RESPONDIO = 107566787  # No respondió
