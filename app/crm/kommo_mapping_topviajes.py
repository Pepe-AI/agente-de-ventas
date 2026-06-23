"""Top Viajes' Kommo CRM mapping — the PER-CLIENT data artifact.

This module holds ONLY data, no logic: the numeric IDs are specific to the Top
Viajes Kommo account. Another client gets its own copy of this file (the same
``_topviajes`` per-client convention as ``corpus_topviajes.md``); the logic that
consumes it (``app/crm/lead_payload.py``, the client, the orchestration) is core
and shared. Keeping the IDs in typed Python — not JSON/env — lets pyright and the
mapping-integrity test catch a missing concept or a fat-fingered id at check time.

The domain slot names carry a per-trip suffix (``paises_europa`` vs
``destinos_asia`` vs ``ruta_crucero``), but Kommo has ONE generic field per
concept, so several slot names collapse onto the same concept. ``SLOT_CONCEPTS``
is the explicit normalization table (no common stem exists, so a suffix stripper
would not work). Its ORDER matters: the concrete destination slots are listed
before the experience escapes so the builder, which keeps the first filled slot
per concept, prefers a concrete destination over the escape.
"""

from __future__ import annotations

from enum import StrEnum

from app.domain.models import HandoffReason


class Concept(StrEnum):
    """A Kommo LEAD custom field, by business concept (account-independent name)."""

    DESTINO = "destino"
    FECHA = "fecha"
    DURACION = "duracion"
    PASAJEROS = "pasajeros"
    INVERSION = "inversion"
    CIUDAD_SALIDA = "ciudad_salida"
    SERVICIOS = "servicios"
    NIVEL_HOSPEDAJE = "nivel_hospedaje"
    VUELOS = "vuelos"
    OCASION = "ocasion"
    DOCUMENTACION = "documentacion"
    CABINAS = "cabinas"
    TIPO_CABINA = "tipo_cabina"
    EXPERIENCIA_CRUCERO = "experiencia_crucero"


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


# Domain slot name -> concept. ORDER IS SIGNIFICANT for DESTINO: concrete
# destination slots first, experience escapes last (the builder keeps the first
# filled slot per concept). ``nombre_cliente`` is absent on purpose — it is set on
# the contact by create_lead, never a lead custom field.
SLOT_CONCEPTS: dict[str, Concept] = {
    # destino — concrete first, escapes last (precedence)
    "ruta_crucero": Concept.DESTINO,
    "paises_europa": Concept.DESTINO,
    "destinos_asia": Concept.DESTINO,
    "experiencia_europa": Concept.DESTINO,  # escape
    "experiencia_asia": Concept.DESTINO,  # escape
    # fecha / temporada
    "fechas_crucero": Concept.FECHA,
    "fechas_europa": Concept.FECHA,
    "fechas_asia": Concept.FECHA,
    # duración (no cruise slot)
    "duracion_europa": Concept.DURACION,
    "duracion_asia": Concept.DURACION,
    # pasajeros (Passengers sub-model)
    "pasajeros_crucero": Concept.PASAJEROS,
    "pasajeros_europa": Concept.PASAJEROS,
    "pasajeros_asia": Concept.PASAJEROS,
    # inversión / presupuesto del cliente (Budget sub-model)
    "presupuesto_crucero": Concept.INVERSION,
    "presupuesto_europa": Concept.INVERSION,
    "presupuesto_asia": Concept.INVERSION,
    # ciudad de salida
    "ciudad_salida_crucero": Concept.CIUDAD_SALIDA,
    "ciudad_salida_europa": Concept.CIUDAD_SALIDA,
    "ciudad_salida_asia": Concept.CIUDAD_SALIDA,
    # servicios adicionales
    "servicios_crucero": Concept.SERVICIOS,
    "servicios_europa": Concept.SERVICIOS,
    "servicios_asia": Concept.SERVICIOS,
    # nivel de hospedaje (no cruise slot)
    "nivel_hospedaje_europa": Concept.NIVEL_HOSPEDAJE,
    "nivel_hospedaje_asia": Concept.NIVEL_HOSPEDAJE,
    # vuelos (no cruise slot)
    "vuelos_europa": Concept.VUELOS,
    "vuelos_asia": Concept.VUELOS,
    # ocasión / motivo del viaje (no cruise slot)
    "ocasion_europa": Concept.OCASION,
    "ocasion_asia": Concept.OCASION,
    # documentación / pasaporte
    "pasaporte_crucero": Concept.DOCUMENTACION,
    "pasaporte_europa": Concept.DOCUMENTACION,
    "pasaporte_asia": Concept.DOCUMENTACION,
    # cruise-only fields (dedicated Kommo fields created for Top Viajes)
    "cabinas_crucero": Concept.CABINAS,
    "tipo_cabina": Concept.TIPO_CABINA,
    "experiencia_crucero": Concept.EXPERIENCIA_CRUCERO,
}


# Pipeline "Embudo de ventas" and the stage a handoff moves the lead to.
PIPELINE_ID = 13937935

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
