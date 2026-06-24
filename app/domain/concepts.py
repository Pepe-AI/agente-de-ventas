"""Trip-data concepts: the shared grouping over the schema's slots (core).

A *concept* is what a slot MEANS (destination, dates, budget, ...), independent of
any client or CRM. The schema names slots with a per-trip suffix (``paises_europa``
vs ``destinos_asia`` vs ``ruta_crucero``), but they share one concept; ``SLOT_CONCEPTS``
is the explicit normalization table (no common stem exists, so a suffix stripper
would not work). Its ORDER matters where several slots map to one concept: the
concrete destination slots come before the experience escapes, so a consumer that
keeps the first match per concept prefers a concrete value over the escape.

This lives in the domain so both the per-client CRM mapping (concept -> field_id)
and the handoff-note composer (concept -> label) import it from here. The
dependency only ever points inward: CRM -> domain, composer -> domain, never the
reverse. ``nombre_cliente`` is intentionally absent — it is set on the contact, not
a lead custom field; consumers that still need it (e.g. the note) handle it apart.
"""

from __future__ import annotations

from enum import StrEnum


class Concept(StrEnum):
    """What a slot means, independent of client/CRM (a lead field's concept)."""

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


# Domain slot name -> concept. ORDER IS SIGNIFICANT for DESTINO: concrete
# destination slots first, experience escapes last (a consumer keeping the first
# filled slot per concept prefers the concrete one). ``nombre_cliente`` is absent
# on purpose — it is set on the contact by create_lead, never a lead custom field.
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
