"""Real per-trip schemas as *descriptors* + derived extraction models.

A descriptor lists every slot of a trip type with its name, type, whether it is
required, the bot's question text, and any special completeness rule. It is the
single source of truth for the orchestrator (which computes what is still
missing) and for the engine (which extracts values into a derived model).

Design choices (increment 4):

* The required/optional and rule metadata live on the descriptor, **not** inside
  the Pydantic extraction model. The model derived from a descriptor is *pure*:
  business slots only (each ``T | None``), with no ``question`` field — the
  engine composes that in itself.
* Slots with non-trivial completeness (passengers, budget) are nested models so
  the rule can inspect their parts; everything else is free text.

Slot *names* are the real Spanish business vocabulary the agency uses; code
identifiers (classes, fields, helpers) are English.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, create_model


class TripType(StrEnum):
    """The trip types the agent can run a conversation for."""

    CRUISE = "cruise"
    EUROPE = "europe"
    ASIA = "asia"


class SlotRule(StrEnum):
    """How a slot's completeness is judged (interpreted by the orchestrator)."""

    PLAIN = "plain"  # satisfied by any non-null value
    DESTINATION = "destination"  # a concrete destination, or its experience escape
    BUDGET = "budget"  # a stated amount, or "prefiero revisarlo con asesor"
    PASSENGERS = "passengers"  # adults + minor ages when minors are mentioned


class Budget(BaseModel):
    """Budget slot: an amount/range, or a defer-to-advisor escape."""

    amount: str | None = None
    defer_to_advisor: bool | None = None


class Passengers(BaseModel):
    """Passenger slot: adult count plus minor ages when minors are mentioned."""

    adults: int | None = None
    minors_mentioned: bool | None = None
    minor_ages: list[int] | None = None


@dataclass(frozen=True, slots=True)
class SlotSpec:
    """One slot: its name, extraction type, requirement, prompt and rule."""

    name: str
    field_type: type[Any]
    required: bool
    prompt: str
    rule: SlotRule = SlotRule.PLAIN
    # For DESTINATION slots only: the optional free-text experience slot whose
    # presence also satisfies this required destination.
    escape_slot: str | None = None


@dataclass(frozen=True, slots=True)
class TripSchema:
    """A trip type's ordered slots (the descriptor)."""

    trip_type: TripType
    slots: tuple[SlotSpec, ...]


# --- Cruise descriptor -----------------------------------------------------

_CRUISE = TripSchema(
    trip_type=TripType.CRUISE,
    slots=(
        SlotSpec(
            "nombre_cliente", str, True,
            "Para empezar, ¿me compartes tu nombre completo?",
        ),
        SlotSpec(
            "ruta_crucero", str, True,
            "¿Qué ruta o destino de crucero te interesa?",
            rule=SlotRule.DESTINATION,  # cruise has no experience escape
        ),
        SlotSpec(
            "fechas_crucero", str, True,
            "¿En qué fechas te gustaría viajar?",
        ),
        SlotSpec(
            "pasajeros_crucero", Passengers, True,
            "¿Cuántas personas viajan? Si viajan menores de 15 años, "
            "indícame sus edades.",
            rule=SlotRule.PASSENGERS,
        ),
        SlotSpec(
            "cabinas_crucero", str, False,
            "¿Cuántas cabinas necesitas?",
        ),
        SlotSpec(
            "tipo_cabina", str, False,
            "¿Qué tipo de cabina prefieres (interior, exterior, balcón o suite)?",
        ),
        SlotSpec(
            "experiencia_crucero", str, False,
            "¿Qué tipo de experiencia a bordo buscas?",
        ),
        SlotSpec(
            "servicios_crucero", str, False,
            "¿Te interesa contratar algún servicio adicional?",
        ),
        SlotSpec(
            "ciudad_salida_crucero", str, False,
            "¿Desde qué ciudad te gustaría salir?",
        ),
        SlotSpec(
            "presupuesto_crucero", Budget, True,
            "¿Cuál es tu presupuesto aproximado? Si lo prefieres, podemos "
            "revisarlo con un asesor.",
            rule=SlotRule.BUDGET,
        ),
        SlotSpec(
            "pasaporte_crucero", str, False,
            "¿Cuentas con pasaporte vigente?",
        ),
    ),
)


# --- Europe / Asia descriptors (identical modulo suffix + destination) ------


def _continent_schema(
    trip_type: TripType, suffix: str, destino_slot: str, continent: str
) -> TripSchema:
    """Build a Europe/Asia descriptor; they differ only by suffix + destination."""
    return TripSchema(
        trip_type=trip_type,
        slots=(
            SlotSpec(
                "nombre_cliente", str, True,
                "Para empezar, ¿me compartes tu nombre completo?",
            ),
            SlotSpec(
                destino_slot, str, True,
                f"¿Qué destinos de {continent} te gustaría visitar?",
                rule=SlotRule.DESTINATION,
                escape_slot=f"experiencia{suffix}",
            ),
            SlotSpec(
                f"experiencia{suffix}", str, False,
                "¿Qué tipo de experiencia buscas en este viaje?",
            ),
            SlotSpec(
                f"servicios{suffix}", str, False,
                "¿Te interesa contratar algún servicio adicional?",
            ),
            SlotSpec(
                f"fechas{suffix}", str, True,
                "¿En qué fechas te gustaría viajar?",
            ),
            SlotSpec(
                f"duracion{suffix}", str, True,
                "¿Cuántos días durará el viaje?",
            ),
            SlotSpec(
                f"pasajeros{suffix}", Passengers, True,
                "¿Cuántas personas viajan? Si viajan menores de 15 años, "
                "indícame sus edades.",
                rule=SlotRule.PASSENGERS,
            ),
            SlotSpec(
                f"ciudad_salida{suffix}", str, False,
                "¿Desde qué ciudad te gustaría salir?",
            ),
            SlotSpec(
                f"nivel_hospedaje{suffix}", str, False,
                "¿Qué nivel de hospedaje prefieres?",
            ),
            SlotSpec(
                f"vuelos{suffix}", str, False,
                "¿Quieres que incluyamos los vuelos?",
            ),
            SlotSpec(
                f"ocasion{suffix}", str, False,
                "¿El viaje es para alguna ocasión especial?",
            ),
            SlotSpec(
                f"presupuesto{suffix}", Budget, True,
                "¿Cuál es tu presupuesto aproximado? Si lo prefieres, podemos "
                "revisarlo con un asesor.",
                rule=SlotRule.BUDGET,
            ),
            SlotSpec(
                f"pasaporte{suffix}", str, False,
                "¿Cuentas con pasaporte vigente?",
            ),
        ),
    )


_EUROPE = _continent_schema(TripType.EUROPE, "_europa", "paises_europa", "Europa")
_ASIA = _continent_schema(TripType.ASIA, "_asia", "destinos_asia", "Asia")


# --- Registry + extraction-model derivation --------------------------------

_DESCRIPTORS: dict[TripType, TripSchema] = {
    TripType.CRUISE: _CRUISE,
    TripType.EUROPE: _EUROPE,
    TripType.ASIA: _ASIA,
}


def build_extraction_model(schema: TripSchema) -> type[BaseModel]:
    """Derive a pure Pydantic extraction model (each slot ``type | None``)."""
    fields: dict[str, Any] = {
        slot.name: (slot.field_type | None, None) for slot in schema.slots
    }
    model_name = f"{schema.trip_type.value.capitalize()}Extraction"
    # create_model's typing can't see dynamic field names; the field defs are
    # well-formed (``(type | None, default)``).
    model: type[BaseModel] = create_model(model_name, **fields)  # pyright: ignore[reportCallIssue]
    return model


_MODELS: dict[TripType, type[BaseModel]] = {
    trip_type: build_extraction_model(schema)
    for trip_type, schema in _DESCRIPTORS.items()
}


def descriptor_for(trip_type: TripType) -> TripSchema:
    """Return the descriptor for ``trip_type``."""
    return _DESCRIPTORS[trip_type]


def extraction_model_for(trip_type: TripType) -> type[BaseModel]:
    """Return the (cached) pure extraction model for ``trip_type``."""
    return _MODELS[trip_type]
