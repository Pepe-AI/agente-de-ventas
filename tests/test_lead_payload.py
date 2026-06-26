"""Offline tests for the lead custom-fields payload builder (core) and the
Top Viajes per-client mapping (data).

The builder is client-agnostic: it takes a slot->concept table and a
concept->field_id table and produces Kommo's ``custom_fields_values`` shape,
serializing the structured slots (Budget/Passengers) to text, omitting empty or
unmapped slots (option B), and collapsing the suffix-per-trip slot names onto the
shared concepts. The Top Viajes mapping is verified against the real schema.
"""

from __future__ import annotations

from app.crm import kommo_mapping_topviajes as m
from app.crm.lead_payload import build_custom_fields_values
from app.domain.concepts import SLOT_CONCEPTS, Concept
from app.domain.models import HandoffReason
from app.understanding.schemas import TripType, descriptor_for, escape_slot_names

# --- The builder, exercised with a tiny fake mapping ------------------------

_FAKE_SLOT_CONCEPTS: dict[str, str] = {
    # concrete destination slots BEFORE the escape so the concrete one wins.
    "paises_europa": "destino",
    "destinos_asia": "destino",
    "experiencia_europa": "destino",  # escape: lower precedence
    "fechas_europa": "fecha",
    "presupuesto_europa": "inversion",
    "pasajeros_europa": "pasajeros",
}
_FAKE_FIELD_IDS: dict[str, int] = {
    "destino": 1112708,
    "fecha": 1112714,
    "inversion": 1114102,
    "pasajeros": 1112718,
}


def test_builder_maps_plain_text_slot_to_field() -> None:
    cfv = build_custom_fields_values(
        {"fechas_europa": "Julio 2026"},
        slot_concepts=_FAKE_SLOT_CONCEPTS,
        concept_field_ids=_FAKE_FIELD_IDS,
    )
    assert cfv == [{"field_id": 1112714, "values": [{"value": "Julio 2026"}]}]


def test_builder_omits_empty_and_unmapped_slots() -> None:
    cfv = build_custom_fields_values(
        {
            "fechas_europa": "",  # empty -> omitted
            "paises_europa": None,  # null -> omitted
            "ruta_crucero": "Caribe",  # not in the fake mapping -> omitted
        },
        slot_concepts=_FAKE_SLOT_CONCEPTS,
        concept_field_ids=_FAKE_FIELD_IDS,
    )
    assert cfv == []


def test_builder_concrete_destination_wins_over_escape() -> None:
    # Concrete destination present: written raw, NO escape prefix even though the
    # escape slot is also filled and declared an escape.
    cfv = build_custom_fields_values(
        {"paises_europa": "Italia", "experiencia_europa": "luna de miel"},
        slot_concepts=_FAKE_SLOT_CONCEPTS,
        concept_field_ids=_FAKE_FIELD_IDS,
        escape_slots={"experiencia_europa"},
    )
    assert cfv == [{"field_id": 1112708, "values": [{"value": "Italia"}]}]


def test_builder_escape_value_is_prefixed_when_concrete_destination_missing() -> None:
    # Only the escape slot is filled: the value is a preference, not a confirmed
    # destination, so it is prefixed to tell the advisor.
    cfv = build_custom_fields_values(
        {"experiencia_europa": "playa y descanso"},
        slot_concepts=_FAKE_SLOT_CONCEPTS,
        concept_field_ids=_FAKE_FIELD_IDS,
        escape_slots={"experiencia_europa"},
    )
    assert cfv == [
        {
            "field_id": 1112708,
            "values": [{"value": "Sin destino definido — busca: playa y descanso"}],
        }
    ]


def test_builder_serializes_budget_defer_and_amount() -> None:
    deferred = build_custom_fields_values(
        {"presupuesto_europa": {"defer_to_advisor": True}},
        slot_concepts=_FAKE_SLOT_CONCEPTS,
        concept_field_ids=_FAKE_FIELD_IDS,
    )
    assert deferred == [
        {
            "field_id": 1114102,
            "values": [{"value": "Prefiere revisarlo con un asesor"}],
        }
    ]

    amount = build_custom_fields_values(
        {"presupuesto_europa": {"amount": "5000 USD"}},
        slot_concepts=_FAKE_SLOT_CONCEPTS,
        concept_field_ids=_FAKE_FIELD_IDS,
    )
    assert amount == [{"field_id": 1114102, "values": [{"value": "5000 USD"}]}]


def test_builder_serializes_passengers_with_minors() -> None:
    cfv = build_custom_fields_values(
        {
            "pasajeros_europa": {
                "adults": 2,
                "minors_mentioned": True,
                "minor_ages": [8, 12],
            }
        },
        slot_concepts=_FAKE_SLOT_CONCEPTS,
        concept_field_ids=_FAKE_FIELD_IDS,
    )
    assert cfv == [
        {"field_id": 1112718, "values": [{"value": "2 adultos, 2 menores (8, 12)"}]}
    ]


def test_builder_serializes_passengers_singular_and_adults_only() -> None:
    one_adult = build_custom_fields_values(
        {"pasajeros_europa": {"adults": 1}},
        slot_concepts=_FAKE_SLOT_CONCEPTS,
        concept_field_ids=_FAKE_FIELD_IDS,
    )
    assert one_adult == [{"field_id": 1112718, "values": [{"value": "1 adulto"}]}]


def test_builder_passengers_minors_mentioned_without_ages() -> None:
    cfv = build_custom_fields_values(
        {"pasajeros_europa": {"adults": 2, "minors_mentioned": True}},
        slot_concepts=_FAKE_SLOT_CONCEPTS,
        concept_field_ids=_FAKE_FIELD_IDS,
    )
    assert cfv == [
        {
            "field_id": 1112718,
            "values": [{"value": "2 adultos, menores (edades sin especificar)"}],
        }
    ]


def test_builder_empty_passengers_and_budget_are_omitted() -> None:
    cfv = build_custom_fields_values(
        {
            "pasajeros_europa": {"adults": None},
            "presupuesto_europa": {"amount": None, "defer_to_advisor": None},
        },
        slot_concepts=_FAKE_SLOT_CONCEPTS,
        concept_field_ids=_FAKE_FIELD_IDS,
    )
    assert cfv == []


# --- The Top Viajes per-client mapping, verified against the real schema -----

# nombre_cliente lives on the CONTACT (set by create_lead), never a lead field.
_EXCLUDED_FROM_LEAD = {"nombre_cliente"}


def test_mapping_covers_every_real_schema_slot_or_excludes_it() -> None:
    for trip in TripType:
        for slot in descriptor_for(trip).slots:
            if slot.name in _EXCLUDED_FROM_LEAD:
                assert slot.name not in SLOT_CONCEPTS
            else:
                assert slot.name in SLOT_CONCEPTS, slot.name
                assert SLOT_CONCEPTS[slot.name] in m.CONCEPT_FIELD_IDS


def test_mapping_concept_field_ids_match_the_account() -> None:
    assert m.CONCEPT_FIELD_IDS == {
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


def test_mapping_reason_status_and_pipeline_ids() -> None:
    assert m.PIPELINE_ID == 13937935
    assert m.REASON_STATUS_IDS == {
        HandoffReason.COMPLETE: 107566779,
        HandoffReason.STUCK: 107566783,
        HandoffReason.HUMAN_REQUESTED: 107566783,
        HandoffReason.NO_RESPONSE: 107566787,  # inactivity timer -> "No respondió"
    }
    assert m.STATUS_NO_RESPONDIO == 107566787


def test_mapping_destination_escape_has_lower_precedence_than_concrete() -> None:
    # In SLOT_CONCEPTS order, the concrete destination slots must precede the
    # experience escapes so the builder picks the concrete one when both exist.
    names = list(SLOT_CONCEPTS)
    assert names.index("paises_europa") < names.index("experiencia_europa")
    assert names.index("destinos_asia") < names.index("experiencia_asia")


def test_topviajes_mapping_builds_a_real_europe_payload() -> None:
    slots: dict[str, object] = {
        "nombre_cliente": "Ana",  # excluded -> not in the payload
        "paises_europa": "Italia y Francia",
        "fechas_europa": "Septiembre 2026",
        "pasajeros_europa": {"adults": 2, "minor_ages": [10]},
        "presupuesto_europa": {"defer_to_advisor": True},
    }
    cfv = build_custom_fields_values(
        slots,
        slot_concepts=SLOT_CONCEPTS,
        concept_field_ids=m.CONCEPT_FIELD_IDS,
        escape_slots=escape_slot_names(),
    )
    assert {entry["field_id"] for entry in cfv} == {1112708, 1112714, 1112718, 1114102}
    by_id = {entry["field_id"]: entry["values"][0]["value"] for entry in cfv}
    assert by_id[1112708] == "Italia y Francia"  # concrete -> raw, no prefix
    assert by_id[1112718] == "2 adultos, 1 menor (10)"
    assert by_id[1114102] == "Prefiere revisarlo con un asesor"


def test_topviajes_mapping_prefixes_real_destination_escape() -> None:
    # Asia conversation that gave only an experience (no concrete country): the
    # destino field is filled from experiencia_asia, flagged as a preference.
    cfv = build_custom_fields_values(
        {"experiencia_asia": "templos y gastronomía"},
        slot_concepts=SLOT_CONCEPTS,
        concept_field_ids=m.CONCEPT_FIELD_IDS,
        escape_slots=escape_slot_names(),
    )
    assert cfv == [
        {
            "field_id": 1112708,
            "values": [
                {"value": "Sin destino definido — busca: templos y gastronomía"}
            ],
        }
    ]


def test_escape_slot_names_are_the_destination_escapes() -> None:
    assert escape_slot_names() == frozenset({"experiencia_europa", "experiencia_asia"})
