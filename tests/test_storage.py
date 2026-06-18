"""StateStore contract tests against the in-memory fake (fast, no DB)."""

from __future__ import annotations

from app.domain.state import ConversationState, Phase
from app.understanding.schemas import TripType
from tests.fakes import InMemoryStateStore

CID = "whatsapp:+5215512345678"


async def test_save_then_load_returns_equal_state() -> None:
    store = InMemoryStateStore()
    state = ConversationState(
        trip_type=TripType.CRUISE,
        slots={"nombre_cliente": "Ana", "pasajeros_crucero": {"adults": 2}},
        phase=Phase.COLLECTING,
        last_asked="ruta_crucero",
        asked={"nombre_cliente", "ruta_crucero"},
        attempts={"presupuesto_crucero": 1},
        pending=set(),
        last_bot_message="¿Qué ruta?",
    )

    await store.save(CID, state)

    assert await store.load(CID) == state


async def test_load_absent_returns_none() -> None:
    store = InMemoryStateStore()

    assert await store.load(CID) is None


async def test_save_overwrites_existing_state() -> None:
    store = InMemoryStateStore()
    await store.save(CID, ConversationState(trip_type=TripType.CRUISE))

    await store.save(
        CID, ConversationState(trip_type=TripType.EUROPE, slots={"x": 1})
    )

    loaded = await store.load(CID)
    assert loaded is not None
    assert loaded.trip_type is TripType.EUROPE
    assert loaded.slots == {"x": 1}


async def test_none_trip_type_is_round_tripped() -> None:
    store = InMemoryStateStore()
    await store.save(CID, ConversationState())  # not routed yet

    loaded = await store.load(CID)

    assert loaded is not None
    assert loaded.trip_type is None
