"""Tests for the pure campaign trip-type classifier (no LLM, no I/O)."""

from __future__ import annotations

from app.domain.models import Referral
from app.routing.campaign import RoutingConfig, classify_trip_type
from app.understanding.schemas import TripType

_NO_PREFILL = RoutingConfig(
    prefill_crucero=None, prefill_europa=None, prefill_asia=None
)


def _ref(headline: str, body: str) -> Referral:
    return Referral(source_id="sid", headline=headline, body=body, ctwa_clid="clid")


# --- (a) pre-fill phrase ----------------------------------------------------


def test_prefill_phrase_matches_each_type() -> None:
    cfg = RoutingConfig(
        prefill_crucero="viaje por el mediterraneo",
        prefill_europa="tour por el viejo continente",
        prefill_asia="aventura en oriente lejano",
    )

    assert classify_trip_type("Quiero un viaje por el Mediterráneo", None, cfg) is (
        TripType.CRUISE
    )
    assert classify_trip_type("Un Tour por el Viejo Continente", None, cfg) is (
        TripType.EUROPE
    )
    assert classify_trip_type("Busco aventura en oriente lejano", None, cfg) is (
        TripType.ASIA
    )


# --- (b) keyword in text ----------------------------------------------------


def test_keyword_in_text_matches() -> None:
    assert classify_trip_type("me interesa un crucero", None, _NO_PREFILL) is (
        TripType.CRUISE
    )
    assert classify_trip_type("queremos viajar en barco", None, _NO_PREFILL) is (
        TripType.CRUISE  # synonym stem
    )
    assert classify_trip_type("quiero ir a Europa", None, _NO_PREFILL) is (
        TripType.EUROPE
    )
    assert classify_trip_type("un viaje por Asia", None, _NO_PREFILL) is TripType.ASIA


# --- (c) keyword in referral ------------------------------------------------


def test_keyword_in_referral_headline_or_body() -> None:
    by_headline = _ref("Cruceros por el Caribe", "Reserva tu lugar")
    assert classify_trip_type("hola", by_headline, _NO_PREFILL) is TripType.CRUISE

    by_body = _ref("Promoción", "Descubre Europa este verano")
    assert classify_trip_type("hola", by_body, _NO_PREFILL) is TripType.EUROPE


# --- priority: pre-fill > text keyword > referral ---------------------------


def test_prefill_beats_text_keyword() -> None:
    cfg = RoutingConfig(
        prefill_crucero="paquete especial", prefill_europa=None, prefill_asia=None
    )
    # Text contains the "europ" stem, but the pre-fill phrase maps to cruise.
    assert classify_trip_type("paquete especial rumbo a Europa", None, cfg) is (
        TripType.CRUISE
    )


def test_text_keyword_beats_referral() -> None:
    referral = _ref("Promo Asia", "ofertas de temporada")
    assert classify_trip_type("quiero un crucero", referral, _NO_PREFILL) is (
        TripType.CRUISE
    )


# --- normalization ----------------------------------------------------------


def test_normalization_handles_case_accents_and_spaces() -> None:
    assert classify_trip_type("   CRUCERO   ", None, _NO_PREFILL) is TripType.CRUISE
    assert classify_trip_type("Európa", None, _NO_PREFILL) is TripType.EUROPE
    assert classify_trip_type("viaje ASIÁTICO", None, _NO_PREFILL) is TripType.ASIA


# --- word-boundary keyword matching (no false positives mid-word) -----------


def test_stem_inside_unrelated_word_is_not_matched() -> None:
    # "asia" inside "gimnasia"; "barco" inside "embarco"/"desembarco".
    assert classify_trip_type("hago gimnasia los lunes", None, _NO_PREFILL) is None
    assert classify_trip_type("tuve un desembarco lento", None, _NO_PREFILL) is None
    assert classify_trip_type("el embarco fue confuso", None, _NO_PREFILL) is None


def test_stem_at_word_start_mid_phrase_still_matches() -> None:
    assert classify_trip_type("quiero un crucero por el caribe", None, _NO_PREFILL) is (
        TripType.CRUISE
    )
    assert classify_trip_type("me interesa europa este año", None, _NO_PREFILL) is (
        TripType.EUROPE
    )
    assert classify_trip_type("algo asiatico, por favor", None, _NO_PREFILL) is (
        TripType.ASIA
    )
    assert classify_trip_type("queremos viajar en barco", None, _NO_PREFILL) is (
        TripType.CRUISE
    )


# --- no signal --------------------------------------------------------------


def test_no_signal_returns_none() -> None:
    assert classify_trip_type("hola buenas tardes", None, _NO_PREFILL) is None
    assert (
        classify_trip_type("hola", _ref("Promoción", "Reserva ya"), _NO_PREFILL)
        is None
    )
