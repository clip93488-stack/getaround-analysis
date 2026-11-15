"""Tests de la vue « Estimation de prix » et de son client HTTP."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import requests
from streamlit.testing.v1 import AppTest

import price_estimator as pe

APP = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
CHOICES = json.loads(pe.CHOICES_FILE.read_text(encoding="utf-8"))
API = "http://api.test"


class FakeResponse:
    """Réponse minimale, compatible avec l'usage de requests.Response."""

    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body if body is not None else {}

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        return self._body


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Les pauses entre tentatives ne doivent pas ralentir les tests.

    Seul le module testé est concerné : remplacer time.sleep globalement
    transformerait en boucle active l'attente interne d'AppTest.
    """
    monkeypatch.setattr(pe, "time", SimpleNamespace(sleep=lambda s: None))


@pytest.fixture
def in_process_api(client, monkeypatch):
    """Redirige les appels du dashboard vers l'API réelle, en mémoire."""

    def route(method, url, timeout=None, **kwargs):
        reply = client.request(method, url.removeprefix(pe.API_URL), **kwargs)
        return FakeResponse(reply.status_code, reply.json())

    monkeypatch.setattr(pe.requests, "request", route)
    monkeypatch.setattr(pe, "wake_api", lambda api_url: True)


def default_row():
    """Voiture telle que le formulaire la propose avant toute saisie."""
    values = dict(CHOICES["defaults"])
    for column, spec in CHOICES["numeric"].items():
        values[column] = int(spec["median"])
    return pe.build_row(values, CHOICES["columns"])


# Cohérence entre le formulaire et le modèle servi
def test_form_choices_match_the_model_contract(client):
    info = client.get("/model-info").json()
    assert CHOICES["columns"] == info["columns_in_order"]
    assert CHOICES["categories"] == info["categorical_values"]
    assert CHOICES["booleans"] == info["boolean_columns"]
    assert list(CHOICES["numeric"]) == info["numeric_columns"]


def test_every_category_has_a_label_or_falls_back_to_its_value():
    for labels, column in (
        (pe.FUEL_LABELS, "fuel"),
        (pe.COLOR_LABELS, "paint_color"),
        (pe.TYPE_LABELS, "car_type"),
    ):
        assert set(labels) == set(CHOICES["categories"][column])
    assert set(pe.OPTION_LABELS) == set(CHOICES["booleans"])


def test_build_row_follows_the_api_column_order():
    values = {column: index for index, column in enumerate(CHOICES["columns"])}
    shuffled = dict(reversed(list(values.items())))
    assert list(pe.build_row(shuffled, CHOICES["columns"])) == (
        CHOICES["columns"]
    )


# Client HTTP
def test_call_predict_posts_the_row_and_returns_the_price(monkeypatch):
    request = mock.Mock(
        return_value=FakeResponse(200, {"prediction": [111.88]})
    )
    monkeypatch.setattr(pe.requests, "request", request)

    price = pe.call_predict(default_row(), API)

    assert price == 111.88
    args, kwargs = request.call_args
    assert args == ("POST", f"{API}/predict")
    assert kwargs["json"] == {"input": [default_row()]}
    assert kwargs["timeout"] == pe.TIMEOUT_S


def test_gateway_errors_are_retried_then_succeed(monkeypatch):
    replies = iter(
        [
            FakeResponse(503),
            FakeResponse(502),
            FakeResponse(200, {"prediction": [99.5]}),
        ]
    )
    monkeypatch.setattr(
        pe.requests, "request", lambda *args, **kwargs: next(replies)
    )
    assert pe.call_predict(default_row(), API) == 99.5


def test_persistent_gateway_error_ends_with_a_readable_message(monkeypatch):
    monkeypatch.setattr(
        pe.requests, "request", lambda *args, **kwargs: FakeResponse(503)
    )
    with pytest.raises(pe.ApiError, match="503"):
        pe.call_predict(default_row(), API)


def test_unreachable_api_is_reported_with_its_address(monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(pe.requests, "request", refuse)
    with pytest.raises(pe.ApiError, match="Impossible de joindre"):
        pe.call_predict(default_row(), API)


def test_timeout_asks_the_user_to_retry(monkeypatch):
    def too_slow(*args, **kwargs):
        raise requests.Timeout("slow")

    monkeypatch.setattr(pe.requests, "request", too_slow)
    with pytest.raises(pe.ApiError, match="réessayez"):
        pe.call_predict(default_row(), API)


def test_validation_error_detail_is_shown(monkeypatch):
    detail = "Colonne 'mileage' : valeur numérique attendue."
    monkeypatch.setattr(
        pe.requests,
        "request",
        lambda *args, **kwargs: FakeResponse(422, {"detail": detail}),
    )
    with pytest.raises(pe.ApiError, match="mileage"):
        pe.call_predict(default_row(), API)


def test_model_info_failure_is_not_blocking(monkeypatch):
    monkeypatch.setattr(
        pe.requests, "request", lambda *args, **kwargs: FakeResponse(500)
    )
    assert pe.fetch_model_info(API) is None


def test_default_row_is_accepted_by_the_real_api(in_process_api):
    price = pe.call_predict(default_row())
    low, high = CHOICES["price"]["min"], CHOICES["price"]["max"]
    assert low <= price <= high


# Parcours complet dans l'interface Streamlit
def open_price_view():
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("Estimation de prix")
    return at.run()


def test_price_view_shows_the_form_without_calling_the_api(monkeypatch):
    called = mock.Mock()
    monkeypatch.setattr(pe.requests, "request", called)
    monkeypatch.setattr(pe, "wake_api", lambda api_url: True)

    at = open_price_view()

    assert not at.exception
    assert at.title[0].value == "Estimation du prix de location"
    assert len(at.selectbox) == 4
    assert len(at.number_input) == 2
    assert len(at.checkbox) == len(CHOICES["booleans"])
    assert not at.metric
    called.assert_not_called()


def test_submitting_the_form_displays_the_price_from_the_api(in_process_api):
    at = open_price_view()
    at.button[0].click().run()

    assert not at.exception
    expected = pe.call_predict(default_row())
    assert at.metric[0].label == "Prix journalier estimé"
    assert at.metric[0].value == f"{expected:.0f} €"
    assert at.metric[1].label == "Erreur moyenne du modèle"


def test_changing_the_brand_changes_the_estimate(in_process_api):
    at = open_price_view()
    at.button[0].click().run()
    before = at.metric[0].value

    brand = next(s for s in at.selectbox if s.label == "Marque")
    brand.set_value("Ferrari")
    at.button[0].click().run()

    assert at.metric[0].value != before


def test_api_failure_is_displayed_as_an_error(monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(pe.requests, "request", refuse)
    monkeypatch.setattr(pe, "wake_api", lambda api_url: True)

    at = open_price_view()
    at.button[0].click().run()

    assert not at.exception
    assert "Impossible de joindre l'API" in at.error[0].value
    assert not at.metric


def test_delay_view_is_still_the_default_view():
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    assert at.title[0].value.startswith("GetAround")
    assert [t.label for t in at.tabs][0] == "Arbitrage bénéfice / coût"
