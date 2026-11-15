"""Tests de l'API de prédiction (FastAPI TestClient, modèle réel)."""

import ast
import html
import re

import pytest

import api

NAMED = dict(zip(api.ALL_COLS, api.EXAMPLE))


def post_predict(client, rows):
    return client.post("/predict", json={"input": rows})


def predicted(client, row):
    response = post_predict(client, [row])
    assert response.status_code == 200
    return response.json()["prediction"][0]


# /health et /model-info
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "loaded"


def test_model_info_describes_the_input_contract(client):
    info = client.get("/model-info").json()
    assert info["columns_in_order"] == api.ALL_COLS
    assert info["n_features"] == len(api.ALL_COLS)
    assert set(info["categorical_values"]) == set(api.CAT_COLS)
    assert "Citroën" in info["categorical_values"]["model_key"]
    assert set(info["metrics"]) >= {"cv_rmse", "test_rmse", "test_r2"}


# /predict : cas nominaux
def test_predict_positional_row_returns_a_plausible_price(client):
    low, high = client.get("/model-info").json()["observed_price_range_eur"]
    price = predicted(client, api.EXAMPLE)
    assert isinstance(price, float)
    assert low <= price <= high


def test_predict_named_row_matches_positional_row(client):
    assert predicted(client, NAMED) == predicted(client, api.EXAMPLE)


def test_predict_batch_returns_one_price_per_car_in_order(client):
    cheap = dict(NAMED, model_key="Renault", mileage=250000, engine_power=60)
    luxury = dict(NAMED, model_key="Ferrari", mileage=5000, engine_power=300)
    response = post_predict(client, [cheap, luxury, cheap])
    prices = response.json()["prediction"]
    assert len(prices) == 3
    assert prices[0] == prices[2]
    assert prices[1] > prices[0]


# /predict : normalisation des libellés catégoriels
@pytest.mark.parametrize(
    "spelling", ["Citroen", "citroën", "CITROEN", "  citroen "]
)
def test_brand_spelling_without_accent_or_case_gives_same_price(
    client, spelling
):
    reference = predicted(client, NAMED)
    assert predicted(client, dict(NAMED, model_key=spelling)) == reference


def test_unknown_brand_is_not_rejected(client):
    price = predicted(client, dict(NAMED, model_key="Marque inconnue"))
    assert isinstance(price, float)


# /predict : erreurs
def test_wrong_number_of_values_returns_422(client):
    response = post_predict(client, [api.EXAMPLE[:-1]])
    assert response.status_code == 422


def test_missing_column_returns_422(client):
    incomplete = {k: v for k, v in NAMED.items() if k != "mileage"}
    response = post_predict(client, [incomplete])
    assert response.status_code == 422
    assert "mileage" in response.json()["detail"]


def test_non_numeric_value_returns_422(client):
    response = post_predict(client, [dict(NAMED, mileage="beaucoup")])
    assert response.status_code == 422
    assert "mileage" in response.json()["detail"]


def test_invalid_boolean_returns_422(client):
    response = post_predict(client, [dict(NAMED, has_gps="peut-être")])
    assert response.status_code == 422
    assert "has_gps" in response.json()["detail"]


def test_mixed_formats_return_422(client):
    response = post_predict(client, [api.EXAMPLE, NAMED])
    assert response.status_code == 422


def test_empty_input_returns_422(client):
    assert post_predict(client, []).status_code == 422


# /docs : documentation rédigée demandée par l'énoncé
@pytest.fixture(scope="module")
def docs_page(client):
    response = client.get("/docs")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    return response.text


def test_docs_has_a_single_h1_and_one_h2_per_endpoint(docs_page):
    assert len(re.findall(r"<h1>", docs_page)) == 1
    h2_titles = re.findall(r"<h2>(.*?)</h2>", docs_page)
    for endpoint in ("POST /predict", "GET /model-info", "GET /health"):
        assert endpoint in h2_titles


def test_docs_lists_every_endpoint_with_its_method(docs_page):
    for path in ("/predict", "/model-info", "/health", "/", "/docs"):
        assert f"<code>{path}</code>" in docs_page
    assert 'href="/swagger"' in docs_page
    assert "m-post" in docs_page and "m-get" in docs_page


def test_docs_has_no_unfilled_placeholder(docs_page):
    assert "$" not in docs_page


def test_docs_examples_use_the_real_url_of_the_service(docs_page):
    assert 'curl -X POST "http://testserver/predict"' in docs_page
    assert '"http://testserver/predict",' in docs_page


def test_docs_python_example_is_valid_python(docs_page):
    block = re.search(
        r"<pre><code>(import requests.*?)</code></pre>", docs_page, re.S
    )
    code = html.unescape(block.group(1))
    tree = ast.parse(code)
    row = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and node.targets[0].id == "row"
    )
    assert ast.literal_eval(row) == NAMED


def test_docs_shows_the_prediction_the_api_really_returns(client, docs_page):
    price = predicted(client, api.EXAMPLE)
    assert f'{{"prediction": [{price}]}}' in docs_page


def test_docs_honours_the_https_proxy_header(client):
    response = client.get("/docs", headers={"x-forwarded-proto": "https"})
    assert "https://testserver/predict" in response.text


def test_old_docs_custom_address_redirects_to_docs(client):
    response = client.get("/docs-custom", follow_redirects=False)
    assert response.status_code in (301, 302, 307, 308)
    assert response.headers["location"] == "/docs"


# Swagger UI générée par FastAPI, déplacée sur /swagger
def test_swagger_ui_is_served_on_swagger(client):
    response = client.get("/swagger")
    assert response.status_code == 200
    assert "swagger-ui" in response.text


def test_openapi_schema_lists_predict(client):
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
    assert "post" in schema["paths"]["/predict"]
