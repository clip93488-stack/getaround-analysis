"""Vue « Estimation de prix » du dashboard.

Le formulaire envoie les caractéristiques d'une voiture à l'API de prédiction
(`POST /predict`) et affiche le prix journalier estimé. Le dashboard ne charge
pas le modèle : il interroge l'API comme n'importe quel autre client.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import requests
import streamlit as st

API_URL = os.environ.get(
    "API_URL", "https://getaround-api-ztfm.onrender.com"
).rstrip("/")
CHOICES_FILE = (
    Path(__file__).resolve().parent / "data" / "pricing_choices.json"
)

# L'hébergement gratuit met l'API en veille après 15 minutes sans trafic :
# le premier appel peut attendre 30 à 60 secondes son réveil.
TIMEOUT_S = 90
GATEWAY_ERRORS = {502, 503, 504}
ATTEMPTS = 3
PAUSE_S = 5

NEUTRAL_DELTA = dict(delta_color="off", delta_arrow="off")

FUEL_LABELS = {
    "diesel": "Diesel",
    "petrol": "Essence",
    "hybrid_petrol": "Hybride essence",
    "electro": "Électrique",
}
COLOR_LABELS = {
    "beige": "Beige",
    "black": "Noir",
    "blue": "Bleu",
    "brown": "Marron",
    "green": "Vert",
    "grey": "Gris",
    "orange": "Orange",
    "red": "Rouge",
    "silver": "Argent",
    "white": "Blanc",
}
TYPE_LABELS = {
    "convertible": "Cabriolet",
    "coupe": "Coupé",
    "estate": "Break",
    "hatchback": "Compacte à hayon",
    "sedan": "Berline",
    "subcompact": "Citadine",
    "suv": "SUV",
    "van": "Van",
}
OPTION_LABELS = {
    "private_parking_available": "Parking privé disponible",
    "has_gps": "GPS",
    "has_air_conditioning": "Climatisation",
    "automatic_car": "Boîte automatique",
    "has_getaround_connect": "Getaround Connect",
    "has_speed_regulator": "Régulateur de vitesse",
    "winter_tires": "Pneus hiver",
}


class ApiError(Exception):
    """Erreur de l'API, formulée pour être affichée à l'utilisateur."""


def build_row(values: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    """Ordonne les valeurs du formulaire selon les colonnes de l'API."""
    return {column: values[column] for column in columns}


def _request(method: str, path: str, api_url: str, **kwargs) -> Any:
    """Appelle l'API et retourne le JSON, ou lève une ApiError lisible.

    Trois tentatives espacées de 5 s si la passerelle répond 502/503/504 ou
    refuse la connexion : c'est ce que renvoie l'hébergement pendant le
    réveil du service.
    """
    url = f"{api_url}{path}"
    for attempt in range(1, ATTEMPTS + 1):
        last = attempt == ATTEMPTS
        try:
            response = requests.request(
                method, url, timeout=TIMEOUT_S, **kwargs
            )
        except requests.Timeout as exc:
            raise ApiError(
                "L'API ne répond pas. Elle est probablement en train de se "
                "réveiller : réessayez dans une minute."
            ) from exc
        except requests.ConnectionError as exc:
            if last:
                raise ApiError(
                    f"Impossible de joindre l'API ({url})."
                ) from exc
            time.sleep(PAUSE_S)
            continue

        if response.status_code in GATEWAY_ERRORS and not last:
            time.sleep(PAUSE_S)
            continue
        if response.status_code == 422:
            detail = response.json().get("detail")
            raise ApiError(f"L'API a refusé la saisie : {detail}")
        if not response.ok:
            raise ApiError(
                f"L'API a répondu avec l'erreur {response.status_code}."
            )
        return response.json()

    raise ApiError(f"Impossible de joindre l'API ({url}).")  # pragma: no cover


def call_predict(row: dict[str, Any], api_url: str = API_URL) -> float:
    """Appelle `POST /predict` et retourne le prix journalier (en euros)."""
    body = _request("POST", "/predict", api_url, json={"input": [row]})
    return float(body["prediction"][0])


def fetch_model_info(api_url: str = API_URL) -> Optional[dict[str, Any]]:
    """Appelle `GET /model-info` ; None si l'API ne répond pas.

    Cette information est un complément : son absence ne bloque pas la vue.
    """
    try:
        return _request("GET", "/model-info", api_url)
    except ApiError:
        return None


@st.cache_data
def load_choices() -> dict[str, Any]:
    """Charge les choix du formulaire, produits par api/train.py."""
    return json.loads(CHOICES_FILE.read_text(encoding="utf-8"))


@st.cache_resource(ttl=600, show_spinner=False)
def wake_api(api_url: str) -> bool:
    """Réveille l'API en arrière-plan dès l'ouverture de la vue.

    Le ping part dans un thread : la page s'affiche immédiatement et, le
    temps de remplir le formulaire, l'API a eu le temps de démarrer.
    """

    def ping() -> None:
        try:
            requests.get(f"{api_url}/health", timeout=TIMEOUT_S)
        except requests.RequestException:
            pass

    threading.Thread(target=ping, daemon=True).start()
    return True


def _select(label, column, choices, labels=None):
    options = choices["categories"][column]
    labels = labels or {}
    return st.selectbox(
        label,
        options,
        index=options.index(choices["defaults"][column]),
        format_func=lambda value: labels.get(value, value),
    )


def _number(label, column, choices, step):
    spec = choices["numeric"][column]
    return st.number_input(
        label,
        min_value=int(spec["min"]),
        max_value=int(spec["max"]),
        value=int(spec["median"]),
        step=step,
    )


def _show_result(price, row, choices, info) -> None:
    reference = choices["price"]["mean"]
    left, right = st.columns(2)
    left.metric(
        "Prix journalier estimé",
        f"{price:.0f} €",
        delta=(
            f"{price - reference:+.0f} € par rapport au prix moyen "
            f"observé ({reference:.0f} €)"
        ),
        **NEUTRAL_DELTA,
    )
    if info:
        right.metric(
            "Erreur moyenne du modèle",
            f"± {info['metrics']['test_mae']:.1f} €",
            delta=(
                f"{info['model_type']}, mesurée sur des voitures absentes "
                "de l'entraînement"
            ),
            **NEUTRAL_DELTA,
        )

    with st.expander("Voir l'échange avec l'API"):
        st.markdown(f"**Requête** : `POST {API_URL}/predict`")
        st.code(
            json.dumps({"input": [row]}, ensure_ascii=False, indent=2),
            language="json",
        )
        st.markdown("**Réponse**")
        st.code(json.dumps({"prediction": [round(price, 2)]}), language="json")


def render_price_estimator() -> None:
    """Affiche la vue : formulaire, appel de l'API, résultat."""
    choices = load_choices()
    wake_api(API_URL)

    st.title("Estimation du prix de location")
    st.markdown(
        '<p class="lead">Renseignez les caractéristiques de la voiture : le '
        "prix journalier conseillé est calculé par l'API de prédiction "
        "(<code>POST /predict</code>), qui sert le modèle entraîné.</p>",
        unsafe_allow_html=True,
    )

    values = {}
    with st.form("price_form"):
        left, right = st.columns(2)
        with left:
            values["model_key"] = _select("Marque", "model_key", choices)
            values["car_type"] = _select(
                "Type de carrosserie", "car_type", choices, TYPE_LABELS
            )
            values["fuel"] = _select("Carburant", "fuel", choices, FUEL_LABELS)
            values["paint_color"] = _select(
                "Couleur", "paint_color", choices, COLOR_LABELS
            )
        with right:
            values["mileage"] = _number(
                "Kilométrage (km)", "mileage", choices, step=1000
            )
            values["engine_power"] = _number(
                "Puissance du moteur", "engine_power", choices, step=5
            )
            st.markdown("**Équipements**")
            for column in choices["booleans"]:
                values[column] = st.checkbox(
                    OPTION_LABELS.get(column, column),
                    value=choices["defaults"][column],
                )
        submitted = st.form_submit_button("Estimer le prix", type="primary")

    if not submitted:
        st.caption(
            "L'API est hébergée sur une offre gratuite : après une période "
            "d'inactivité, le premier calcul peut durer jusqu'à une minute."
        )
        return

    row = build_row(values, choices["columns"])
    with st.spinner(
        "Appel de l'API en cours. Le premier appel peut durer jusqu'à une "
        "minute."
    ):
        try:
            price = call_predict(row)
        except ApiError as exc:
            st.error(str(exc))
            return
        info = fetch_model_info()
    _show_result(price, row, choices, info)
