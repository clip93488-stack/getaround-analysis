"""GetAround - API de prédiction du prix de location journalier.

Endpoints :
    GET  /             liste des endpoints
    GET  /health       état du service
    GET  /model-info   modèle servi, performances, colonnes et modalités
    POST /predict      prédiction du prix journalier (lot supporté)
    GET  /docs         documentation rédigée (HTML)
    GET  /swagger      Swagger UI généré par FastAPI

Lancement local :  uvicorn api:app --reload --port 8000   (depuis api/)
Le port d'écoute n'est jamais codé en dur : Docker et Render passent par la
variable d'environnement PORT, avec 8000 comme valeur de repli.
"""

import json
import os
import unicodedata
from html import escape
from pathlib import Path
from pprint import pformat
from string import Template
from typing import Any, Dict, List, Union

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

# Chargement des artefacts (même répertoire que ce fichier, en local comme
# dans Docker)
HERE = Path(__file__).resolve().parent

pipeline = joblib.load(HERE / "model.pkl")
feature_info = joblib.load(HERE / "feature_info.pkl")

ALL_COLS: List[str] = feature_info["all_cols"]
NUM_COLS: List[str] = feature_info["num_cols"]
BOOL_COLS: List[str] = feature_info["bool_cols"]
CAT_COLS: List[str] = feature_info["cat_cols"]
CAT_VALUES: Dict[str, List[str]] = feature_info["cat_values"]
METRICS: Dict[str, float] = feature_info["metrics"]
MODEL_NAME: str = feature_info["model_name"]
EXAMPLE: List[Any] = feature_info["example"]

DASHBOARD_URL = os.environ.get(
    "DASHBOARD_URL", "https://getaround-dashboard-z675.onrender.com"
)

app = FastAPI(
    title="GetAround Pricing API",
    description=(
        "Prédiction du prix de location journalier d'une voiture GetAround, "
        "à partir de ses caractéristiques.\n\n"
        f"Modèle servi : **{MODEL_NAME}** - R² test "
        f"{METRICS['test_r2']:.3f}, "
        f"MAE {METRICS['test_mae']:.2f} EUR/jour.\n\n"
        "Documentation rédigée : [/docs](/docs)"
    ),
    version="1.1.0",
    contact={"name": "GetAround Analysis - Jedha Bootcamp Bloc 5"},
    # /docs est réservé à la documentation rédigée demandée par l'énoncé ;
    # l'interface Swagger générée par FastAPI est déplacée sur /swagger.
    docs_url="/swagger",
    redoc_url=None,
    swagger_ui_oauth2_redirect_url=None,
)


# Schémas
Row = Union[List[Any], Dict[str, Any]]


class PredictRequest(BaseModel):
    """Une ou plusieurs voitures à tarifer.

    Deux formats acceptés pour chaque ligne :
      - **positionnel** : liste de valeurs dans l'ordre exact de `all_cols` ;
      - **nommé** : objet {nom_de_colonne: valeur}, plus lisible et robuste.
    """

    input: List[Row] = Field(
        ...,
        min_length=1,
        description=(
            "Liste de voitures, chacune sous forme de liste ordonnée "
            "ou d'objet nommé."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"input": [EXAMPLE]},
                {"input": [dict(zip(ALL_COLS, EXAMPLE))]},
            ]
        }
    }


class PredictResponse(BaseModel):
    prediction: List[float] = Field(
        ..., description="Prix journalier prédit, en euros."
    )

    model_config = {
        "json_schema_extra": {"examples": [{"prediction": [111.88]}]}
    }


# Préparation des entrées
_TRUE = {True, 1, "1", "true", "True", "TRUE", "yes", "oui"}
_FALSE = {False, 0, "0", "false", "False", "FALSE", "no", "non"}


def _fold(text: Any) -> str:
    """Forme comparable d'un libellé : sans accents, en minuscules."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    return (
        decomposed.encode("ascii", "ignore").decode("ascii").casefold().strip()
    )


# Pour chaque colonne catégorielle : libellé "replié" -> modalité vue à
# l'entraînement (ex. "citroen" -> "Citroën").
_CAT_LOOKUP: Dict[str, Dict[str, str]] = {
    col: {_fold(v): v for v in CAT_VALUES[col]} for col in CAT_COLS
}


def _canonical(col: str, value: Any) -> str:
    """Ramène un libellé reçu à la modalité connue du modèle, si elle existe.

    La casse et les accents sont ignorés : "Citroen" est lu comme "Citroën".
    Sans cela, le OneHotEncoder traiterait "Citroen" comme une marque
    inconnue et la prédiction perdrait l'effet de la marque. Un libellé
    réellement inconnu est conservé tel quel.
    """
    value = str(value)
    if value in CAT_VALUES[col]:
        return value
    return _CAT_LOOKUP[col].get(_fold(value), value)


def _to_bool(value: Any, col: str) -> bool:
    """Convertit une valeur reçue en booléen, ou lève une erreur 422."""
    try:
        if value in _TRUE:
            return True
        if value in _FALSE:
            return False
    except TypeError:  # valeur non hachable (liste, dict...)
        pass
    raise HTTPException(
        422, f"Colonne '{col}' : booléen attendu, reçu {value!r}."
    )


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Aligne les types reçus sur ceux attendus par le pipeline."""
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            bad = df.index[df[col].isna()].tolist()
            raise HTTPException(
                422,
                f"Colonne '{col}' : valeur numérique attendue "
                f"(ligne(s) {bad}).",
            )

    for col in BOOL_COLS:
        df[col] = df[col].map(lambda v, c=col: _to_bool(v, c)).astype(bool)

    for col in CAT_COLS:
        df[col] = df[col].map(lambda v, c=col: _canonical(c, v))

    return df


def build_dataframe(rows: List[Row]) -> pd.DataFrame:
    """Convertit la charge utile en DataFrame aux colonnes du modèle."""
    if all(isinstance(r, dict) for r in rows):
        missing = [c for c in ALL_COLS if any(c not in r for r in rows)]
        if missing:
            raise HTTPException(
                422, f"Colonnes manquantes : {missing}. Attendu : {ALL_COLS}"
            )
        df = pd.DataFrame(rows)[ALL_COLS]

    elif all(isinstance(r, list) for r in rows):
        wrong = [i for i, r in enumerate(rows) if len(r) != len(ALL_COLS)]
        if wrong:
            raise HTTPException(
                422,
                f"Chaque ligne doit compter {len(ALL_COLS)} valeurs dans "
                f"l'ordre {ALL_COLS}. Ligne(s) non conforme(s) : {wrong}.",
            )
        df = pd.DataFrame(rows, columns=ALL_COLS)

    else:
        raise HTTPException(
            422,
            "Format hétérogène : utilisez soit uniquement des listes "
            "positionnelles, soit uniquement des objets nommés.",
        )

    return _coerce(df)


def predict_rows(rows: List[Row]) -> List[float]:
    """Prédit le prix journalier (en euros) de chaque voiture reçue."""
    df = build_dataframe(rows)
    try:
        predictions = pipeline.predict(df)
    except Exception as exc:  # pragma: no cover - garde-fou
        raise HTTPException(500, f"Échec de la prédiction : {exc}") from exc
    return [round(float(p), 2) for p in predictions]


# Endpoints
@app.get("/", tags=["General"])
def root():
    """Point d'entrée : rappelle les endpoints disponibles."""
    return {
        "message": "GetAround Pricing API",
        "model": MODEL_NAME,
        "endpoints": {
            "POST /predict": "prédiction du prix journalier",
            "GET /model-info": "colonnes attendues, modalités, performances",
            "GET /health": "état du service",
            "GET /docs": "documentation rédigée",
            "GET /swagger": "Swagger UI interactif",
        },
    }


def _health_payload() -> Dict[str, str]:
    return {"status": "ok", "model": "loaded", "model_type": MODEL_NAME}


@app.get("/health", tags=["General"])
def health():
    """Vérifie que le service et le modèle sont opérationnels."""
    return _health_payload()


@app.get("/model-info", tags=["General"])
def model_info():
    """Décrit le modèle servi et le contrat d'entrée de `/predict`."""
    return {
        "model_type": MODEL_NAME,
        "target": feature_info["target"],
        "metrics": METRICS,
        "n_features": len(ALL_COLS),
        "columns_in_order": ALL_COLS,
        "numeric_columns": NUM_COLS,
        "boolean_columns": BOOL_COLS,
        "categorical_values": CAT_VALUES,
        "categorical_matching": "insensible à la casse et aux accents",
        "observed_price_range_eur": feature_info["price_range"],
        "example_positional": EXAMPLE,
        "example_named": dict(zip(ALL_COLS, EXAMPLE)),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """Prédit le prix de location journalier, en euros.

    Accepte un lot de voitures. Chaque ligne est soit une **liste
    positionnelle** respectant l'ordre de `columns_in_order` (cf.
    `/model-info`), soit un **objet nommé**.

    Les libellés catégoriels sont reconnus sans tenir compte de la casse ni
    des accents. Une modalité inconnue du modèle n'est pas rejetée : elle
    est encodée à zéro par le `OneHotEncoder`, la prédiction reste donc
    exploitable mais moins précise.
    """
    return {"prediction": predict_rows(request.input)}


# Documentation HTML (/docs)
_DOCS_TEMPLATE = Template(r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>API GetAround Pricing : documentation</title>
<style>
  :root {
    --bg: #F8FAFC; --fg: #0F172A; --muted: #475569; --card: #FFFFFF;
    --border: #E2E8F0; --accent: #1E3A5F; --code-bg: #F1F5F9;
    --get: #0369A1; --post: #15803D;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0B1120; --fg: #E2E8F0; --muted: #94A3B8; --card: #111827;
      --border: #1F2937; --accent: #93C5FD; --code-bg: #1E293B;
      --get: #38BDF8; --post: #4ADE80;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 15px/1.65 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 880px; margin: 0 auto; padding: 48px 22px 80px; }
  h1 { font-size: 2rem; margin: 0 0 6px; letter-spacing: -.02em; }
  h2 {
    font-size: 1.3rem; margin: 44px 0 14px; padding-bottom: 8px;
    border-bottom: 2px solid var(--border);
  }
  h3 { font-size: 1rem; margin: 18px 0 8px; }
  .lede { color: var(--muted); font-size: 1.05rem; margin-bottom: 28px; }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px 22px; margin: 16px 0;
  }
  .badges { display: flex; gap: 10px; flex-wrap: wrap; margin: 18px 0 8px; }
  .badge {
    background: var(--code-bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 14px; font-size: .85rem;
  }
  .badge b { display: block; font-size: 1.15rem; color: var(--accent); }
  .m {
    font-weight: 700; padding: 3px 9px; border-radius: 6px; color: #fff;
    font-size: .78rem; letter-spacing: .03em;
  }
  .m-get { background: var(--get); }
  .m-post { background: var(--post); }
  code {
    background: var(--code-bg); padding: 2px 6px; border-radius: 5px;
    font: .88em ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  }
  pre {
    background: var(--code-bg); border: 1px solid var(--border);
    padding: 14px 16px; border-radius: 10px; overflow-x: auto;
  }
  pre code { background: none; padding: 0; }
  table { width: 100%; border-collapse: collapse; font-size: .9rem; }
  th, td {
    text-align: left; padding: 8px 10px; vertical-align: top;
    border-bottom: 1px solid var(--border);
  }
  th {
    color: var(--muted); font-weight: 600; font-size: .8rem;
    text-transform: uppercase; letter-spacing: .04em;
  }
  .scroll { overflow-x: auto; }
  .note { color: var(--muted); margin-bottom: 0; }
  a { color: var(--accent); }
  footer {
    margin-top: 56px; padding-top: 20px; color: var(--muted);
    border-top: 1px solid var(--border); font-size: .87rem;
  }
</style>
</head>
<body>
<div class="wrap">

  <h1>API GetAround Pricing : documentation</h1>
  <p class="lede">Estime le prix de location journalier d'une voiture à
  partir de ses caractéristiques. Le modèle a été entraîné sur $n_rows
  annonces GetAround. Cette API alimente la vue « Estimation de prix » du
  <a href="$dashboard_url">dashboard</a>.</p>

  <div class="badges">
    <span class="badge">Modèle<b>$model</b></span>
    <span class="badge">R² (test)<b>$r2</b></span>
    <span class="badge">Erreur moyenne (MAE)<b>$mae €</b></span>
    <span class="badge">RMSE (test)<b>$rmse €</b></span>
  </div>

  <h2>Endpoints disponibles</h2>
  <div class="card scroll">
    <table>
      <thead>
        <tr><th>Méthode</th><th>Chemin</th><th>Entrée</th><th>Sortie</th></tr>
      </thead>
      <tbody>
        <tr><td><span class="m m-post">POST</span></td>
            <td><code>/predict</code></td>
            <td>JSON <code>{"input": [...]}</code></td>
            <td>JSON <code>{"prediction": [...]}</code></td></tr>
        <tr><td><span class="m m-get">GET</span></td>
            <td><code>/model-info</code></td><td>aucune</td>
            <td>JSON : colonnes, modalités acceptées, performances</td></tr>
        <tr><td><span class="m m-get">GET</span></td>
            <td><code>/health</code></td><td>aucune</td>
            <td>JSON : état du service</td></tr>
        <tr><td><span class="m m-get">GET</span></td>
            <td><code>/</code></td><td>aucune</td>
            <td>JSON : liste des endpoints</td></tr>
        <tr><td><span class="m m-get">GET</span></td>
            <td><code>/docs</code></td><td>aucune</td>
            <td>cette page (HTML)</td></tr>
        <tr><td><span class="m m-get">GET</span></td>
            <td><code><a href="/swagger">/swagger</a></code></td>
            <td>aucune</td><td>Swagger UI interactif (HTML)</td></tr>
      </tbody>
    </table>
  </div>

  <h2>POST /predict</h2>
  <div class="card">
    <p><span class="m m-post">POST</span> &nbsp;<code>/predict</code> -
    <code>Content-Type: application/json</code></p>
    <p>Retourne le prix journalier prédit, en euros, pour un <b>lot</b> de
    voitures. Chaque ligne s'écrit de deux façons.</p>

    <h3>Entrée, format positionnel : $n_cols valeurs dans l'ordre</h3>
    <pre><code>{"input": [$positional]}</code></pre>

    <h3>Entrée, format nommé : plus lisible, ordre indifférent</h3>
    <pre><code>{"input": [$named_json]}</code></pre>

    <h3>Sortie</h3>
    <pre><code>{"prediction": [$prediction]}</code></pre>
    <p class="note">Une valeur par voiture envoyée, dans le même ordre. Les
    libellés catégoriels sont reconnus sans tenir compte de la casse ni des
    accents (<code>citroen</code> est lu comme <code>Citroën</code>). Une
    modalité absente du jeu d'entraînement n'est pas rejetée : elle est
    encodée à zéro, la prédiction reste exploitable mais moins précise.</p>
  </div>

  <h2>Colonnes attendues par /predict</h2>
  <div class="card scroll">
    <table>
      <thead>
        <tr><th>#</th><th>Colonne</th><th>Type</th><th>Domaine</th></tr>
      </thead>
      <tbody>$cols_rows</tbody>
    </table>
  </div>

  <h2>GET /model-info</h2>
  <div class="card">
    <p><span class="m m-get">GET</span> &nbsp;<code>/model-info</code> -
    aucune entrée.</p>
    <p>Sortie : un objet JSON qui décrit le contrat de <code>/predict</code>
    et le modèle servi : <code>model_type</code>, <code>metrics</code>,
    <code>columns_in_order</code>, <code>numeric_columns</code>,
    <code>boolean_columns</code>, <code>categorical_values</code> (toutes les
    modalités acceptées), <code>observed_price_range_eur</code>,
    <code>example_positional</code> et <code>example_named</code>.</p>
  </div>

  <h2>GET /health</h2>
  <div class="card">
    <p><span class="m m-get">GET</span> &nbsp;<code>/health</code> - aucune
    entrée. Utilisé par Docker et par Render pour surveiller le service.</p>
    <h3>Sortie</h3>
    <pre><code>$health</code></pre>
  </div>

  <h2>Exemples d'appel</h2>
  <div class="card">
    <h3>curl</h3>
    <pre><code>curl -X POST "$base_url/predict" \
     -H "Content-Type: application/json" \
     -d '{"input": [$positional]}'</code></pre>

    <h3>Python</h3>
    <pre><code>import requests

row = $named_py

response = requests.post(
    "$base_url/predict",
    json={"input": [row]},
)
print(response.json())  # {'prediction': [$prediction]}</code></pre>
  </div>

  <h2>Codes de réponse</h2>
  <div class="card">
    <table>
      <thead><tr><th>Code</th><th>Signification</th></tr></thead>
      <tbody>
        <tr><td><code>200</code></td><td>Prédiction retournée.</td></tr>
        <tr><td><code>422</code></td><td>Charge utile invalide : nombre de
            valeurs incorrect, colonne manquante, type non convertible. Le
            détail précise la colonne en cause.</td></tr>
        <tr><td><code>500</code></td><td>Erreur interne du modèle.</td></tr>
      </tbody>
    </table>
  </div>

  <footer>GetAround Analysis - Jedha Bootcamp Bloc 5. Modèle
  <code>$model</code> entraîné avec scikit-learn, expériences suivies avec
  MLflow.</footer>
</div>
</body>
</html>""")


def _kind(col: str) -> str:
    if col in NUM_COLS:
        return "nombre"
    if col in BOOL_COLS:
        return "booléen"
    return "texte"


def _domain(col: str) -> str:
    if col in BOOL_COLS:
        return "<code>true</code> / <code>false</code>"
    if col in CAT_COLS:
        vals = CAT_VALUES[col]
        shown = ", ".join(f"<code>{escape(v)}</code>" for v in vals[:6])
        if len(vals) > 6:
            shown += f" … <i>({len(vals)} modalités)</i>"
        return shown
    return "<i>nombre positif</i>"


def _base_url(request: Request) -> str:
    """URL publique du service, y compris derrière le proxy HTTPS de Render."""
    forwarded = request.headers.get("x-forwarded-proto", "")
    scheme = forwarded.split(",")[0].strip()
    if scheme not in ("http", "https"):
        scheme = request.url.scheme
    host = request.headers.get("host", request.url.netloc)
    return f"{scheme}://{host}"


@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
def docs(request: Request):
    """Documentation rédigée, générée à partir du modèle réellement chargé."""
    named = dict(zip(ALL_COLS, EXAMPLE))
    cols_rows = "\n".join(
        f"<tr><td>{i}</td><td><code>{escape(c)}</code></td>"
        f"<td>{_kind(c)}</td><td>{_domain(c)}</td></tr>"
        for i, c in enumerate(ALL_COLS)
    )
    page = _DOCS_TEMPLATE.safe_substitute(
        model=escape(MODEL_NAME),
        n_rows=f"{feature_info['n_rows']:,}".replace(",", " "),
        n_cols=len(ALL_COLS),
        r2=f"{METRICS['test_r2']:.3f}",
        mae=f"{METRICS['test_mae']:.2f}",
        rmse=f"{METRICS['test_rmse']:.2f}",
        base_url=escape(_base_url(request)),
        dashboard_url=escape(DASHBOARD_URL),
        cols_rows=cols_rows,
        positional=escape(json.dumps(EXAMPLE, ensure_ascii=False)),
        named_json=escape(json.dumps(named, ensure_ascii=False, indent=2)),
        named_py=escape(pformat(named, sort_dicts=False, width=60)),
        prediction=predict_rows([EXAMPLE])[0],
        health=escape(json.dumps(_health_payload())),
    )
    return HTMLResponse(page)


@app.get("/docs-custom", include_in_schema=False)
def docs_custom():
    """Ancienne adresse de la documentation rédigée, conservée en alias."""
    return RedirectResponse("/docs")


# Lancement direct : `python api.py`.
# Render injecte la variable d'environnement PORT (10000 par défaut) et impose
# une écoute sur 0.0.0.0. En local et sous Docker classique, PORT est absent
# et le service retombe sur 8000.
if __name__ == "__main__":  # pragma: no cover
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
