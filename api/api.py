# -*- coding: utf-8 -*-
"""
GetAround — API de prediction du prix de location journalier.

Endpoints :
    GET  /             informations generales
    GET  /health       etat du service
    GET  /model-info   modele servi, performances, colonnes et modalites acceptees
    POST /predict      prediction du prix journalier (batch supporte)
    GET  /docs         Swagger UI (auto-genere par FastAPI)
    GET  /docs-custom  documentation HTML redigee

Lancement local :  uvicorn api:app --reload --port 8000   (depuis le dossier api/)
Le port d'ecoute n'est jamais code en dur : Docker et Render passent par la
variable d'environnement PORT, avec 8000 comme valeur de repli.
"""

from html import escape
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Chargement des artefacts (memes repertoire que ce fichier, en local comme dans Docker)
# ─────────────────────────────────────────────────────────────────────────────
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

app = FastAPI(
    title="GetAround Pricing API",
    description=(
        "Prediction du prix de location journalier d'une voiture GetAround, "
        "a partir de ses caracteristiques.\n\n"
        f"Modele servi : **{MODEL_NAME}** — R² test {METRICS['test_r2']:.3f}, "
        f"MAE {METRICS['test_mae']:.2f} EUR/jour.\n\n"
        "Documentation redigee : [/docs-custom](/docs-custom)"
    ),
    version="1.0.0",
    contact={"name": "GetAround Analysis · Jedha Bootcamp Bloc 5"},
)


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────
Row = Union[List[Any], Dict[str, Any]]


class PredictRequest(BaseModel):
    """Une ou plusieurs voitures a tarifer.

    Deux formats acceptes pour chaque ligne :
      - **positionnel** : liste de valeurs dans l'ordre exact de `all_cols` ;
      - **nomme** : objet {nom_de_colonne: valeur}, plus lisible et robuste.
    """

    input: List[Row] = Field(
        ...,
        min_length=1,
        description="Liste de voitures, chacune sous forme de liste ordonnee ou d'objet nomme.",
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
    prediction: List[float] = Field(..., description="Prix journalier predit, en euros.")

    model_config = {"json_schema_extra": {"examples": [{"prediction": [111.88]}]}}


# ─────────────────────────────────────────────────────────────────────────────
# Preparation des entrees
# ─────────────────────────────────────────────────────────────────────────────
_TRUE = {True, 1, "1", "true", "True", "TRUE", "yes", "oui"}
_FALSE = {False, 0, "0", "false", "False", "FALSE", "no", "non"}


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Aligne les types recus sur ceux attendus par le pipeline."""
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            bad = df.index[df[col].isna()].tolist()
            raise HTTPException(
                422, f"Colonne '{col}' : valeur numerique attendue (ligne(s) {bad})."
            )

    for col in BOOL_COLS:
        def to_bool(v, _col=col):
            try:
                if v in _TRUE:
                    return True
                if v in _FALSE:
                    return False
            except TypeError:  # valeur non hachable (liste, dict...)
                pass
            raise HTTPException(
                422, f"Colonne '{_col}' : booleen attendu, recu {v!r}."
            )

        df[col] = df[col].map(to_bool).astype(bool)

    for col in CAT_COLS:
        df[col] = df[col].astype(str)

    return df


def build_dataframe(rows: List[Row]) -> pd.DataFrame:
    """Convertit la charge utile en DataFrame aux colonnes attendues par le modele."""
    if all(isinstance(r, dict) for r in rows):
        missing = [c for c in ALL_COLS if any(c not in r for r in rows)]
        if missing:
            raise HTTPException(422, f"Colonnes manquantes : {missing}. Attendu : {ALL_COLS}")
        df = pd.DataFrame(rows)[ALL_COLS]

    elif all(isinstance(r, list) for r in rows):
        wrong = [i for i, r in enumerate(rows) if len(r) != len(ALL_COLS)]
        if wrong:
            raise HTTPException(
                422,
                f"Chaque ligne doit compter {len(ALL_COLS)} valeurs dans l'ordre {ALL_COLS}. "
                f"Ligne(s) non conforme(s) : {wrong}.",
            )
        df = pd.DataFrame(rows, columns=ALL_COLS)

    else:
        raise HTTPException(
            422,
            "Format heterogene : utilisez soit uniquement des listes positionnelles, "
            "soit uniquement des objets nommes.",
        )

    return _coerce(df)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", tags=["General"])
def root():
    """Point d'entree : rappelle les endpoints disponibles."""
    return {
        "message": "GetAround Pricing API",
        "model": MODEL_NAME,
        "endpoints": {
            "POST /predict": "prediction du prix journalier",
            "GET /model-info": "colonnes attendues, modalites, performances",
            "GET /health": "etat du service",
            "GET /docs": "Swagger UI",
            "GET /docs-custom": "documentation redigee",
        },
    }


@app.get("/health", tags=["General"])
def health():
    """Verifie que le service et le modele sont operationnels."""
    return {"status": "ok", "model": "loaded", "model_type": MODEL_NAME}


@app.get("/model-info", tags=["General"])
def model_info():
    """Decrit le modele servi et le contrat d'entree de `/predict`."""
    return {
        "model_type": MODEL_NAME,
        "target": feature_info["target"],
        "metrics": METRICS,
        "n_features": len(ALL_COLS),
        "columns_in_order": ALL_COLS,
        "numeric_columns": NUM_COLS,
        "boolean_columns": BOOL_COLS,
        "categorical_values": CAT_VALUES,
        "observed_price_range_eur": feature_info["price_range"],
        "example_positional": EXAMPLE,
        "example_named": dict(zip(ALL_COLS, EXAMPLE)),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """Predit le prix de location journalier, en euros.

    Accepte un lot de voitures. Chaque ligne est soit une **liste positionnelle**
    respectant l'ordre de `columns_in_order` (cf. `/model-info`), soit un **objet nomme**.

    Une modalite categorielle inconnue du modele n'est pas rejetee : elle est encodee
    a zero par le `OneHotEncoder`, la prediction reste donc exploitable mais moins precise.
    """
    df = build_dataframe(request.input)
    try:
        predictions = pipeline.predict(df)
    except Exception as exc:  # pragma: no cover - garde-fou
        raise HTTPException(500, f"Echec de la prediction : {exc}") from exc

    return {"prediction": [round(float(p), 2) for p in predictions]}


# ─────────────────────────────────────────────────────────────────────────────
# Documentation HTML
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/docs-custom", response_class=HTMLResponse, tags=["Documentation"])
def custom_docs():
    """Documentation HTML redigee, generee a partir du modele reellement charge."""
    cols_rows = "\n".join(
        f"<tr><td>{i}</td><td><code>{escape(c)}</code></td><td>{_kind(c)}</td>"
        f"<td>{_domain(c)}</td></tr>"
        for i, c in enumerate(ALL_COLS)
    )
    example_positional = _json(EXAMPLE)
    example_named = _json(dict(zip(ALL_COLS, EXAMPLE)), indent=2)

    return HTMLResponse(f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GetAround Pricing API — Documentation</title>
<style>
  :root {{
    --bg:#F8FAFC; --fg:#0F172A; --muted:#475569; --card:#FFFFFF;
    --border:#E2E8F0; --accent:#7C3AED; --code-bg:#F1F5F9;
    --get:#0369A1; --post:#15803D;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg:#0B1120; --fg:#E2E8F0; --muted:#94A3B8; --card:#111827;
      --border:#1F2937; --accent:#A78BFA; --code-bg:#1E293B;
      --get:#38BDF8; --post:#4ADE80;
    }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif; }}
  .wrap {{ max-width:880px; margin:0 auto; padding:48px 22px 80px; }}
  h1 {{ font-size:2rem; margin:0 0 6px; letter-spacing:-.02em; }}
  h2 {{ font-size:1.3rem; margin:44px 0 14px; padding-bottom:8px;
        border-bottom:2px solid var(--border); letter-spacing:-.01em; }}
  h3 {{ font-size:1rem; margin:0 0 10px; }}
  .lede {{ color:var(--muted); font-size:1.05rem; margin-bottom:28px; }}
  .card {{ background:var(--card); border:1px solid var(--border);
           border-radius:12px; padding:20px 22px; margin:16px 0; }}
  .badges {{ display:flex; gap:10px; flex-wrap:wrap; margin:18px 0 8px; }}
  .badge {{ background:var(--code-bg); border:1px solid var(--border);
            border-radius:8px; padding:8px 14px; font-size:.85rem; }}
  .badge b {{ display:block; font-size:1.15rem; color:var(--accent); }}
  .m {{ font-weight:700; padding:3px 9px; border-radius:6px; color:#fff;
        font-size:.78rem; letter-spacing:.03em; }}
  .m-get {{ background:var(--get); }} .m-post {{ background:var(--post); }}
  code {{ background:var(--code-bg); padding:2px 6px; border-radius:5px;
          font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace; font-size:.88em; }}
  pre {{ background:var(--code-bg); border:1px solid var(--border); padding:14px 16px;
         border-radius:10px; overflow-x:auto; }}
  pre code {{ background:none; padding:0; }}
  table {{ width:100%; border-collapse:collapse; font-size:.9rem; }}
  th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--border);
           vertical-align:top; }}
  th {{ color:var(--muted); font-weight:600; font-size:.8rem;
        text-transform:uppercase; letter-spacing:.04em; }}
  .scroll {{ overflow-x:auto; }}
  a {{ color:var(--accent); }}
  footer {{ margin-top:56px; padding-top:20px; border-top:1px solid var(--border);
            color:var(--muted); font-size:.87rem; }}
</style>
</head>
<body>
<div class="wrap">

  <h1>🚗 GetAround Pricing API</h1>
  <p class="lede">Estime le prix de location journalier d'une voiture a partir de ses
  caracteristiques. Modele entraine sur {feature_info['n_rows']:,} annonces GetAround.</p>

  <div class="badges">
    <span class="badge">Modele<b>{escape(MODEL_NAME)}</b></span>
    <span class="badge">R² (test)<b>{METRICS['test_r2']:.3f}</b></span>
    <span class="badge">Erreur moyenne<b>{METRICS['test_mae']:.2f} €</b></span>
    <span class="badge">RMSE (test)<b>{METRICS['test_rmse']:.2f} €</b></span>
  </div>

  <h2>POST /predict</h2>
  <div class="card">
    <p><span class="m m-post">POST</span> &nbsp;<code>/predict</code> —
    <code>Content-Type: application/json</code></p>
    <p>Accepte un <b>lot</b> de voitures. Chaque ligne peut s'ecrire de deux facons.</p>

    <h3>Format positionnel — {len(ALL_COLS)} valeurs dans l'ordre</h3>
    <pre><code>{{"input": [{escape(example_positional)}]}}</code></pre>

    <h3>Format nomme — plus lisible, ordre indifferent</h3>
    <pre><code>{{"input": [{escape(example_named)}]}}</code></pre>

    <h3>Reponse</h3>
    <pre><code>{{"prediction": [111.88]}}</code></pre>
  </div>

  <h2>Colonnes attendues</h2>
  <div class="card scroll">
    <table>
      <thead><tr><th>#</th><th>Colonne</th><th>Type</th><th>Domaine</th></tr></thead>
      <tbody>{cols_rows}</tbody>
    </table>
  </div>

  <h2>Exemples d'appel</h2>
  <div class="card">
    <h3>curl</h3>
    <pre><code>curl -X POST "$API_URL/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{"input": [{escape(example_positional)}]}}'</code></pre>

    <h3>Python</h3>
    <pre><code>import requests

r = requests.post(
    "$API_URL/predict",
    json={{"input": [{escape(example_named)}]}},
)
print(r.json())          # -&gt; {{'prediction': [111.88]}}</code></pre>
  </div>

  <h2>Autres endpoints</h2>
  <div class="card">
    <p><span class="m m-get">GET</span> &nbsp;<code>/health</code> — etat du service.
       Reponse : <code>{{"status": "ok", "model": "loaded"}}</code></p>
    <p><span class="m m-get">GET</span> &nbsp;<code>/model-info</code> — ordre des colonnes,
       modalites categorielles acceptees et performances du modele.</p>
    <p><span class="m m-get">GET</span> &nbsp;<code><a href="/docs">/docs</a></code> —
       Swagger UI interactif, genere par FastAPI.</p>
  </div>

  <h2>Codes de reponse</h2>
  <div class="card">
    <table>
      <thead><tr><th>Code</th><th>Signification</th></tr></thead>
      <tbody>
        <tr><td><code>200</code></td><td>Prediction retournee.</td></tr>
        <tr><td><code>422</code></td><td>Charge utile invalide : nombre de valeurs incorrect,
            colonne manquante, ou type non convertible. Le detail precise la colonne fautive.</td></tr>
        <tr><td><code>500</code></td><td>Erreur interne du modele.</td></tr>
      </tbody>
    </table>
    <p style="color:var(--muted);margin-bottom:0">Une modalite categorielle inconnue
    (une marque absente du jeu d'entrainement, par exemple) n'est <b>pas</b> rejetee :
    elle est encodee a zero. La prediction reste exploitable, mais moins precise.</p>
  </div>

  <footer>GetAround Analysis · Jedha Bootcamp Bloc 5 · modele
  <code>{escape(MODEL_NAME)}</code> entraine avec scikit-learn, suivi via MLflow.</footer>
</div>
</body>
</html>""")


def _kind(col: str) -> str:
    if col in NUM_COLS:
        return "entier"
    if col in BOOL_COLS:
        return "booleen"
    return "chaine"


def _domain(col: str) -> str:
    if col in BOOL_COLS:
        return "<code>true</code> / <code>false</code>"
    if col in CAT_COLS:
        vals = CAT_VALUES[col]
        shown = ", ".join(f"<code>{escape(v)}</code>" for v in vals[:6])
        return shown + (f" … <i>({len(vals)} modalites)</i>" if len(vals) > 6 else "")
    return "<i>entier positif</i>"


def _json(obj, indent=None) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=indent)


# ─────────────────────────────────────────────────────────────────────────────
# Lancement direct : `python api.py`
# ─────────────────────────────────────────────────────────────────────────────
# Render injecte la variable d'environnement PORT (10000 par defaut) et impose
# une ecoute sur 0.0.0.0. En local et sous Docker classique, PORT est absent et
# le service retombe sur 8000 : le comportement historique est inchange.
if __name__ == "__main__":  # pragma: no cover
    import os

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
