# -*- coding: utf-8 -*-
"""
GetAround — entrainement du modele de pricing.

Compare plusieurs regresseurs en validation croisee, trace chaque essai dans MLflow,
puis sauvegarde le meilleur pipeline pour l'API FastAPI.

Usage :  python api/train.py     (depuis n'importe quel repertoire)

Artefacts produits :
    api/model.pkl         pipeline sklearn complet (preprocessing + modele)
    api/feature_info.pkl  ordre des colonnes et metadonnees attendus par l'API
    mlruns/               historique des experiences MLflow
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "projet_getarond_3.csv"
API_DIR = ROOT / "api"

TARGET = "rental_price_per_day"
RANDOM_STATE = 42

EXPERIMENT = "getaround-pricing"
# MLflow >= 3.10 a place le backend "fichier" (./mlruns) en maintenance et leve une
# exception a l'usage : le backend local recommande est desormais SQLite. Les artefacts
# (modeles serialises) restent ecrits dans ./mlruns.
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
ARTIFACT_URI = (ROOT / "mlruns").as_uri()


# ─────────────────────────────────────────────────────────────────────────────
# 1 · Chargement et nettoyage
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Charge le dataset pricing et retire les lignes physiquement impossibles."""
    # Le fichier est en UTF-8 : le lire en latin-1 transformerait silencieusement
    # 'Citroen' en mojibake, or cette marque represente 20 % des lignes -- elle
    # deviendrait alors une categorie inconnue a l'inference.
    # La premiere colonne est un index sans nom -> index_col=0.
    df = pd.read_csv(DATA, encoding="utf-8", index_col=0)
    n_raw = len(df)

    df = df.dropna(subset=[TARGET])

    invalid = (df["mileage"] < 0) | (df["engine_power"] <= 0)
    df = df[~invalid]

    print(f"Donnees   : {n_raw} lignes brutes -> {len(df)} conservees "
          f"({invalid.sum()} valeurs aberrantes retirees : kilometrage negatif "
          f"ou puissance nulle)")
    return df.reset_index(drop=True)


def build_preprocessor(X: pd.DataFrame):
    """ColumnTransformer : numeriques standardisees, booleens bruts, categorielles one-hot."""
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(f"Features  : {len(num_cols)} numeriques, {len(bool_cols)} booleennes, "
          f"{len(cat_cols)} categorielles")

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            # Les booleens sont deja sur une echelle 0/1 : les standardiser n'apporte rien.
            ("bool", "passthrough", bool_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    return pre, num_cols, bool_cols, cat_cols


# ─────────────────────────────────────────────────────────────────────────────
# 2 · Entrainement
# ─────────────────────────────────────────────────────────────────────────────
CANDIDATES = {
    "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "RandomForest": RandomForestRegressor(
        n_estimators=300, max_depth=14, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.9, random_state=RANDOM_STATE,
    ),
    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=6,
        early_stopping=True, random_state=RANDOM_STATE,
    ),
}


def main() -> None:
    df = load_data()

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    preprocessor, num_cols, bool_cols, cat_cols = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Split     : {len(X_train)} train / {len(X_test)} test")
    print(f"Cible     : moyenne {y.mean():.1f} EUR/jour, ecart-type {y.std():.1f}\n")

    (ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_URI)
    mlflow.set_experiment(EXPERIMENT)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    leaderboard, best = [], {"cv_rmse": np.inf}

    for name, model in CANDIDATES.items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

            # Selection sur la validation croisee du train : le test reste intouche.
            cv_rmse = -cross_val_score(
                pipe, X_train, y_train, cv=cv,
                scoring="neg_root_mean_squared_error", n_jobs=-1,
            )

            pipe.fit(X_train, y_train)
            pred_test = pipe.predict(X_test)
            pred_train = pipe.predict(X_train)

            metrics = {
                "cv_rmse": float(cv_rmse.mean()),
                "cv_rmse_std": float(cv_rmse.std()),
                "train_rmse": float(np.sqrt(mean_squared_error(y_train, pred_train))),
                "train_r2": float(r2_score(y_train, pred_train)),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
                "test_mae": float(mean_absolute_error(y_test, pred_test)),
                "test_r2": float(r2_score(y_test, pred_test)),
            }

            mlflow.log_param("model_type", name)
            mlflow.log_params({f"model__{k}": v for k, v in model.get_params().items()})
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_features_raw", X.shape[1])
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, name="model", input_example=X_train.head(3))

            print(
                f"{name:22s} CV-RMSE {metrics['cv_rmse']:6.2f} "
                f"(+/-{metrics['cv_rmse_std']:.2f})  |  "
                f"test RMSE {metrics['test_rmse']:6.2f}  "
                f"MAE {metrics['test_mae']:5.2f}  R2 {metrics['test_r2']:.4f}"
            )

            leaderboard.append({"model": name, **metrics})
            if metrics["cv_rmse"] < best["cv_rmse"]:
                best = {"name": name, "pipeline": pipe, **metrics}

    # ─────────────────────────────────────────────────────────────────────────
    # 3 · Selection et sauvegarde
    # ─────────────────────────────────────────────────────────────────────────
    print(
        f"\nMeilleur modele : {best['name']} "
        f"(CV-RMSE {best['cv_rmse']:.2f} | test R2 {best['test_r2']:.4f} | "
        f"test MAE {best['test_mae']:.2f} EUR)"
    )

    # Reentrainement sur l'integralite des donnees : le modele servi en production
    # beneficie aussi des 20 % mis de cote pour l'evaluation.
    final = Pipeline([
        ("preprocessor", preprocessor),
        ("model", CANDIDATES[best["name"]]),
    ]).fit(X, y)

    joblib.dump(final, API_DIR / "model.pkl")

    feature_info = {
        "all_cols": X.columns.tolist(),
        "num_cols": num_cols,
        "bool_cols": bool_cols,
        "cat_cols": cat_cols,
        "cat_values": {c: sorted(X[c].dropna().unique().tolist()) for c in cat_cols},
        "target": TARGET,
        "n_rows": int(len(df)),
        "model_name": best["name"],
        "metrics": {k: best[k] for k in ("cv_rmse", "test_rmse", "test_mae", "test_r2")},
        # Types Python natifs (et non numpy) pour rester serialisable en JSON cote API.
        "example": [v.item() if hasattr(v, "item") else v for v in X.iloc[0]],
        "price_range": [float(y.min()), float(y.max())],
    }
    joblib.dump(feature_info, API_DIR / "feature_info.pkl")

    pd.DataFrame(leaderboard).to_csv(ROOT / "data" / "model_leaderboard.csv", index=False)

    print(f"\nArtefacts :")
    print(f"  {API_DIR / 'model.pkl'}")
    print(f"  {API_DIR / 'feature_info.pkl'}")
    print(f"  {ROOT / 'data' / 'model_leaderboard.csv'}")
    print(f"  {ROOT / 'mlflow.db'}  (mlflow ui --backend-store-uri sqlite:///mlflow.db)")
    print(f"\nOrdre des colonnes attendu par /predict :\n  {X.columns.tolist()}")


if __name__ == "__main__":
    main()
