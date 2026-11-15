# GetAround — Pricing API

API de prédiction du **prix de location journalier** d'une voiture GetAround, à partir de ses
caractéristiques. Modèle `GradientBoostingRegressor` entraîné sur 4 841 annonces.

| Métrique (jeu de test) | Valeur |
|---|---|
| R² | **0,761** |
| MAE | **10,48 €** / jour |
| RMSE | 16,17 € / jour |

## Endpoints

| Méthode | Route | Description |
|---|---|---|
| `POST` | `/predict` | Prédiction du prix journalier (lot supporté) |
| `GET` | `/model-info` | Ordre des colonnes, modalités acceptées, performances |
| `GET` | `/health` | État du service |
| `GET` | `/docs` | Documentation rédigée (page HTML) |
| `GET` | `/swagger` | Swagger UI interactif, généré par FastAPI |

`/docs-custom`, ancienne adresse de la documentation rédigée, redirige vers `/docs`.

## `POST /predict`

Chaque voiture s'écrit au choix en **positionnel** (13 valeurs dans l'ordre) ou en **nommé**.

### Positionnel

```bash
curl -X POST "$API_URL/predict" \
     -H "Content-Type: application/json" \
     -d '{"input": [["Citroën", 140411, 100, "diesel", "black", "convertible", true, true, false, false, true, true, true]]}'
```

### Nommé

```python
import requests

r = requests.post(f"{API_URL}/predict", json={"input": [{
    "model_key": "Citroën",
    "mileage": 140411,
    "engine_power": 100,
    "fuel": "diesel",
    "paint_color": "black",
    "car_type": "convertible",
    "private_parking_available": True,
    "has_gps": True,
    "has_air_conditioning": False,
    "automatic_car": False,
    "has_getaround_connect": True,
    "has_speed_regulator": True,
    "winter_tires": True,
}]})
print(r.json())          # {'prediction': [111.88]}
```

### Ordre des colonnes

```
model_key, mileage, engine_power, fuel, paint_color, car_type,
private_parking_available, has_gps, has_air_conditioning, automatic_car,
has_getaround_connect, has_speed_regulator, winter_tires
```

Les modalités acceptées pour chaque colonne catégorielle sont exposées par `GET /model-info`.

### Libellés catégoriels

Les libellés sont reconnus **sans tenir compte de la casse ni des accents** : `Citroen`,
`citroën` et `CITROEN` sont lus comme `Citroën` et donnent la même prédiction. Sans cette
normalisation, `Citroen` serait une marque inconnue du `OneHotEncoder` et la prédiction
perdrait l'effet de la marque.

Une modalité réellement inconnue (marque absente du jeu d'entraînement) n'est **pas** rejetée :
elle est encodée à zéro. La prédiction reste exploitable, mais moins précise.

### Codes de réponse

| Code | Signification |
|---|---|
| `200` | Prédiction retournée |
| `422` | Charge utile invalide — le détail précise la colonne fautive |
| `500` | Erreur interne du modèle |

## Entraînement

```bash
pip install -r requirements-train.txt
python train.py          # depuis n'importe quel répertoire
```

Le script compare quatre régresseurs en validation croisée 5-fold, trace chaque essai dans MLflow,
retient le meilleur sur la RMSE de validation croisée, puis le réentraîne sur l'intégralité des
données avant sérialisation.

| Modèle | CV-RMSE | RMSE test | MAE test | R² test |
|---|---|---|---|---|
| **GradientBoosting** | **16,75** | **16,17** | **10,48** | **0,761** |
| RandomForest | 16,79 | 17,15 | 10,83 | 0,731 |
| HistGradientBoosting | 16,83 | 17,18 | 11,06 | 0,730 |
| Ridge *(référence)* | 18,43 | 17,93 | 12,11 | 0,706 |

Consulter les runs :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Artefacts produits : `model.pkl` (pipeline complet), `feature_info.pkl` (contrat d'entrée),
`data/model_leaderboard.csv` et `dashboard/data/pricing_choices.json` (choix du formulaire du
dashboard). `model.pkl` et `feature_info.pkl` sont **versionnés** : ils sont nécessaires au
démarrage de l'API.

## Lancement local

```bash
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

## Tests

Depuis la racine du dépôt :

```bash
pip install -r requirements-dev.txt
pytest tests/test_api.py
```

## Docker

```bash
docker build -t getaround-api .
docker run -p 8000:8000 getaround-api
curl http://localhost:8000/health
```

Le port d'écoute n'est **jamais codé en dur** : le conteneur lit la variable
d'environnement `PORT` et retombe sur `8000` si elle est absente. C'est ce qui rend
la même image utilisable en local et sur Render, qui injecte son propre `PORT` :

```bash
docker run -e PORT=10000 -p 10000:10000 getaround-api
```

L'image est basée sur `python:3.12-slim` : `scikit-learn 1.9` et `numpy 2.4` exigent Python ≥ 3.11,
et les versions de `requirements.txt` doivent correspondre exactement à celles de l'entraînement,
faute de quoi `model.pkl` ne serait pas désérialisable.

L'image ne contient que `api.py`, `model.pkl` et `feature_info.pkl` : la page `/docs` est donc
générée par `api.py` lui-même, à partir du modèle réellement chargé.

---

*GetAround Analysis · Jedha Bootcamp Bloc 5*
