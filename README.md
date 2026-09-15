# GetAround Analysis — Jedha Bloc 5

Analyse du **délai minimum entre deux locations** et **modèle de pricing** pour GetAround,
le « Airbnb de la voiture ».

**Stack** — Python · Pandas · Scikit-learn · MLflow · Streamlit · FastAPI · Docker · Render

---

## Contexte

Deux problèmes distincts, deux livrables.

**1 · Les retards de restitution.** Les conducteurs rendent parfois la voiture en retard, ce qui
pénalise le conducteur suivant. GetAround envisage d'imposer un **délai minimum** entre deux
locations. Le Product Manager doit arbitrer deux paramètres : la durée du seuil (`threshold`)
et son périmètre (`scope` : toute la flotte, ou seulement les voitures Connect).

**2 · La fixation des prix.** Les propriétaires ont besoin d'une estimation du prix journalier
de location, servie par une API.

## Livrables

| Livrable | URL |
|---|---|
| Dashboard Streamlit | `https://getaround-dashboard.onrender.com` |
| API FastAPI `/predict` | `https://getaround-api.onrender.com/predict` |
| Documentation Swagger | `https://getaround-api.onrender.com/docs` |
| Documentation rédigée | `https://getaround-api.onrender.com/docs-custom` |

> Render suffixe le nom du service s'il est déjà pris (`getaround-api-a1b2`). Reporter les URL
> réellement attribuées après le premier déploiement (cf. [Déploiement](#déploiement)).

---

## Résultats

### Analyse du délai minimum

Une location est **problématique** quand le conducteur *précédent* rend la voiture après l'heure
de début prévue de la location suivante. Sur 21 310 locations, 1 841 ont une location précédente
à moins de 12 h, dont 1 729 avec un retard précédent mesuré : **218 cas problématiques (12,6 %)**.

**Recommandation : seuil de 120 minutes sur l'ensemble de la flotte.**

| Indicateur | Valeur |
|---|---|
| Cas problématiques résolus | **180 / 218 — 82,6 %** |
| Locations bloquées | 666 — **3,1 % du volume total** |
| Rendement | 0,27 cas résolu par location bloquée |

Trois constats structurent cette recommandation :

- **La courbe de bénéfice sature vers 2 h.** Passer de 60 à 120 min gagne 15,6 points de cas
  résolus pour 1,2 point de locations bloquées ; passer de 120 à 180 min n'en gagne plus que 7,3
  pour 1,0 point. Au-delà, on n'ajoute pratiquement plus que du coût.
- **Le périmètre `connect` plafonne à 32 % de cas résolus**, quel que soit le seuil. Le flux
  `mobile` concentre 80 % du volume et affiche un taux de cas problématiques 1,8× supérieur
  (15,9 % contre 8,7 %) : l'exclure revient à ignorer l'essentiel du problème.
- **Le retard se paie en annulations** — 17,0 % des locations dont le conducteur précédent était
  en retard sont annulées, contre 11,2 % sinon, soit **+52 % en relatif**.

### Modèle de pricing

`GradientBoostingRegressor` retenu parmi quatre candidats — **R² test 0,761, MAE 10,48 €/jour**
sur un prix moyen de 121 €.

| Modèle | CV-RMSE | RMSE test | MAE test | R² test |
|---|---|---|---|---|
| **GradientBoosting** | **16,75** | **16,17** | **10,48** | **0,761** |
| RandomForest | 16,79 | 17,15 | 10,83 | 0,731 |
| HistGradientBoosting | 16,83 | 17,18 | 11,06 | 0,730 |
| Ridge *(référence)* | 18,43 | 17,93 | 12,11 | 0,706 |

---

## Structure du dépôt

```
getaround/
├── notebooks/
│   └── eda_delay.ipynb       Analyse exploratoire + simulation seuil × périmètre
├── dashboard/
│   ├── app.py                Dashboard Streamlit
│   ├── data/                 Copie du dataset (service Render autonome)
│   ├── requirements.txt
│   └── README.md
├── api/
│   ├── api.py                Application FastAPI
│   ├── train.py              Entraînement + suivi MLflow
│   ├── model.pkl             Pipeline sérialisé (versionné)
│   ├── feature_info.pkl      Contrat d'entrée (versionné)
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── requirements.txt      Runtime de l'API (= versions d'entraînement)
│   ├── requirements-train.txt
│   └── README.md
├── data/
│   ├── projet_getaround_2.xlsx      Dataset retards
│   ├── projet_getarond_3.csv        Dataset pricing
│   ├── threshold_simulation.csv     Export de la simulation
│   └── model_leaderboard.csv        Comparaison des modèles
├── render.yaml              Blueprint Render : les deux services web
└── README.md
```

---

## Installation

```bash
pip install pandas numpy plotly scikit-learn mlflow joblib fastapi "uvicorn[standard]" openpyxl streamlit
```

Python **3.11 ou plus** est requis (`scikit-learn 1.9` et `numpy 2.4`).

## Utilisation

### Notebook d'analyse

```bash
jupyter lab notebooks/eda_delay.ipynb
```

### Dashboard

```bash
cd dashboard && streamlit run app.py
```

### Entraînement du modèle

```bash
python api/train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### API

```bash
cd api && uvicorn api:app --reload --port 8000
```

Puis :

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"input": [["Citroën", 140411, 100, "diesel", "black", "convertible", true, true, false, false, true, true, true]]}'
```

Réponse : `{"prediction":[111.88]}`

### Docker

```bash
docker build -t getaround-api ./api
docker run -p 8000:8000 getaround-api
curl http://localhost:8000/health
```

Le conteneur lit la variable d'environnement `PORT` et retombe sur `8000` si elle est absente :
`docker run -e PORT=10000 -p 10000:10000 getaround-api` sert la même image sur un autre port.

---

## Déploiement

Les deux services sont décrits dans **`render.yaml`** à la racine du dépôt : Render lit ce
*Blueprint* et crée les deux services web en une fois, sans configuration manuelle.

| Service | Type Render | `rootDir` | Démarrage |
|---|---|---|---|
| `getaround-dashboard` | runtime `python` | `dashboard` | `streamlit run app.py --server.port $PORT …` |
| `getaround-api` | runtime `docker` | `api` | `Dockerfile` → `uvicorn --port ${PORT:-8000}` |

Les deux tournent sur le plan **`free`**.

### Étape 0 — Prérequis

1. Pousser le dépôt sur GitHub (cf. section suivante).
2. Créer un compte sur <https://render.com> et y connecter son compte GitHub.

Les artefacts versionnés (`model.pkl` 691 Ko, `projet_getaround_2.xlsx` 734 Ko) restent sous le
seuil de 100 Mo de GitHub : **git-lfs n'est pas nécessaire**.

### Étape 1 — Créer le Blueprint

1. <https://dashboard.render.com/blueprints> → **New Blueprint Instance**.
2. Sélectionner le dépôt GitHub. Render détecte `render.yaml` à la racine.
3. Vérifier que les deux services apparaissent, puis **Apply**.

Le dashboard se construit en 3–5 min, l'image Docker de l'API en 5–8 min. Les URL publiques sont
de la forme `https://<nom-du-service>.onrender.com`.

### Étape 2 — Le port

Render injecte la variable d'environnement **`PORT`** (10000 par défaut) et exige une écoute sur
`0.0.0.0`. Aucun port n'est codé en dur dans le dépôt :

- le dashboard reçoit `--server.port $PORT --server.address 0.0.0.0` dans son `startCommand` ;
- l'API démarre sur `${PORT:-8000}` — la valeur de repli `8000` préserve le comportement en local
  et sous `docker run` sans variable.

`healthCheckPath: /health` fait échouer le déploiement de l'API si le modèle ne se charge pas.

### Étape 3 — Vérifier le déploiement

```bash
curl https://getaround-api.onrender.com/health
```

Réponse attendue : `{"status":"ok","model":"loaded","model_type":"GradientBoosting"}`

```bash
curl -X POST https://getaround-api.onrender.com/predict \
     -H "Content-Type: application/json" \
     -d '{"input": [["Citroën", 140411, 100, "diesel", "black", "convertible", true, true, false, false, true, true, true]]}'
```

Réponse attendue : `{"prediction":[111.88]}`

Enfin, ouvrir `/docs` (Swagger UI) et `/docs-custom` (documentation rédigée), puis reporter les
quatre URL obtenues dans le tableau [Livrables](#livrables) en tête de ce README.

> **Plan gratuit :** un service inactif depuis 15 min est mis en veille. La première requête
> qui le réveille peut prendre ~50 s — le premier `curl` après une pause n'est donc pas un échec.

### Mettre à jour un service

Render redéploie automatiquement à chaque `git push` sur la branche suivie. Grâce à `rootDir`,
un changement dans `dashboard/` ne reconstruit pas l'image de l'API, et inversement.

---

## Notes de méthode

**Jointure sur la location précédente.** Le retard qui gêne le conducteur d'une location est celui
de la location *précédente*, récupéré via `previous_ended_rental_id` → `rental_id`. Comparer le
`delay_at_checkout_in_minutes` d'une ligne à son propre `time_delta_with_previous_rental_in_minutes`
confronterait deux grandeurs qui ne portent pas sur la même location.

**Valeurs manquantes.** Les 112 locations consécutives dont le retard précédent n'a pas été
enregistré sont *exclues* plutôt qu'imputées à zéro, ce qui sous-estimerait le problème.

**Encodage du CSV pricing.** Le fichier est en UTF-8. Le lire en latin-1 transformerait
silencieusement `Citroën` en mojibake — or cette marque représente 20 % des lignes, et deviendrait
une catégorie inconnue à l'inférence.

**Portée des résultats.** L'analyse du seuil repose sur 1 729 locations, soit 8 % du dataset.
Le modèle suppose qu'une location bloquée est une location *perdue* — hypothèse pessimiste,
puisqu'en pratique une partie des conducteurs décalerait simplement sa réservation. Un **A/B test**
sur un sous-ensemble de voitures validerait ces estimations avant généralisation.

---

*GetAround Analysis · Jedha Bootcamp Bloc 5 · Industrialisation ML*
