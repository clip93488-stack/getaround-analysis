# GetAround — Dashboard

Interface web du projet, avec deux vues choisies dans la barre latérale :

| Vue | Public | Contenu |
|---|---|---|
| **Délai minimum entre locations** | Product Manager | Chiffre l'effet d'un délai minimum entre deux locations (`threshold`) appliqué à un périmètre (`scope`) et rend visible l'arbitrage entre bénéfice conducteur et coût en revenus |
| **Estimation de prix** | Propriétaires | Formulaire qui appelle l'API de prédiction (`POST /predict`) et affiche le prix journalier estimé |

## Vue « Délai minimum entre locations »

### Le problème

Les conducteurs rendent parfois la voiture en retard, ce qui pénalise le conducteur suivant :
attente, voire annulation. Imposer un délai minimum entre deux locations réduit ces frictions —
mais bloque aussi des réservations qui auraient eu lieu.

### Ce que montre la vue

| Onglet | Contenu |
|---|---|
| **Arbitrage bénéfice / coût** | Courbes cas résolus vs locations bloquées, frontière coût/bénéfice, grille de décision |
| **Analyse des retards** | Distribution des retards, comparaison Connect / mobile, impact sur les annulations |
| **Recommandation** | Le réglage retenu et sa justification chiffrée |
| **Méthodologie** | Définitions, filtrage appliqué, limites de l'analyse |

### Recommandation

**Seuil de 120 minutes sur l'ensemble de la flotte** : 82,6 % des cas problématiques résolus
pour 3,1 % des locations bloquées. Le périmètre `connect` plafonne à 32 % de cas résolus quel
que soit le seuil, car le flux `mobile` concentre 80 % du volume et 1,8× plus de cas problématiques.

### Méthode

Une location est **problématique** quand le conducteur *précédent* rend la voiture après l'heure
de début prévue de la location suivante :

```
retard_utile(r) = delay_at_checkout(précédente) − time_delta(r)
r est problématique  ⟺  retard_utile(r) > 0
```

Le retard mobilisé est celui de la location **précédente**, récupéré par jointure
`previous_ended_rental_id` → `rental_id`.

Pour un seuil `T` : une location dont l'écart planifié est `< T` est **bloquée** (coût) ;
un cas problématique dont l'écart est `< T` est **résolu** (bénéfice), la réservation
conflictuelle n'ayant pas lieu.

## Vue « Estimation de prix »

Le formulaire (marque, carrosserie, carburant, couleur, kilométrage, puissance, équipements)
envoie la voiture à l'API et affiche :

- le **prix journalier estimé**, comparé au prix moyen observé dans les données ;
- l'**erreur moyenne du modèle** (MAE sur le jeu de test), lue sur `GET /model-info` ;
- l'**échange avec l'API** (requête et réponse JSON), dans un volet repliable.

Le dashboard ne charge pas le modèle : le calcul est entièrement fait par l'API.

Deux choix de conception, liés à l'hébergement gratuit où l'API est mise en veille après 15 min
d'inactivité (réveil de 30 à 60 s) :

- les listes du formulaire viennent de `data/pricing_choices.json`, écrit par `api/train.py`, et non
  d'un appel à l'API au chargement de la page : le formulaire s'affiche immédiatement ;
- à l'ouverture de la vue, un appel à `/health` réveille l'API en arrière-plan ; le calcul du prix
  retente jusqu'à trois fois si la passerelle répond 502/503/504 pendant le réveil.

L'adresse de l'API se règle avec la variable d'environnement `API_URL` (par défaut :
`https://getaround-api-ztfm.onrender.com`).

## Lancement local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Pour utiliser une API lancée en local (`uvicorn api:app --port 8000` dans `api/`) :

```bash
API_URL=http://localhost:8000 streamlit run app.py
```

Le dashboard cherche `projet_getaround_2.xlsx` dans `dashboard/data/` puis dans `../data/` :
il fonctionne donc aussi bien depuis la racine du dépôt que depuis `dashboard/` seul — c'est
ce dernier cas qui s'applique sur Render, dont le `rootDir` est fixé à `dashboard`.

## Tests

Depuis la racine du dépôt :

```bash
pip install -r requirements-dev.txt
pytest tests/test_price_estimator.py
```

## Déploiement

Service `getaround-dashboard` du `render.yaml` à la racine du dépôt (runtime `python`,
`rootDir: dashboard`). Streamlit est démarré sur le `$PORT` injecté par Render :

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

---

*GetAround Analysis · Jedha Bootcamp Bloc 5*
