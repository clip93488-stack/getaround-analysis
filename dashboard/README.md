# 🚗 GetAround — Dashboard délai minimum

Outil d'aide à la décision destiné au Product Manager : il chiffre l'effet d'un **délai minimum
entre deux locations** (`threshold`) appliqué à un **périmètre** (`scope`), et rend visible
l'arbitrage entre le bénéfice conducteur et le coût en revenus.

## Le problème

Les conducteurs rendent parfois la voiture en retard, ce qui pénalise le conducteur suivant :
attente, voire annulation. Imposer un délai minimum entre deux locations réduit ces frictions —
mais bloque aussi des réservations qui auraient eu lieu.

## Ce que montre le dashboard

| Onglet | Contenu |
|---|---|
| **Arbitrage bénéfice / coût** | Courbes cas résolus vs locations bloquées, frontière coût/bénéfice, grille de décision |
| **Analyse des retards** | Distribution des retards, comparaison Connect / mobile, impact sur les annulations |
| **Recommandation** | Le réglage retenu et sa justification chiffrée |
| **Méthodologie** | Définitions, filtrage appliqué, limites de l'analyse |

## Recommandation

**Seuil de 120 minutes sur l'ensemble de la flotte** : 82,6 % des cas problématiques résolus
pour 3,1 % des locations bloquées. Le périmètre `connect` plafonne à 32 % de cas résolus quel
que soit le seuil, car le flux `mobile` concentre 80 % du volume et 1,8× plus de cas problématiques.

## Méthode

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

## Lancement local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Le dashboard cherche `projet_getaround_2.xlsx` dans `dashboard/data/` puis dans `../data/` :
il fonctionne donc aussi bien depuis la racine du dépôt que depuis `dashboard/` seul — c'est
ce dernier cas qui s'applique sur Render, dont le `rootDir` est fixé à `dashboard`.

## Déploiement

Service `getaround-dashboard` du `render.yaml` à la racine du dépôt (runtime `python`,
`rootDir: dashboard`). Streamlit est démarré sur le `$PORT` injecté par Render :

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

---

*GetAround Analysis · Jedha Bootcamp Bloc 5*
