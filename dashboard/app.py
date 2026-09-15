# -*- coding: utf-8 -*-
"""
GetAround — Dashboard d'aide a la decision sur le delai minimum entre deux locations.

Outil destine au Product Manager : il simule l'effet d'un delai minimum (`threshold`)
applique sur un perimetre (`scope`) et chiffre l'arbitrage benefice / cout.

Lancement local :  streamlit run app.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GetAround — Delai minimum",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONNECT, MOBILE = "#7C3AED", "#EA580C"
OK, WARN, BAD, INFO = "#22C55E", "#F59E0B", "#EF4444", "#0EA5E9"
TYPE_COLORS = {"connect": CONNECT, "mobile": MOBILE}

RECOMMENDED_THRESHOLD = 120
RECOMMENDED_SCOPE = "all"

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
      h1, h2, h3 {letter-spacing: -0.02em;}
      div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 12px;
        padding: 14px 16px;
      }
      div[data-testid="stMetricLabel"] {opacity: 0.75;}
      .callout {
        border-left: 4px solid #22C55E;
        background: rgba(34, 197, 94, 0.08);
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin: 6px 0 18px 0;
      }
      .callout-warn {border-left-color: #F59E0B; background: rgba(245, 158, 11, 0.08);}
      .callout-info {border-left-color: #0EA5E9; background: rgba(14, 165, 233, 0.08);}
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Chargement & preparation des donnees
# ─────────────────────────────────────────────────────────────────────────────
def _find_data_file() -> Path:
    """Localise le classeur, depuis la racine du depot (../data) comme depuis dashboard/ (./data)."""
    here = Path(__file__).parent
    for candidate in (
        here / "data" / "projet_getaround_2.xlsx",
        here.parent / "data" / "projet_getaround_2.xlsx",
        Path("data/projet_getaround_2.xlsx"),
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "projet_getaround_2.xlsx introuvable — attendu dans dashboard/data/ ou data/"
    )


@st.cache_data(show_spinner="Chargement des donnees...")
def load_data():
    """Charge les locations et derive le perimetre des locations consecutives.

    Le retard qui gene le conducteur d'une location est celui de la location
    PRECEDENTE : on le recupere par jointure previous_ended_rental_id -> rental_id.
    """
    df = pd.read_excel(_find_data_file(), sheet_name="rentals_data", engine="openpyxl")

    delay_by_id = df.set_index("rental_id")["delay_at_checkout_in_minutes"]

    # Locations ayant une location precedente a moins de 12 h
    consec = df[df["previous_ended_rental_id"].notna()].copy()
    consec["previous_delay"] = consec["previous_ended_rental_id"].map(delay_by_id)

    # Perimetre d'analyse : celles dont le retard precedent a bien ete mesure
    analysed = consec.dropna(subset=["previous_delay"]).copy()
    analysed["overlap"] = (
        analysed["previous_delay"] - analysed["time_delta_with_previous_rental_in_minutes"]
    )
    analysed["is_problematic"] = analysed["overlap"] > 0

    return df, consec, analysed


df, consec, analysed = load_data()

N_TOTAL = len(df)
N_CONSEC = len(consec)
N_PROBLEMATIC = int(analysed["is_problematic"].sum())


def simulate(threshold: int, scope: str) -> dict:
    """Chiffre l'effet d'un delai minimum `threshold` applique au perimetre `scope`.

    - bloquees : locations du perimetre dont l'ecart planifie est < threshold ;
      elles n'auraient pas pu etre reservees (c'est le cout de la regle).
    - resolues : cas problematiques du perimetre qui disparaissent, la reservation
      conflictuelle n'ayant pas lieu (c'est le benefice).
    """
    if scope == "connect":
        pool = consec[consec["checkin_type"] == "connect"]
        probs = analysed[analysed["checkin_type"] == "connect"]
    else:
        pool, probs = consec, analysed

    gap = "time_delta_with_previous_rental_in_minutes"
    blocked = int((pool[gap] < threshold).sum())
    solved = int((probs["is_problematic"] & (probs[gap] < threshold)).sum())

    return {
        "scope": scope,
        "threshold": threshold,
        "nb_blocked": blocked,
        "pct_blocked_consec": blocked / N_CONSEC * 100,
        "pct_blocked_total": blocked / N_TOTAL * 100,
        "nb_solved": solved,
        "pct_solved": solved / N_PROBLEMATIC * 100 if N_PROBLEMATIC else 0.0,
        "nb_remaining": N_PROBLEMATIC - solved,
        "efficiency": solved / blocked if blocked else np.nan,
    }


@st.cache_data
def simulation_grid() -> pd.DataFrame:
    return pd.DataFrame(
        [simulate(t, s) for s in ("all", "connect") for t in range(0, 721, 30)]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — les deux leviers de decision
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parametres de la regle")

    threshold = st.slider(
        "Delai minimum entre deux locations",
        min_value=0,
        max_value=720,
        value=RECOMMENDED_THRESHOLD,
        step=30,
        format="%d min",
        help="Une voiture n'est plus reservable si l'ecart avec la location "
        "precedente est inferieur a ce seuil.",
    )
    st.caption(f"Soit **{threshold // 60} h {threshold % 60:02d} min**")

    scope = st.radio(
        "Perimetre d'application",
        options=["all", "connect"],
        format_func=lambda s: (
            "Toutes les voitures" if s == "all" else "Voitures Connect uniquement"
        ),
        help="La regle s'applique-t-elle a l'ensemble de la flotte, ou seulement "
        "aux voitures equipees de la technologie Connect ?",
    )

    st.divider()
    st.caption(
        f"**Perimetre de l'analyse**\n\n"
        f"- {N_TOTAL:,} locations au total\n"
        f"- {N_CONSEC:,} avec une location precedente < 12 h\n"
        f"- {N_PROBLEMATIC:,} cas problematiques identifies"
    )
    st.caption(
        "Un cas est **problematique** quand le conducteur precedent rend la voiture "
        "apres l'heure de debut prevue de la location suivante."
    )

sim = simulate(threshold, scope)
grid = simulation_grid()

# ─────────────────────────────────────────────────────────────────────────────
# En-tete
# ─────────────────────────────────────────────────────────────────────────────
st.title("🚗 GetAround — Delai minimum entre deux locations")
st.markdown(
    "**Outil d'aide a la decision.** Les conducteurs rendent parfois la voiture en retard, "
    "ce qui penalise le conducteur suivant. Imposer un delai minimum entre deux locations "
    "reduit ces frictions — mais bloque aussi des reservations. "
    "Reglez le seuil et le perimetre dans la barre laterale pour chiffrer l'arbitrage."
)

# ─────────────────────────────────────────────────────────────────────────────
# KPI
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Impact de la regle selectionnee")

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "✅ Cas resolus",
    f"{sim['nb_solved']:,}",
    f"{sim['pct_solved']:.1f}% des {N_PROBLEMATIC} cas",
)
k2.metric(
    "⚠️ Cas restants",
    f"{sim['nb_remaining']:,}",
    f"{100 - sim['pct_solved']:.1f}% non traites",
    delta_color="inverse",
)
k3.metric(
    "🔒 Locations bloquees",
    f"{sim['nb_blocked']:,}",
    f"{sim['pct_blocked_total']:.2f}% du volume total",
    delta_color="inverse",
)
k4.metric(
    "⚖️ Rendement",
    "—" if np.isnan(sim["efficiency"]) else f"{sim['efficiency']:.2f}",
    "cas resolu par location bloquee",
    delta_color="off",
)

# Lecture automatique du reglage courant
if threshold == 0:
    verdict, css = (
        "**Aucune regle active.** Les 218 cas problematiques subsistent en totalite.",
        "callout-warn",
    )
elif scope == "connect":
    verdict, css = (
        f"Le perimetre **Connect** plafonne a **{grid[grid.scope == 'connect'].pct_solved.max():.0f}% "
        "de cas resolus**, quel que soit le seuil : le flux mobile concentre 80 % du volume "
        "et 1,8x plus de cas problematiques. Elargir a toutes les voitures traite bien plus "
        "de cas a cout comparable.",
        "callout-warn",
    )
elif threshold >= 300:
    verdict, css = (
        f"Seuil eleve : **{sim['pct_solved']:.0f}% des cas resolus**, mais "
        f"**{sim['nb_blocked']:,} locations bloquees** ({sim['pct_blocked_total']:.1f}% du volume). "
        "Au-dela de 2 h, chaque point de benefice supplementaire coute de plus en plus cher — "
        "la courbe des cas resolus a sature.",
        "callout-warn",
    )
else:
    verdict, css = (
        f"Reglage equilibre : **{sim['pct_solved']:.0f}% des cas problematiques resolus** "
        f"pour seulement **{sim['pct_blocked_total']:.1f}% des locations bloquees**.",
        "callout",
    )

# Le callout est du HTML : on convertit le gras markdown en balises <b>
verdict = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", verdict)
st.markdown(f'<div class="callout {css}">{verdict}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Arbitrage
# ─────────────────────────────────────────────────────────────────────────────
tab_tradeoff, tab_delays, tab_reco, tab_method = st.tabs(
    ["📈 Arbitrage benefice / cout", "⏱️ Analyse des retards", "🎯 Recommandation", "🔬 Methodologie"]
)

with tab_tradeoff:
    left, right = st.columns([3, 2])

    with left:
        st.markdown("##### Benefice et cout selon le seuil")
        current = grid[grid.scope == scope]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=current.threshold,
                y=current.pct_solved,
                name="Cas problematiques resolus",
                line=dict(color=OK, width=3),
                fill="tozeroy",
                fillcolor="rgba(34, 197, 94, 0.12)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=current.threshold,
                y=current.pct_blocked_consec,
                name="Locations consecutives bloquees",
                line=dict(color=BAD, width=3, dash="dot"),
            )
        )
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="#64748B",
            annotation_text=f"{threshold} min",
            annotation_position="top",
        )
        fig.update_layout(
            xaxis_title="Seuil (minutes)",
            yaxis_title="%",
            height=420,
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=20, l=10, r=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("##### Frontiere cout / benefice")
        pts = grid[grid.threshold > 0].copy()
        pts["Perimetre"] = pts.scope.map(
            {"all": "Toutes les voitures", "connect": "Connect uniquement"}
        )

        fig2 = px.scatter(
            pts,
            x="pct_blocked_total",
            y="pct_solved",
            color="Perimetre",
            color_discrete_map={"Toutes les voitures": INFO, "Connect uniquement": CONNECT},
            hover_data={"threshold": True},
            labels={
                "pct_blocked_total": "Cout — % de toutes les locations bloquees",
                "pct_solved": "Benefice — % de cas resolus",
            },
        )
        fig2.add_trace(
            go.Scatter(
                x=[sim["pct_blocked_total"]],
                y=[sim["pct_solved"]],
                mode="markers",
                marker=dict(size=18, color=OK, line=dict(width=3, color="white")),
                name="Reglage actuel",
            )
        )
        fig2.update_layout(
            height=420,
            legend=dict(orientation="h", y=-0.25),
            margin=dict(t=20, l=10, r=10, b=10),
        )
        st.plotly_chart(fig2, width="stretch")
        st.caption("Plus un point est **haut et a gauche**, meilleur est le compromis.")

    st.markdown("##### Grille de decision")
    table = (
        grid[(grid.scope == scope) & (grid.threshold.isin([30, 60, 90, 120, 180, 240, 360, 480, 720]))]
        .assign(
            Seuil=lambda d: d.threshold.astype(str) + " min",
            **{
                "Cas resolus": lambda d: d.nb_solved.astype(str)
                + " (" + d.pct_solved.round(1).astype(str) + "%)",
                "Cas restants": lambda d: d.nb_remaining,
                "Locations bloquees": lambda d: d.nb_blocked.astype(str)
                + " (" + d.pct_blocked_total.round(2).astype(str) + "%)",
                "Rendement": lambda d: d.efficiency.round(3),
            },
        )
        .loc[:, ["Seuil", "Cas resolus", "Cas restants", "Locations bloquees", "Rendement"]]
    )
    st.dataframe(table, width="stretch", hide_index=True)

with tab_delays:
    st.markdown("##### D'ou vient le probleme ?")

    c1, c2 = st.columns(2)

    with c1:
        ended = df[df["state"] == "ended"]
        viz = ended[ended["delay_at_checkout_in_minutes"].between(-720, 720)]
        fig3 = px.histogram(
            viz,
            x="delay_at_checkout_in_minutes",
            color="checkin_type",
            nbins=80,
            barmode="overlay",
            opacity=0.75,
            color_discrete_map=TYPE_COLORS,
            labels={
                "delay_at_checkout_in_minutes": "Retard au checkout (min)",
                "checkin_type": "Canal",
            },
            title="Distribution des retards (fenetre ±12 h)",
        )
        fig3.add_vline(x=0, line_dash="dash", line_color="#64748B")
        fig3.update_layout(height=380, bargap=0.02, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig3, width="stretch")
        st.caption(
            "95 % des restitutions tiennent dans cette fenetre. Les extremes "
            "(jusqu'a 49 jours) sont conserves dans les calculs mais exclus de ce graphique."
        )

    with c2:
        by_type = (
            analysed.groupby("checkin_type")
            .agg(
                n_consecutives=("is_problematic", "size"),
                n_problematiques=("is_problematic", "sum"),
            )
            .assign(taux=lambda d: (d.n_problematiques / d.n_consecutives * 100).round(1))
            .reset_index()
        )
        fig4 = px.bar(
            by_type,
            x="checkin_type",
            y="taux",
            text="taux",
            color="checkin_type",
            color_discrete_map=TYPE_COLORS,
            labels={"checkin_type": "Canal", "taux": "% de cas problematiques"},
            title="Taux de cas problematiques par canal",
        )
        fig4.update_traces(texttemplate="%{text}%", textposition="outside")
        fig4.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig4, width="stretch")
        st.caption(
            "Le flux **mobile** est 1,8x plus problematique que **Connect** — et represente "
            "80 % du volume. C'est l'argument central en faveur du perimetre `all`."
        )

    st.markdown("##### Le retard du conducteur precedent se paie en annulations")
    cancel = (
        analysed.groupby("is_problematic")["state"]
        .value_counts(normalize=True)
        .unstack()
        .mul(100)
        .round(1)
    )
    cancel.index = ["Location normale", "Conducteur precedent en retard"]

    c3, c4 = st.columns([2, 1])
    with c3:
        fig5 = px.bar(
            cancel.reset_index().melt(id_vars="index", var_name="Etat", value_name="pct"),
            x="index",
            y="pct",
            color="Etat",
            barmode="group",
            text="pct",
            color_discrete_map={"canceled": BAD, "ended": OK},
            labels={"index": "", "pct": "% des locations"},
        )
        fig5.update_traces(texttemplate="%{text}%", textposition="outside")
        fig5.update_layout(height=340, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig5, width="stretch")
    with c4:
        st.metric(
            "Taux d'annulation",
            f"{cancel.loc['Conducteur precedent en retard', 'canceled']:.1f}%",
            f"vs {cancel.loc['Location normale', 'canceled']:.1f}% en temps normal",
            delta_color="inverse",
        )
        st.markdown(
            '<div class="callout callout-info">Le retard du conducteur precedent fait bondir '
            "les annulations de <b>+52 % en relatif</b>. Le probleme ne degrade pas seulement "
            "l'experience : il detruit du chiffre d'affaires.</div>",
            unsafe_allow_html=True,
        )

with tab_reco:
    st.markdown("### Recommandation : **120 minutes**, perimetre **toutes les voitures**")

    reco = simulate(RECOMMENDED_THRESHOLD, RECOMMENDED_SCOPE)
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Cas resolus", f"{reco['nb_solved']}", f"{reco['pct_solved']:.1f}% du probleme")
    r2.metric("Cas restants", f"{reco['nb_remaining']}")
    r3.metric("Locations bloquees", f"{reco['nb_blocked']}", f"{reco['pct_blocked_total']:.1f}% du volume")
    r4.metric("Rendement", f"{reco['efficiency']:.2f}", "cas / location bloquee", delta_color="off")

    st.markdown(
        """
#### Pourquoi 120 minutes

- **vs 60 min** — +15,6 pts de cas resolus (67 % → 83 %) pour seulement +1,2 pt de locations
  bloquees. C'est le meilleur rapport marginal de toute la grille.
- **vs 180 min** — +7,3 pts de benefice seulement, pour +1,0 pt de cout : on entre dans la zone
  de rendement decroissant. **90 min** reste une alternative defendable si la priorite est de
  preserver les revenus (79 % de cas resolus pour 2,7 % de locations bloquees).

#### Pourquoi toutes les voitures plutot que Connect

Le perimetre Connect **plafonne a 32 % de cas resolus**, quel que soit le seuil retenu.
Le flux mobile concentre 80 % du volume et affiche un taux de cas problematiques 1,8x superieur :
l'exclure du perimetre revient a ignorer l'essentiel du probleme.

#### Precautions

Ces chiffres reposent sur **1 729 locations consecutives** avec retard precedent mesure, soit 8 %
du dataset. Le modele suppose par ailleurs qu'une location bloquee est une location **perdue** —
hypothese volontairement pessimiste, puisqu'en pratique une partie des conducteurs decalerait
simplement sa reservation. Le cout reel est donc vraisemblablement **inferieur** aux 3,1 % annonces.

#### Prochaine etape

Un **A/B test** sur un sous-ensemble de voitures, avec suivi du taux d'annulation et du revenu par
voiture, permettrait de valider ces estimations en conditions reelles avant generalisation.
        """
    )

with tab_method:
    st.markdown(
        f"""
#### Definition d'un cas problematique

Pour une location *r* precedee d'une location *p* sur la meme voiture :

```
retard_utile(r) = delay_at_checkout(p) − time_delta(r)
r est problematique  ⟺  retard_utile(r) > 0
```

Le retard qui gene le conducteur d'une location est celui de la location **precedente**.
Il est recupere par jointure `previous_ended_rental_id` → `rental_id` : comparer le
`delay_at_checkout_in_minutes` d'une ligne a son propre `time_delta_with_previous_rental_in_minutes`
confronterait deux grandeurs qui ne portent pas sur la meme location.

#### Modelisation de la regle

Pour un seuil `T` sur un perimetre `S` :

| Metrique | Definition |
|---|---|
| Locations bloquees | locations de `S` dont `time_delta < T` — elles n'auraient pas pu etre reservees |
| Cas resolus | cas problematiques de `S` dont `time_delta < T` — la reservation conflictuelle n'a pas lieu |
| Cas restants | cas problematiques dont `time_delta ≥ T` — la regle ne les empeche pas |

#### Filtrage applique

| Etape | Locations |
|---|---|
| Dataset complet | {N_TOTAL:,} |
| Avec une location precedente < 12 h | {N_CONSEC:,} |
| Dont retard precedent mesure (perimetre d'analyse) | {len(analysed):,} |
| Dont cas problematiques | {N_PROBLEMATIC:,} |

Les {N_CONSEC - len(analysed):,} locations dont le retard precedent n'a pas ete enregistre sont
**exclues** plutot qu'imputees a zero, ce qui sous-estimerait le probleme.

`previous_ended_rental_id` est `NULL` des que l'ecart avec la location precedente depasse 12 h :
c'est ce qui borne naturellement l'analyse a un seuil maximum de 720 minutes.
        """
    )

    with st.expander("Apercu du perimetre d'analyse"):
        st.dataframe(
            analysed[
                [
                    "rental_id",
                    "car_id",
                    "checkin_type",
                    "state",
                    "time_delta_with_previous_rental_in_minutes",
                    "previous_delay",
                    "overlap",
                    "is_problematic",
                ]
            ].head(200),
            width="stretch",
            hide_index=True,
        )

st.divider()
st.caption("GetAround Analysis · Jedha Bootcamp Bloc 5 · Donnees : 21 310 locations")
