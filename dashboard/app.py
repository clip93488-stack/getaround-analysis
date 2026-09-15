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
    page_title="GetAround — Délai minimum",
    page_icon=":material/directions_car:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette : une seule couleur d'accent, declinee en deux teintes pour les graphiques
# (ecart perceptible y compris pour les daltoniens), et des gris neutres pour le texte.
# Le theme des widgets (slider, onglets, titres) est dans .streamlit/config.toml.
ACCENT = "#1E3A5F"
ACCENT_LIGHT = "#7A8BA2"
ACCENT_WASH = "rgba(30, 58, 95, 0.08)"
INK, INK_SECONDARY, INK_MUTED = "#1A1A1A", "#4A4A4A", "#6B6B6B"
GRID, AXIS, SURFACE = "#EBEBEB", "#CFCFCF", "#FAFAFA"
TYPE_COLORS = {"mobile": ACCENT, "connect": ACCENT_LIGHT}

# Les deltas des st.metric servent de legende chiffree, pas de variation :
# ni fleche ni couleur verte/rouge.
NEUTRAL_DELTA = dict(delta_color="off", delta_arrow="off")

RECOMMENDED_THRESHOLD = 120
RECOMMENDED_SCOPE = "all"

st.markdown(
    f"""
    <style>
      .block-container {{padding-top: 2.5rem; padding-bottom: 3rem;}}
      h1, h2, h3, h4, h5, h6 {{letter-spacing: -0.01em;}}
      .lead {{color: {INK_SECONDARY}; max-width: 62rem; margin-bottom: 1.5rem;}}
      div[data-testid="stMetric"] {{
        background: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 6px;
        padding: 16px 20px;
      }}
      /* Cartes KPI d'une meme rangee a hauteur egale (colonnes ne contenant qu'un KPI) */
      div[data-testid="stColumn"]:has(> div > div:only-child > div[data-testid="stMetric"]) > div,
      div[data-testid="stElementContainer"]:only-child:has(> div[data-testid="stMetric"]),
      div[data-testid="stElementContainer"]:only-child > div[data-testid="stMetric"] {{height: 100%;}}
      div[data-testid="stMetricLabel"] p {{color: {INK_SECONDARY}; font-size: 0.8125rem;}}
      div[data-testid="stMetricDelta"] {{background: transparent; padding: 0;}}
      div[data-testid="stMetricDelta"] p {{white-space: normal; font-size: 0.8125rem;}}
      .callout {{
        background: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-left: 3px solid {ACCENT};
        border-radius: 0 6px 6px 0;
        padding: 14px 18px;
        margin: 12px 0 24px 0;
        color: {INK};
      }}
      .callout b {{font-weight: 600;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def style_chart(fig: go.Figure) -> go.Figure:
    """Habillage commun des graphiques : fond transparent, grille fine, texte en gris."""
    fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color=INK_SECONDARY, size=12),
        legend_font_color=INK_SECONDARY,
        legend_title_font_color=INK_MUTED,
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#E5E5E5", font_color=INK),
    )
    if fig.layout.title.text:  # sans texte, Plotly afficherait « undefined »
        fig.update_layout(title_font=dict(color=INK, size=14, weight=600))
    fig.update_xaxes(
        showgrid=False, zeroline=False, showline=True, linecolor=AXIS,
        tickfont_color=INK_MUTED, title_font_color=INK_MUTED,
    )
    fig.update_yaxes(
        gridcolor=GRID, zeroline=False, showline=False,
        tickfont_color=INK_MUTED, title_font_color=INK_MUTED,
    )
    return fig


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


@st.cache_data(show_spinner="Chargement des données...")
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
    st.header("Paramètres de la règle")

    threshold = st.slider(
        "Délai minimum entre deux locations",
        min_value=0,
        max_value=720,
        value=RECOMMENDED_THRESHOLD,
        step=30,
        format="%d min",
        help="Une voiture n'est plus réservable si l'écart avec la location "
        "précédente est inférieur à ce seuil.",
    )
    st.caption(f"Soit **{threshold // 60} h {threshold % 60:02d} min**")

    scope = st.radio(
        "Périmètre d'application",
        options=["all", "connect"],
        format_func=lambda s: (
            "Toutes les voitures" if s == "all" else "Voitures Connect uniquement"
        ),
        help="La règle s'applique-t-elle à l'ensemble de la flotte, ou seulement "
        "aux voitures équipées de la technologie Connect ?",
    )

    st.divider()
    st.caption(
        f"**Périmètre de l'analyse**\n\n"
        f"- {N_TOTAL:,} locations au total\n"
        f"- {N_CONSEC:,} avec une location précédente < 12 h\n"
        f"- {N_PROBLEMATIC:,} cas problématiques identifiés"
    )
    st.caption(
        "Un cas est *problématique* quand le conducteur précédent rend la voiture "
        "après l'heure de début prévue de la location suivante."
    )

sim = simulate(threshold, scope)
grid = simulation_grid()

# ─────────────────────────────────────────────────────────────────────────────
# En-tete
# ─────────────────────────────────────────────────────────────────────────────
st.title("GetAround — Délai minimum entre deux locations")
st.markdown(
    '<p class="lead">Outil d\'aide à la décision. Les conducteurs rendent parfois la voiture '
    "en retard, ce qui pénalise le conducteur suivant. Imposer un délai minimum entre deux "
    "locations réduit ces frictions — mais bloque aussi des réservations. "
    "Réglez le seuil et le périmètre dans la barre latérale pour chiffrer l'arbitrage.</p>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# KPI
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Impact de la règle sélectionnée")

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Cas résolus",
    f"{sim['nb_solved']:,}",
    f"{sim['pct_solved']:.1f}% des {N_PROBLEMATIC} cas",
    **NEUTRAL_DELTA,
)
k2.metric(
    "Cas restants",
    f"{sim['nb_remaining']:,}",
    f"{100 - sim['pct_solved']:.1f}% non traités",
    **NEUTRAL_DELTA,
)
k3.metric(
    "Locations bloquées",
    f"{sim['nb_blocked']:,}",
    f"{sim['pct_blocked_total']:.2f}% du volume total",
    **NEUTRAL_DELTA,
)
k4.metric(
    "Rendement",
    "—" if np.isnan(sim["efficiency"]) else f"{sim['efficiency']:.2f}",
    "cas résolu par location bloquée",
    **NEUTRAL_DELTA,
)

# Lecture automatique du reglage courant
if threshold == 0:
    verdict = "**Aucune règle active.** Les 218 cas problématiques subsistent en totalité."
elif scope == "connect":
    verdict = (
        f"Le périmètre **Connect** plafonne à **{grid[grid.scope == 'connect'].pct_solved.max():.0f}% "
        "de cas résolus**, quel que soit le seuil : le flux mobile concentre 80 % du volume "
        "et 1,8x plus de cas problématiques. Élargir à toutes les voitures traite bien plus "
        "de cas à coût comparable."
    )
elif threshold >= 300:
    verdict = (
        f"Seuil élevé : **{sim['pct_solved']:.0f}% des cas résolus**, mais "
        f"**{sim['nb_blocked']:,} locations bloquées** ({sim['pct_blocked_total']:.1f}% du volume). "
        "Au-delà de 2 h, chaque point de bénéfice supplémentaire coûte de plus en plus cher — "
        "la courbe des cas résolus a saturé."
    )
else:
    verdict = (
        f"Réglage équilibré : **{sim['pct_solved']:.0f}% des cas problématiques résolus** "
        f"pour seulement **{sim['pct_blocked_total']:.1f}% des locations bloquées**."
    )

# Le callout est du HTML : on convertit le gras markdown en balises <b>
verdict = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", verdict)
st.markdown(f'<div class="callout">{verdict}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Arbitrage
# ─────────────────────────────────────────────────────────────────────────────
tab_tradeoff, tab_delays, tab_reco, tab_method = st.tabs(
    ["Arbitrage bénéfice / coût", "Analyse des retards", "Recommandation", "Méthodologie"]
)

with tab_tradeoff:
    left, right = st.columns([3, 2])

    with left:
        st.markdown("##### Bénéfice et coût selon le seuil")
        current = grid[grid.scope == scope]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=current.threshold,
                y=current.pct_solved,
                name="Cas problématiques résolus",
                line=dict(color=ACCENT, width=2),
                fill="tozeroy",
                fillcolor=ACCENT_WASH,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=current.threshold,
                y=current.pct_blocked_consec,
                name="Locations consécutives bloquées",
                line=dict(color=ACCENT_LIGHT, width=2, dash="dot"),
            )
        )
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_width=1,
            line_color=INK_MUTED,
            annotation_text=f"{threshold} min",
            annotation_position="top",
            annotation_font_color=INK_SECONDARY,
        )
        fig.update_layout(
            xaxis_title="Seuil (minutes)",
            yaxis_title="%",
            height=420,
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=20, l=10, r=10, b=10),
        )
        st.plotly_chart(style_chart(fig), width="stretch")

    with right:
        st.markdown("##### Frontière coût / bénéfice")
        pts = grid[grid.threshold > 0].copy()
        pts["Périmètre"] = pts.scope.map(
            {"all": "Toutes les voitures", "connect": "Connect uniquement"}
        )

        fig2 = px.scatter(
            pts,
            x="pct_blocked_total",
            y="pct_solved",
            color="Périmètre",
            color_discrete_map={"Toutes les voitures": ACCENT, "Connect uniquement": ACCENT_LIGHT},
            hover_data={"threshold": True},
            labels={
                "pct_blocked_total": "Coût — % de toutes les locations bloquées",
                "pct_solved": "Bénéfice — % de cas résolus",
            },
        )
        fig2.update_traces(marker=dict(size=8, line=dict(width=1, color=SURFACE)))
        fig2.add_trace(
            go.Scatter(
                x=[sim["pct_blocked_total"]],
                y=[sim["pct_solved"]],
                mode="markers",
                marker=dict(size=18, color="rgba(0, 0, 0, 0)", line=dict(width=2, color=INK)),
                name="Réglage actuel",
            )
        )
        fig2.update_layout(
            height=420,
            legend=dict(orientation="h", y=-0.25),
            margin=dict(t=20, l=10, r=10, b=10),
        )
        st.plotly_chart(style_chart(fig2), width="stretch")
        st.caption("Plus un point est **haut et à gauche**, meilleur est le compromis.")

    st.markdown("##### Grille de décision")
    table = (
        grid[(grid.scope == scope) & (grid.threshold.isin([30, 60, 90, 120, 180, 240, 360, 480, 720]))]
        .assign(
            Seuil=lambda d: d.threshold.astype(str) + " min",
            **{
                "Cas résolus": lambda d: d.nb_solved.astype(str)
                + " (" + d.pct_solved.round(1).astype(str) + "%)",
                "Cas restants": lambda d: d.nb_remaining,
                "Locations bloquées": lambda d: d.nb_blocked.astype(str)
                + " (" + d.pct_blocked_total.round(2).astype(str) + "%)",
                "Rendement": lambda d: d.efficiency.round(3),
            },
        )
        .loc[:, ["Seuil", "Cas résolus", "Cas restants", "Locations bloquées", "Rendement"]]
    )
    st.dataframe(table, width="stretch", hide_index=True)

with tab_delays:
    st.markdown("##### D'où vient le problème ?")

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
            opacity=0.9,
            color_discrete_map=TYPE_COLORS,
            labels={
                "delay_at_checkout_in_minutes": "Retard au checkout (min)",
                "checkin_type": "Canal",
            },
            title="Distribution des retards (fenêtre ±12 h)",
        )
        fig3.add_vline(x=0, line_dash="dash", line_width=1, line_color=INK_MUTED)
        fig3.update_layout(height=380, bargap=0.02, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(style_chart(fig3), width="stretch")
        st.caption(
            "95 % des restitutions tiennent dans cette fenêtre. Les extrêmes "
            "(jusqu'à 49 jours) sont conservés dans les calculs mais exclus de ce graphique."
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
            labels={"checkin_type": "Canal", "taux": "% de cas problématiques"},
            title="Taux de cas problématiques par canal",
        )
        fig4.update_traces(texttemplate="%{text}%", textposition="outside", textfont_color=INK_SECONDARY)
        fig4.update_layout(height=380, showlegend=False, bargap=0.55)
        st.plotly_chart(style_chart(fig4), width="stretch")
        st.caption(
            "Le flux **mobile** est 1,8x plus problématique que **Connect** — et représente "
            "80 % du volume. C'est l'argument central en faveur du périmètre `all`."
        )

    st.markdown("##### Le retard du conducteur précédent se paie en annulations")
    cancel = (
        analysed.groupby("is_problematic")["state"]
        .value_counts(normalize=True)
        .unstack()
        .mul(100)
        .round(1)
    )
    cancel.index = ["Location normale", "Conducteur précédent en retard"]

    c3, c4 = st.columns([2, 1])
    with c3:
        fig5 = px.bar(
            cancel.reset_index().melt(id_vars="index", var_name="État", value_name="pct"),
            x="index",
            y="pct",
            color="État",
            barmode="group",
            text="pct",
            color_discrete_map={"canceled": ACCENT, "ended": ACCENT_LIGHT},
            labels={"index": "", "pct": "% des locations"},
        )
        fig5.update_traces(texttemplate="%{text}%", textposition="outside", textfont_color=INK_SECONDARY)
        fig5.update_layout(
            height=340, legend=dict(orientation="h", y=-0.2), bargap=0.35, bargroupgap=0.08
        )
        st.plotly_chart(style_chart(fig5), width="stretch")
    with c4:
        st.metric(
            "Taux d'annulation",
            f"{cancel.loc['Conducteur précédent en retard', 'canceled']:.1f}%",
            f"vs {cancel.loc['Location normale', 'canceled']:.1f}% en temps normal",
            **NEUTRAL_DELTA,
        )
        st.markdown(
            '<div class="callout">Le retard du conducteur précédent fait bondir '
            "les annulations de <b>+52 % en relatif</b>. Le problème ne dégrade pas seulement "
            "l'expérience : il détruit du chiffre d'affaires.</div>",
            unsafe_allow_html=True,
        )

with tab_reco:
    st.markdown("### Recommandation : 120 minutes, périmètre toutes les voitures")

    reco = simulate(RECOMMENDED_THRESHOLD, RECOMMENDED_SCOPE)
    r1, r2, r3, r4 = st.columns(4)
    r1.metric(
        "Cas résolus",
        f"{reco['nb_solved']}",
        f"{reco['pct_solved']:.1f}% du problème",
        **NEUTRAL_DELTA,
    )
    r2.metric("Cas restants", f"{reco['nb_remaining']}")
    r3.metric(
        "Locations bloquées",
        f"{reco['nb_blocked']}",
        f"{reco['pct_blocked_total']:.1f}% du volume",
        **NEUTRAL_DELTA,
    )
    r4.metric("Rendement", f"{reco['efficiency']:.2f}", "cas / location bloquée", **NEUTRAL_DELTA)

    st.markdown(
        """
#### Pourquoi 120 minutes

- **vs 60 min** — +15,6 pts de cas résolus (67 % → 83 %) pour seulement +1,2 pt de locations
  bloquées. C'est le meilleur rapport marginal de toute la grille.
- **vs 180 min** — +7,3 pts de bénéfice seulement, pour +1,0 pt de coût : on entre dans la zone
  de rendement décroissant. 90 min reste une alternative défendable si la priorité est de
  préserver les revenus (79 % de cas résolus pour 2,7 % de locations bloquées).

#### Pourquoi toutes les voitures plutôt que Connect

Le périmètre Connect **plafonne à 32 % de cas résolus**, quel que soit le seuil retenu.
Le flux mobile concentre 80 % du volume et affiche un taux de cas problématiques 1,8x supérieur :
l'exclure du périmètre revient à ignorer l'essentiel du problème.

#### Précautions

Ces chiffres reposent sur 1 729 locations consécutives avec retard précédent mesuré, soit 8 %
du dataset. Le modèle suppose par ailleurs qu'une location bloquée est une location *perdue* —
hypothèse volontairement pessimiste, puisqu'en pratique une partie des conducteurs décalerait
simplement sa réservation. Le coût réel est donc vraisemblablement *inférieur* aux 3,1 % annoncés.

#### Prochaine étape

Un A/B test sur un sous-ensemble de voitures, avec suivi du taux d'annulation et du revenu par
voiture, permettrait de valider ces estimations en conditions réelles avant généralisation.
        """
    )

with tab_method:
    st.markdown(
        f"""
#### Définition d'un cas problématique

Pour une location *r* précédée d'une location *p* sur la même voiture :

```
retard_utile(r) = delay_at_checkout(p) − time_delta(r)
r est problématique  ⟺  retard_utile(r) > 0
```

Le retard qui gêne le conducteur d'une location est celui de la location *précédente*.
Il est récupéré par jointure `previous_ended_rental_id` → `rental_id` : comparer le
`delay_at_checkout_in_minutes` d'une ligne à son propre `time_delta_with_previous_rental_in_minutes`
confronterait deux grandeurs qui ne portent pas sur la même location.

#### Modélisation de la règle

Pour un seuil `T` sur un périmètre `S` :

| Métrique | Définition |
|---|---|
| Locations bloquées | locations de `S` dont `time_delta < T` — elles n'auraient pas pu être réservées |
| Cas résolus | cas problématiques de `S` dont `time_delta < T` — la réservation conflictuelle n'a pas lieu |
| Cas restants | cas problématiques dont `time_delta ≥ T` — la règle ne les empêche pas |

#### Filtrage appliqué

| Étape | Locations |
|---|---|
| Dataset complet | {N_TOTAL:,} |
| Avec une location précédente < 12 h | {N_CONSEC:,} |
| Dont retard précédent mesuré (périmètre d'analyse) | {len(analysed):,} |
| Dont cas problématiques | {N_PROBLEMATIC:,} |

Les {N_CONSEC - len(analysed):,} locations dont le retard précédent n'a pas été enregistré sont
*exclues* plutôt qu'imputées à zéro, ce qui sous-estimerait le problème.

`previous_ended_rental_id` est `NULL` dès que l'écart avec la location précédente dépasse 12 h :
c'est ce qui borne naturellement l'analyse à un seuil maximum de 720 minutes.
        """
    )

    with st.expander("Aperçu du périmètre d'analyse"):
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
st.caption("GetAround Analysis · Jedha Bootcamp Bloc 5 · Données : 21 310 locations")
