import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from utils.fairness import (
    demographic_parity_difference,
    disparate_impact_ratio,
    equalized_odds_difference,
)

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stroke Bias Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS minimal et fiable ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Reset Streamlit chrome */
header[data-testid="stHeader"]   { display: none !important; }
[data-testid="stDecoration"]     { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
#MainMenu, footer                { visibility: hidden !important; }

/* White background */
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"],
[data-testid="stMain"], section.main, .block-container {
    background-color: #ffffff !important;
}

/* Layout */
.block-container {
    padding-top: 1.5rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 100% !important;
}

/* === FONT SIZE — multiple Streamlit selectors for robustness === */
/* Markdown text */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stText"],
.stMarkdown p, .stMarkdown li {
    font-size: 1.15rem !important;
    line-height: 1.75 !important;
    color: #111111 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em { color: #111111 !important; font-size: inherit !important; }
[data-testid="stMarkdownContainer"] h1, .stMarkdown h1 { font-size: 2.2rem !important; font-weight: 700 !important; color: #111111 !important; }
[data-testid="stMarkdownContainer"] h2, .stMarkdown h2 { font-size: 1.8rem !important; font-weight: 700 !important; color: #111111 !important; }
[data-testid="stMarkdownContainer"] h3, .stMarkdown h3 { font-size: 1.4rem !important; font-weight: 600 !important; color: #111111 !important; }
/* Direct paragraph injection via st.markdown raw HTML */
.element-container p { font-size: 1.15rem !important; color: #111111 !important; line-height: 1.75 !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #F8F9FA !important;
    border: 1px solid #E9ECEF !important;
    border-radius: 10px !important;
    padding: 20px !important;
}
[data-testid="stMetricValue"] div {
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    color: #111111 !important;
}
[data-testid="stMetricLabel"] div {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #6B7280 !important;
}

/* Selectbox */
[data-baseweb="select"] > div:first-child {
    background: #ffffff !important;
    border: 1.5px solid #D1D5DB !important;
    border-radius: 8px !important;
    min-height: 44px !important;
}
/* Selectbox: all children dark */
[data-baseweb="select"] * { color: #111111 !important; }
/* Dropdown container */
[data-baseweb="popover"] > div,
[data-baseweb="menu"],
ul[role="listbox"] { background: #ffffff !important; border: 1px solid #E9ECEF !important; border-radius: 8px !important; }
/* Each option row */
li[role="option"],
div[role="option"],
[data-baseweb="menu-item"] {
    color: #111111 !important;
    background: #ffffff !important;
    font-size: 1.05rem !important;
}
li[role="option"]:hover,
div[role="option"]:hover,
li[role="option"][aria-selected="true"],
div[role="option"][aria-selected="true"] {
    background: #F3F4F6 !important;
    color: #111111 !important;
}
/* All text inside option */
li[role="option"] *,
div[role="option"] * {
    color: #111111 !important;
    font-size: 1.05rem !important;
}
[data-testid="stSelectbox"] label {
    font-size: 1.05rem !important; font-weight: 600 !important; color: #374151 !important;
}

/* Radio nav */
[data-testid="stRadio"] > div { flex-direction: row !important; gap: 8px !important; flex-wrap: wrap !important; }
[data-testid="stRadio"] label {
    background: #F3F4F6 !important;
    border: 1.5px solid #E5E7EB !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    cursor: pointer !important;
    white-space: nowrap !important;
}
[data-testid="stRadio"] label p,
[data-testid="stRadio"] label span { font-size: 1.05rem !important; font-weight: 500 !important; color: #374151 !important; }
[data-testid="stRadio"] label:has(input:checked) { background: #1e293b !important; border-color: #1e293b !important; }
[data-testid="stRadio"] label:has(input:checked) p,
[data-testid="stRadio"] label:has(input:checked) span { color: #ffffff !important; }
[data-testid="stRadio"] input[type="radio"] { display: none !important; }
[data-testid="stRadio"] label p { display: inline !important; margin: 0 !important; }

/* Alerts */
[data-testid="stAlert"] { border-radius: 8px !important; }
[data-testid="stAlert"] p { font-size: 1.05rem !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #E9ECEF !important; border-radius: 10px !important; }

/* Divider */
hr { border: none !important; border-top: 1px solid #E9ECEF !important; margin: 1.5rem 0 !important; }

/* Section titles */
.sec-title { font-size: 1.6rem !important; font-weight: 700 !important; color: #111111 !important; margin: 8px 0 4px !important; font-family: 'Inter', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

# ── Données ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    except FileNotFoundError:
        st.error("❌ Fichier introuvable : placez 'healthcare-dataset-stroke-data.csv' dans le même dossier que app.py")
        st.stop()
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    df = df[df["gender"] != "Other"].reset_index(drop=True)
    return df

df = load_data()

# ── Plotly theme ──────────────────────────────────────────────────────────────
COLORS = ["#1e293b", "#e11d48", "#f59e0b", "#10b981", "#6366f1", "#0ea5e9"]

def clean_chart(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#111111")),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter", size=12, color="#111111"),
        margin=dict(t=50, b=30, l=20, r=20),
        xaxis=dict(showgrid=False, linecolor="#E5E7EB", tickcolor="#E5E7EB",
                   tickfont=dict(color="#374151")),
        yaxis=dict(gridcolor="#F3F4F6", linecolor="#E5E7EB",
                   tickfont=dict(color="#374151")),
        legend=dict(font=dict(color="#374151")),
    )
    return fig

# ── Navigation ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
    <span style="font-size:1.3rem; font-weight:700; color:#111111; white-space:nowrap;">🧠 Stroke Bias Analysis</span>
</div>
""", unsafe_allow_html=True)

pages = ["🏠 Accueil", "📊 Exploration", "⚠️ Détection de Biais", "🤖 Modélisation", "🎯 Prédiction"]
page = st.radio("Navigation", pages, horizontal=True, label_visibility="collapsed")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Accueil":

    st.markdown("## 🧠 Prédiction du Risque d'AVC & Détection de Biais")
    st.markdown("**Dataset :** Stroke Prediction — Kaggle | **Parcours A — Détection de Biais**")

    st.markdown("---")

    # 4 KPIs obligatoires
    st.markdown('<p class="sec-title">Métriques Clés</p>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Nombre de patients", f"{len(df):,}")
    k2.metric("Nombre de colonnes", f"{df.shape[1]}")
    k3.metric("Taux de valeurs manquantes", "3.9%", help="201 valeurs manquantes dans BMI, imputées par la médiane")
    k4.metric("Taux d'AVC (variable cible)", "4.87%", help="Dataset fortement déséquilibré")

    st.markdown("---")

    # Contexte
    st.markdown('<p class="sec-title">Contexte & Problématique</p>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
        L'accident vasculaire cérébral (AVC) est l'une des principales causes de mortalité et de handicap dans le monde.
        Des modèles de machine learning sont utilisés pour **prédire le risque d'AVC** à partir de données cliniques.

        Cependant, ces modèles peuvent reproduire des **biais présents dans les données**. Si un modèle prédit
        différemment selon le **genre** ou la **zone géographique**, des groupes entiers de patients pourraient être
        sous-diagnostiqués — avec des conséquences directes sur leur santé.

        Cette application analyse le dataset Stroke Prediction pour **détecter et quantifier** ces biais.
        """)

    with c2:
        st.info("**🎯 Biais analysés**\n\n**Genre** : Homme vs Femme\n\n**Zone** : Rural vs Urbain")

    st.markdown("---")

    # Aperçu des données
    st.markdown('<p class="sec-title">Aperçu des Données</p>', unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Description des colonnes
    st.markdown('<p class="sec-title">Description des Colonnes</p>', unsafe_allow_html=True)
    desc = pd.DataFrame({
        "Variable": ["age", "gender ⚠️", "hypertension", "heart_disease",
                     "avg_glucose_level", "bmi", "smoking_status",
                     "Residence_type ⚠️", "stroke 🎯"],
        "Description": [
            "Âge du patient (en années)",
            "Genre — Male / Female (attribut sensible)",
            "Hypertension artérielle (0 = Non, 1 = Oui)",
            "Maladie cardiaque (0 = Non, 1 = Oui)",
            "Taux de glucose moyen dans le sang",
            "Indice de masse corporelle (201 valeurs manquantes imputées)",
            "Statut tabagique (formerly smoked / never smoked / smokes / Unknown)",
            "Zone géographique — Rural / Urban (attribut sensible)",
            "AVC survenu (0 = Non, 1 = Oui) — Variable cible",
        ],
        "Type": [
            "Numérique", "Catégoriel", "Binaire", "Binaire",
            "Numérique", "Numérique", "Catégoriel", "Catégoriel", "Binaire"
        ],
    })
    st.dataframe(desc, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Exploration":

    st.markdown("## 📊 Exploration des Données")
    st.markdown("Visualisations clés pour comprendre la distribution du dataset.")

    st.markdown("---")

    # 4 KPIs
    st.markdown('<p class="sec-title">Métriques Clés</p>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total patients", f"{len(df):,}")
    k2.metric("Âge moyen", f"{df['age'].mean():.0f} ans")
    k3.metric("Glucose moyen", f"{df['avg_glucose_level'].mean():.0f} mg/dL")
    k4.metric("BMI médian", f"{df['bmi'].median():.1f}")

    st.markdown("---")

    # Viz 1 — Distribution variable cible
    st.markdown('<p class="sec-title">Visualisation 1 — Distribution de la Variable Cible</p>', unsafe_allow_html=True)

    v1a, v1b = st.columns(2)
    with v1a:
        counts = df["stroke"].value_counts().reset_index()
        counts.columns = ["stroke", "count"]
        counts["label"] = counts["stroke"].map({0: "Pas d'AVC", 1: "AVC"})
        fig = px.bar(counts, x="label", y="count", color="label",
                     color_discrete_sequence=["#1e293b", "#e11d48"],
                     text="count")
        fig.update_traces(textposition="outside", marker_line_width=0, showlegend=False)
        clean_chart(fig, "Nombre de cas par statut AVC")
        st.plotly_chart(fig, use_container_width=True)

    with v1b:
        fig2 = px.pie(counts, values="count", names="label",
                      color_discrete_sequence=["#1e293b", "#e11d48"],
                      hole=0.5)
        fig2.update_traces(textinfo="percent+label", textfont_size=13)
        clean_chart(fig2, "Proportion des cas")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Viz 2 — Comparaison par attribut sensible
    st.markdown('<p class="sec-title">Visualisation 2 — Comparaison par Attribut Sensible</p>', unsafe_allow_html=True)

    v2a, v2b = st.columns(2)
    with v2a:
        g = df.groupby(["gender", "stroke"]).size().reset_index(name="count")
        g["stroke"] = g["stroke"].map({0: "Pas d'AVC", 1: "AVC"})
        fig3 = px.bar(g, x="gender", y="count", color="stroke", barmode="group",
                      color_discrete_sequence=["#1e293b", "#e11d48"],
                      labels={"gender": "Genre", "count": "Patients", "stroke": ""})
        fig3.update_traces(marker_line_width=0)
        clean_chart(fig3, "AVC par Genre")
        st.plotly_chart(fig3, use_container_width=True)

    with v2b:
        r = df.groupby(["Residence_type", "stroke"]).size().reset_index(name="count")
        r["stroke"] = r["stroke"].map({0: "Pas d'AVC", 1: "AVC"})
        fig4 = px.bar(r, x="Residence_type", y="count", color="stroke", barmode="group",
                      color_discrete_sequence=["#1e293b", "#e11d48"],
                      labels={"Residence_type": "Zone", "count": "Patients", "stroke": ""})
        fig4.update_traces(marker_line_width=0)
        clean_chart(fig4, "AVC par Zone Géographique")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # Viz 3 — Heatmap corrélations
    st.markdown('<p class="sec-title">Visualisation 3 — Heatmap des Corrélations</p>', unsafe_allow_html=True)

    num_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "stroke"]
    corr = df[num_cols].corr().round(2)
    fig5 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                     zmin=-1, zmax=1, aspect="auto")
    fig5.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter", size=12, color="#111111"),
        margin=dict(t=50, b=30, l=20, r=20),
        title=dict(text="Matrice de corrélation entre variables numériques",
                   font=dict(size=14, color="#111111"))
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="sec-title">Données Brutes (interactif)</p>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DÉTECTION DE BIAIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Détection de Biais":

    st.markdown("## ⚠️ Détection de Biais")
    st.markdown("Analyse des biais de représentation dans les données.")

    st.markdown("---")

    sensitive_choice = st.selectbox(
        "Choisir l'attribut sensible à analyser :",
        ["Genre (gender)", "Zone géographique (Residence_type)"]
    )

    col = "gender" if "Genre" in sensitive_choice else "Residence_type"
    unpriv = "Female" if col == "gender" else "Rural"
    priv   = "Male"   if col == "gender" else "Urban"

    y = df["stroke"].values
    sensitive = df[col].values

    st.markdown("---")

    # 1. Explication
    st.markdown('<p class="sec-title">1. Explication du Biais Analysé</p>', unsafe_allow_html=True)

    if col == "gender":
        st.markdown("**Attribut sensible : Genre (Male / Female)**")
        st.markdown("""
        Le genre est un attribut protégé corrélé à de nombreux facteurs de santé.
        Dans ce dataset, les **femmes représentent 58.6%** des patients (2 994 vs 2 115 hommes),
        créant un déséquilibre de représentation.

        **Pourquoi c'est problématique ?** Si le modèle apprend principalement sur des données
        féminines, il risque d'être moins performant pour les hommes — et inversement.
        Des patients pourraient être sous-diagnostiqués uniquement en raison de leur genre,
        ce qui retarderait leur prise en charge médicale.
        """)
    else:
        st.markdown("**Attribut sensible : Zone Géographique (Rural / Urban)**")
        st.markdown("""
        La zone de résidence est corrélée à l'accès aux soins et aux conditions de vie.
        Le dataset est quasi-équilibré : **2 596 urbains** vs **2 513 ruraux**.

        **Pourquoi c'est problématique ?** Les données médicales sont souvent collectées dans des
        hôpitaux urbains, potentiellement sous-représentant les pathologies rurales.
        Un modèle biaisé pourrait prédire moins bien pour les patients ruraux,
        aggravant les inégalités de santé existantes.
        """)

    st.markdown("---")

    # 2. Métriques de fairness
    st.markdown('<p class="sec-title">2. Métriques de Fairness</p>', unsafe_allow_html=True)

    res_dp = demographic_parity_difference(y_true=y, y_pred=y, sensitive_attribute=sensitive)
    res_di = disparate_impact_ratio(
        y_true=y, y_pred=y, sensitive_attribute=sensitive,
        unprivileged_value=unpriv, privileged_value=priv
    )

    # Métrique 1
    st.markdown("#### Métrique 1 — Parité Démographique")
    st.markdown(f"Mesure la différence de taux d'AVC entre les groupes. Idéalement proche de **0**.")

    m_cols = st.columns(len(res_dp["group_rates"]) + 1)
    m_cols[0].metric("Différence de Parité", f"{res_dp['difference']:.4f}")
    for i, (grp, rate) in enumerate(res_dp["group_rates"].items()):
        m_cols[i+1].metric(f"Taux AVC — {grp}", f"{rate*100:.2f}%")

    diff = res_dp["difference"]
    if diff < 0.05:
        st.success(f"✅ Différence de parité faible ({diff:.4f} < 0.05) — pas de biais démographique significatif.")
    elif diff < 0.10:
        st.warning(f"⚠️ Différence de parité modérée ({diff:.4f}) — biais à surveiller.")
    else:
        st.error(f"🚨 Différence de parité élevée ({diff:.4f} > 0.10) — biais significatif détecté.")

    st.markdown("#### Métrique 2 — Impact Disproportionné (Disparate Impact)")
    st.markdown(f"Ratio : P(AVC | {unpriv}) / P(AVC | {priv}). **Règle des 4/5 : ratio ≥ 0.8** pour absence de biais.")

    d_cols = st.columns(3)
    d_cols[0].metric("Ratio DI", f"{res_di['ratio']:.4f}")
    d_cols[1].metric(f"Taux — {unpriv}", f"{res_di['rate_unprivileged']*100:.2f}%")
    d_cols[2].metric(f"Taux — {priv}", f"{res_di['rate_privileged']*100:.2f}%")

    di = res_di["ratio"]
    if di >= 0.8:
        st.success(f"✅ Ratio DI = {di:.4f} ≥ 0.8 — règle des 4/5 respectée. Pas d'impact disproportionné.")
    else:
        st.error(f"🚨 Ratio DI = {di:.4f} < 0.8 — règle des 4/5 violée. Biais disproportionné significatif.")

    st.markdown("---")

    # 3. Visualisation
    st.markdown('<p class="sec-title">3. Visualisation des Résultats</p>', unsafe_allow_html=True)

    rates_df = pd.DataFrame({
        "Groupe": list(res_dp["group_rates"].keys()),
        "Taux AVC (%)": [v * 100 for v in res_dp["group_rates"].values()],
    })

    vc1, vc2 = st.columns(2)
    with vc1:
        fig_b = px.bar(rates_df, x="Groupe", y="Taux AVC (%)", color="Groupe",
                       color_discrete_sequence=["#1e293b", "#e11d48"],
                       text_auto=".2f")
        fig_b.add_hline(y=y.mean()*100, line_dash="dash", line_color="#f59e0b",
                        annotation_text=f"Moyenne globale : {y.mean()*100:.2f}%",
                        annotation_position="top right")
        fig_b.update_traces(marker_line_width=0, textposition="outside", showlegend=False)
        clean_chart(fig_b, f"Taux d'AVC par {col}")
        st.plotly_chart(fig_b, use_container_width=True)

    with vc2:
        pop_df = df[col].value_counts().reset_index()
        pop_df.columns = ["Groupe", "Nombre"]
        fig_p = px.pie(pop_df, values="Nombre", names="Groupe",
                       color_discrete_sequence=["#1e293b", "#e11d48"], hole=0.5)
        fig_p.update_traces(textinfo="percent+label", textfont_size=13)
        clean_chart(fig_p, f"Répartition de la population par {col}")
        st.plotly_chart(fig_p, use_container_width=True)

    st.markdown("---")

    # 4. Interprétation
    st.markdown('<p class="sec-title">4. Interprétation</p>', unsafe_allow_html=True)

    rates = res_dp["group_rates"]
    max_grp = max(rates, key=rates.get)
    min_grp = min(rates, key=rates.get)

    st.markdown(f"""
**Que signifie concrètement le biais détecté ?**
La différence de parité est de **{diff:.4f}** entre les groupes. Le groupe **{max_grp}** a un taux
d'AVC de {rates[max_grp]*100:.2f}% contre {rates[min_grp]*100:.2f}% pour **{min_grp}**.

**Quel groupe est défavorisé ?**
Le groupe **{unpriv}** (non-privilégié) présente un ratio DI de **{di:.4f}** par rapport à {priv}.
{"La règle des 4/5 est respectée : pas de biais disproportionné détecté dans les données brutes." if di >= 0.8 else "La règle des 4/5 est violée : biais disproportionné significatif détecté."}

**Quel serait l'impact réel de ce biais ?**
Si un modèle reproduit ce biais, le groupe **{min_grp}** pourrait être sous-estimé dans son risque
réel d'AVC — retardant les interventions préventives et aggravant les inégalités de santé.

**Recommandations pour réduire ce biais :**
- Rééquilibrer les groupes dans les données d'entraînement (SMOTE, undersampling)
- Appliquer des contraintes de fairness lors de l'optimisation du modèle
- Ajuster les seuils de décision par groupe (post-processing)
- Collecter davantage de données pour les groupes sous-représentés
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODÉLISATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Modélisation":

    st.markdown("## 🤖 Modélisation")
    st.markdown("Entraînement d'un modèle prédictif et évaluation de son équité par groupe sensible.")

    st.markdown("---")

    pc1, pc2 = st.columns(2)
    model_choice = pc1.selectbox("Algorithme :", ["Logistic Regression", "Random Forest"])
    sensitive_col = pc2.selectbox("Attribut sensible à évaluer :", ["gender", "Residence_type"])

    @st.cache_data
    def train_model(model_name):
        df_m = df.copy()
        le = LabelEncoder()
        for c in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]:
            df_m[c] = le.fit_transform(df_m[c].astype(str))
        df_m.drop(columns=["id"], errors="ignore", inplace=True)
        X = df_m.drop(columns=["stroke"])
        y_m = df_m["stroke"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_m, test_size=0.2, random_state=42, stratify=y_m
        )
        if model_name == "Logistic Regression":
            m = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        else:
            m = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        m.fit(X_tr, y_tr)
        return m, X, X_te, y_te, m.predict(X_te)

    with st.spinner(f"Entraînement du modèle {model_choice}..."):
        model, X, X_test, y_test, y_pred = train_model(model_choice)

    st.success(f"✅ Modèle **{model_choice}** entraîné avec `class_weight='balanced'` (dataset déséquilibré à 4.87%)")

    st.markdown("---")

    # Performances globales
    st.markdown('<p class="sec-title">Performances Globales</p>', unsafe_allow_html=True)

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Accuracy",  f"{accuracy_score(y_test, y_pred):.3f}")
    g2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.3f}")
    g3.metric("Recall",    f"{recall_score(y_test, y_pred, zero_division=0):.3f}")
    g4.metric("F1-Score",  f"{f1_score(y_test, y_pred, zero_division=0):.3f}")

    st.markdown("---")

    # Fairness sur prédictions
    st.markdown('<p class="sec-title">Métriques de Fairness sur les Prédictions</p>', unsafe_allow_html=True)

    sensitive_test = df.iloc[X_test.index][sensitive_col].values
    unpriv = "Female" if sensitive_col == "gender" else "Rural"
    priv   = "Male"   if sensitive_col == "gender" else "Urban"

    res_dp2 = demographic_parity_difference(y_test.values, y_pred, sensitive_test)
    res_di2 = disparate_impact_ratio(y_test.values, y_pred, sensitive_test, unpriv, priv)
    res_eo2 = equalized_odds_difference(y_test.values, y_pred, sensitive_test)

    f1, f2, f3 = st.columns(3)
    f1.metric("Différence de Parité", f"{res_dp2['difference']:.4f}", help="Idéalement proche de 0")
    f2.metric("Ratio DI", f"{res_di2['ratio']:.4f}", help="Règle des 4/5 : ≥ 0.8")
    f3.metric("Égalité des Chances (EOD)", f"{res_eo2['difference']:.4f}", help="Idéalement proche de 0")

    st.markdown("---")

    # Performances par groupe
    st.markdown('<p class="sec-title">Performances par Groupe Sensible</p>', unsafe_allow_html=True)

    groups = sorted(df.iloc[X_test.index][sensitive_col].unique())
    rows = []
    for grp in groups:
        mask = sensitive_test == grp
        yt, yp_g = y_test.values[mask], y_pred[mask]
        rows.append({
            "Groupe": grp,
            "N patients": int(mask.sum()),
            "Accuracy":  round(accuracy_score(yt, yp_g), 3),
            "Precision": round(precision_score(yt, yp_g, zero_division=0), 3),
            "Recall":    round(recall_score(yt, yp_g, zero_division=0), 3),
            "F1-Score":  round(f1_score(yt, yp_g, zero_division=0), 3),
        })
    perf_df = pd.DataFrame(rows)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    # Bar chart comparaison
    melt = perf_df.melt(id_vars="Groupe",
                         value_vars=["Accuracy", "Precision", "Recall", "F1-Score"])
    fig_perf = px.bar(melt, x="variable", y="value", color="Groupe",
                      barmode="group",
                      color_discrete_sequence=["#1e293b", "#e11d48"],
                      labels={"variable": "Métrique", "value": "Score", "Groupe": "Groupe"})
    fig_perf.update_traces(marker_line_width=0)
    clean_chart(fig_perf, "Comparaison des performances par groupe")
    st.plotly_chart(fig_perf, use_container_width=True)

    st.markdown("---")

    # Matrices de confusion par groupe
    st.markdown('<p class="sec-title">Matrices de Confusion par Groupe</p>', unsafe_allow_html=True)

    cm_cols = st.columns(len(groups))
    for i, grp in enumerate(groups):
        mask = sensitive_test == grp
        yt, yp_g = y_test.values[mask], y_pred[mask]
        cm = confusion_matrix(yt, yp_g, labels=[0, 1])
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Prédit", y="Réel"),
            x=["Pas d'AVC", "AVC"],
            y=["Pas d'AVC", "AVC"],
            color_continuous_scale=[[0, "#EFF6FF"], [1, "#1e293b"]],
        )
        fig_cm.update_layout(
            title=dict(text=f"Groupe : {grp} (N={mask.sum()})",
                       font=dict(size=13, color="#111111")),
            paper_bgcolor="white",
            font=dict(family="Inter", size=12, color="#111111"),
            margin=dict(t=50, b=20, l=20, r=20),
            coloraxis_showscale=False,
            xaxis=dict(tickfont=dict(color="#111111")),
            yaxis=dict(tickfont=dict(color="#111111")),
        )
        cm_cols[i].plotly_chart(fig_cm, use_container_width=True)

    # Feature importance (RF seulement)
    if model_choice == "Random Forest":
        st.markdown("---")
        st.markdown('<p class="sec-title">Importance des Variables (Random Forest)</p>', unsafe_allow_html=True)
        imp = pd.DataFrame({
            "Variable": X.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)
        fig_fi = px.bar(imp, x="Importance", y="Variable", orientation="h",
                        color="Importance",
                        color_continuous_scale=[[0, "#EFF6FF"], [1, "#1e293b"]])
        fig_fi.update_layout(
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Inter", size=12, color="#111111"),
            margin=dict(t=40, b=20, l=20, r=20),
            title=dict(text="Variables les plus prédictives",
                       font=dict(size=14, color="#111111")),
            coloraxis_showscale=False,
            yaxis=dict(tickfont=dict(color="#111111")),
            xaxis=dict(tickfont=dict(color="#111111"), gridcolor="#F3F4F6"),
        )
        st.plotly_chart(fig_fi, use_container_width=True)


# PAGE 5 — PRÉDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Prédiction":

    st.markdown("## 🎯 Prédiction du Risque d'AVC")
    st.markdown("Renseignez les informations du patient pour estimer son risque d'AVC.")
    st.markdown("---")

    @st.cache_resource
    def get_prediction_model():
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        import numpy as np

        df_m = df.copy()
        le_map = {}
        for c in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]:
            le = LabelEncoder()
            df_m[c] = le.fit_transform(df_m[c].astype(str))
            le_map[c] = le
        df_m.drop(columns=["id"], errors="ignore", inplace=True)
        X = df_m.drop(columns=["stroke"])
        y = df_m["stroke"]
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
        )
        model.fit(X_tr, y_tr)

        # Compute reference stats on full population
        all_probas = model.predict_proba(X)[:, 1]
        base_rate = float(y.mean())
        p95 = float(np.percentile(all_probas, 95))

        return model, le_map, list(X.columns), base_rate, p95

    with st.spinner("Chargement du modèle..."):
        model_pred, le_map, feature_cols, base_rate, p95 = get_prediction_model()

    st.markdown('<p class="sec-title">Informations du Patient</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        age       = st.slider("Âge", min_value=1, max_value=82, value=50)
        gender    = st.selectbox("Genre", ["Male", "Female"])
        hyp       = st.selectbox("Hypertension", ["Non", "Oui"])
        heart     = st.selectbox("Maladie cardiaque", ["Non", "Oui"])
    with col2:
        married   = st.selectbox("Déjà marié(e)", ["Yes", "No"])
        work      = st.selectbox("Type de travail", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence = st.selectbox("Zone géographique", ["Urban", "Rural"])
    with col3:
        glucose   = st.slider("Taux de glucose moyen (mg/dL)", 55, 272, 100)
        bmi_val   = st.slider("IMC (BMI)", 10.0, 98.0, 28.0, step=0.5)
        smoking   = st.selectbox("Statut tabagique", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    st.markdown("---")

    if st.button("🔍 Prédire le Risque d'AVC", use_container_width=True):

        input_dict = {
            "gender":            le_map["gender"].transform([gender])[0],
            "age":               age,
            "hypertension":      1 if hyp == "Oui" else 0,
            "heart_disease":     1 if heart == "Oui" else 0,
            "ever_married":      le_map["ever_married"].transform([married])[0],
            "work_type":         le_map["work_type"].transform([work])[0],
            "Residence_type":    le_map["Residence_type"].transform([residence])[0],
            "avg_glucose_level": glucose,
            "bmi":               bmi_val,
            "smoking_status":    le_map["smoking_status"].transform([smoking])[0],
        }

        input_df   = pd.DataFrame([input_dict])[feature_cols]
        proba      = float(model_pred.predict_proba(input_df)[0][1])
        rel_risk   = proba / base_rate  # times more likely than average
        risk_score = min(100, (proba / p95) * 100)  # 0-100 relative to population p95

        # Risk level
        if risk_score < 25:
            level, color, emoji = "Faible", "#22c55e", "✅"
        elif risk_score < 55:
            level, color, emoji = "Modéré", "#f59e0b", "⚠️"
        else:
            level, color, emoji = "Élevé", "#e11d48", "🚨"

        st.markdown("---")
        st.markdown('<p class="sec-title">Résultat de la Prédiction</p>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        r1.metric("Score de risque", f"{risk_score:.0f} / 100",
                  help="Score relatif à la population (0 = risque minimal, 100 = risque maximal)")
        r2.metric("Risque relatif", f"{rel_risk:.1f}×",
                  help=f"Par rapport à la moyenne de la population ({base_rate*100:.1f}%)")
        r3.metric("Niveau de risque", f"{emoji} {level}")

        # Gauge on risk_score (0-100) — much more meaningful
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={"suffix": "/100", "font": {"size": 40, "color": "#111111"}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"color": "#555"},
                         "tickvals": [0, 25, 55, 80, 100],
                         "ticktext": ["0", "25", "55", "80", "100"]},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "white",
                "steps": [
                    {"range": [0,  25], "color": "#dcfce7"},
                    {"range": [25, 55], "color": "#fef9c3"},
                    {"range": [55, 80], "color": "#fed7aa"},
                    {"range": [80, 100],"color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#374151", "width": 2},
                    "thickness": 0.75,
                    "value": risk_score,
                },
            },
            title={"text": f"Score de Risque Relatif<br><span style='font-size:0.85em;color:#888'>Probabilité brute : {proba*100:.1f}% | Moyenne pop. : {base_rate*100:.1f}%</span>",
                   "font": {"size": 15, "color": "#111111"}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="white",
            font=dict(family="Inter", color="#111111"),
            margin=dict(t=80, b=10, l=40, r=40),
            height=340,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Interpretation
        if level == "Faible":
            st.success(f"✅ **Risque faible (score {risk_score:.0f}/100 — {rel_risk:.1f}× la moyenne).**\nLe profil du patient ne présente pas de facteurs de risque majeurs. Un suivi médical régulier est conseillé.")
        elif level == "Modéré":
            st.warning(f"⚠️ **Risque modéré (score {risk_score:.0f}/100 — {rel_risk:.1f}× la moyenne).**\nCertains facteurs de risque sont présents. Une consultation médicale et des mesures préventives sont recommandées.")
        else:
            st.error(f"🚨 **Risque élevé (score {risk_score:.0f}/100 — {rel_risk:.1f}× la moyenne).**\nLe profil présente plusieurs facteurs de risque cumulés. Une consultation médicale est fortement conseillée.")

        st.markdown("---")

        # Factors table
        st.markdown('<p class="sec-title">Facteurs de Risque du Patient vs Population</p>', unsafe_allow_html=True)

        rows = [
            {"Facteur": "Âge",             "Patient": f"{age} ans",       "Moyenne population": f"{df['age'].mean():.0f} ans",            "Écart": f"{'↑' if age > df['age'].mean() else '↓'} {abs(age - df['age'].mean()):.0f} ans"},
            {"Facteur": "Glucose moyen",   "Patient": f"{glucose} mg/dL", "Moyenne population": f"{df['avg_glucose_level'].mean():.0f} mg/dL", "Écart": f"{'↑' if glucose > df['avg_glucose_level'].mean() else '↓'} {abs(glucose - df['avg_glucose_level'].mean()):.0f}"},
            {"Facteur": "IMC",             "Patient": f"{bmi_val:.1f}",   "Moyenne population": f"{df['bmi'].mean():.1f}",               "Écart": f"{'↑' if bmi_val > df['bmi'].mean() else '↓'} {abs(bmi_val - df['bmi'].mean()):.1f}"},
            {"Facteur": "Hypertension",    "Patient": hyp,                "Moyenne population": f"{df['hypertension'].mean()*100:.1f}% oui",  "Écart": "⚠️ Facteur de risque" if hyp == "Oui" else "✅ Pas de risque"},
            {"Facteur": "Maladie cardiaque","Patient": heart,             "Moyenne population": f"{df['heart_disease'].mean()*100:.1f}% oui",  "Écart": "⚠️ Facteur de risque" if heart == "Oui" else "✅ Pas de risque"},
            {"Facteur": "Tabac",           "Patient": smoking,            "Moyenne population": "—",                                     "Écart": "⚠️ Facteur de risque" if smoking == "smokes" else "✅ OK"},
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.caption("⚠️ Cet outil est à des fins éducatives uniquement. Il ne remplace pas un diagnostic médical professionnel.")
