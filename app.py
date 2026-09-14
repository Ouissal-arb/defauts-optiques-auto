"""
Interface Streamlit - Prediction de defauts optiques (composants automobiles)
================================================================================
Cette application presente le projet realise pendant le stage : elle NE
reentraine RIEN, elle lit simplement les artefacts produits par le notebook
`pipeline3_finale.ipynb` (dossier `artifacts/`) pour presenter la demarche
et les resultats au maitre de stage.

Lancer avec :  streamlit run app.py
(a executer depuis le dossier qui contient a la fois app.py et artifacts/)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ARTIFACTS_DIR = Path("artifacts")

st.set_page_config(
    page_title="Defauts optiques - Composants automobiles",
    page_icon="🔎",
    layout="wide",
)

# ------------------------------------------------------------------ styling
st.markdown(
    """
    <style>
    html, body, [class*="css"]  { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8E7;
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }
    .accent { color: #1F7A72; }
    h1, h2, h3 { letter-spacing: -0.01em; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ loaders
@st.cache_data
def load_metadata():
    path = ARTIFACTS_DIR / "metadata.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_predictions():
    path = ARTIFACTS_DIR / "predictions.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_infer_X():
    path = ARTIFACTS_DIR / "infer_X.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_resource
def load_models():
    path = ARTIFACTS_DIR / "models.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_resource
def load_preprocessing():
    imputer_path = ARTIFACTS_DIR / "imputer.joblib"
    scaler_path = ARTIFACTS_DIR / "scaler.joblib"
    if not (imputer_path.exists() and scaler_path.exists()):
        return None, None
    return joblib.load(imputer_path), joblib.load(scaler_path)


metadata = load_metadata()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🔎 Defauts optiques")
st.sidebar.caption("Pieces automobiles transparentes")
page = st.sidebar.radio(
    "Navigation",
    [
        "Vue d'ensemble",
        "Exploration des donnees",
        "Benchmark des modeles",
        "Resultats & performance",
        "Predictions",
    ],
)

if metadata is None:
    st.warning(
        "Aucun artefact trouve dans le dossier `artifacts/`.\n\n"
        "Lancez d'abord le notebook **pipeline3_finale.ipynb** en entier : il "
        "entraine les modeles et sauvegarde tout ce dont cette application a "
        "besoin (`artifacts/metadata.json`, `models.joblib`, ...). Placez "
        "ensuite ce dossier `artifacts/` a cote de `app.py`."
    )
    st.stop()

Y_COLS = metadata["y_cols"]
X_COLS = metadata["x_cols"]
results_df = pd.DataFrame(metadata["results_per_target"]).set_index("cible")
global_row = results_df.loc["GLOBAL (moyenne)"]
per_target = results_df.drop(index="GLOBAL (moyenne)")

# =====================================================================
if page == "Vue d'ensemble":
    st.title("Prediction de defauts optiques sur composants automobiles")
    st.markdown(
        """
        **Objectif du projet.** A partir de mesures de deformation physique de
        la surface d'une piece transparente (pare-brise, phare...), predire la
        distorsion optique en **12 points de mesure** (`value_center_k1` a
        `value_center_k12`). Une surface parfaitement conforme a une
        distorsion proche de **0** : plus les predictions du modele se
        rapprochent de 0 pour une piece conforme, mieux le modele capture le
        phenomene physique.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f'<div class="metric-card"><b>Algorithme retenu</b><br>'
        f'<span class="accent" style="font-size:1.4rem">{metadata["winner_algo"]}</span></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="metric-card"><b>R2 global (validation)</b><br>'
        f'<span class="accent" style="font-size:1.4rem">{global_row["R2"]:.3f}</span></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        f'<div class="metric-card"><b>MAE global (validation)</b><br>'
        f'<span class="accent" style="font-size:1.4rem">{global_row["MAE"]:.3f}</span></div>',
        unsafe_allow_html=True,
    )
    c4.markdown(
        f'<div class="metric-card"><b>Pieces predites (test aveugle)</b><br>'
        f'<span class="accent" style="font-size:1.4rem">{metadata["n_infer"]}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Feuille de route suivie")
    st.markdown(
        """
        1. **Exploration et pretraitement (EDA)** — valeurs manquantes et
           aberrantes controlees, variables X standardisees (`StandardScaler`).
        2. **Strategie de validation** — split entrainement (80%) / validation
           (20%).
        3. **Benchmark** — Random Forest vs Gradient Boosting, sur le meme
           split, un modele par point (k1 a k12).
        4. **Optimisation** — recherche d'hyperparametres du modele gagnant
           pour maximiser le R2 moyen.
        5. **Export** — predictions sur le jeu de test aveugle, fichier CSV
           dans l'ordre d'origine des lignes.
        """
    )
    st.caption(
        f"Donnees labellisees utilisees pour l'entrainement : {metadata['n_train_labelled']} lignes "
        f"— {len(X_COLS)} variables de mesure physique en entree."
    )

# =====================================================================
elif page == "Exploration des donnees":
    st.title("Exploration et pretraitement (EDA)")

    st.markdown(
        f"""
        - **Valeurs manquantes (X)** : imputees par la **mediane**, calculee
          uniquement sur les donnees d'entrainement (pas de fuite vers la
          validation ni le jeu aveugle).
        - **Valeurs aberrantes (Y)** : toute mesure avec `|distorsion| > {metadata['outlier_y_max']}`
          est consideree comme une erreur de capteur et exclue de
          l'apprentissage (mise a `NaN`) plutot que traitee comme un vrai defaut.
        - **Standardisation** : toutes les variables X sont centrees-reduites
          (`StandardScaler`) avant l'entrainement, ajustee sur le train
          uniquement.
        """
    )

    infer_X = load_infer_X()
    if infer_X is not None:
        st.markdown("#### Apercu des variables d'entree (jeu de test aveugle)")
        st.dataframe(infer_X.describe().T.round(3), width='stretch')
    else:
        st.info("Fichier `artifacts/infer_X.csv` non trouve — apercu indisponible.")

# =====================================================================
elif page == "Benchmark des modeles":
    st.title("Benchmark : Random Forest vs Gradient Boosting")
    st.markdown(
        "Les deux algorithmes sont entraines et evalues **sur le meme split** "
        "80/20 pour une comparaison equitable, avec un modele independant par "
        "point (k1 a k12)."
    )

    bench = pd.DataFrame(metadata["benchmark"]).T
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(bench.round(4), width='stretch')
        winner = metadata["winner_algo"]
        st.success(f"Modele retenu pour l'optimisation : **{winner}**")

    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(bench.index, bench["R2"], color=["#1F7A72", "#C97B4A"])
        ax.set_ylabel("R2 global (moyenne des 12 cibles, validation)")
        ax.axhline(0, color="black", linewidth=0.8)
        for i, v in enumerate(bench["R2"]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top")
        st.pyplot(fig, width='stretch')

    st.markdown("#### Hyperparametres retenus apres optimisation")
    st.json(metadata["best_params"])

# =====================================================================
elif page == "Resultats & performance":
    st.title("Performance du modele final")

    c1, c2 = st.columns(2)
    c1.metric("R2 global (validation)", f"{global_row['R2']:.3f}")
    c2.metric("MAE global (validation)", f"{global_row['MAE']:.3f}")

    st.markdown(
        "Suivi granulaire par point de mesure, pour identifier les eventuelles "
        "faiblesses locales du modele :"
    )
    st.dataframe(per_target.round(4), width='stretch')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(per_target.index, per_target["R2"], color="#1F7A72")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("R2 par point")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(per_target.index, per_target["MAE"], color="#C9524A")
    axes[1].set_title("MAE par point")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    st.pyplot(fig, width='stretch')

# =====================================================================
elif page == "Predictions":
    st.title("Predictions sur le jeu de test aveugle")

    pred_df = load_predictions()
    infer_X = load_infer_X()

    if pred_df is None:
        st.info("Fichier `artifacts/predictions.csv` non trouve.")
    else:
        st.markdown(
            f"**{len(pred_df)} pieces** predites, dans l'ordre original du "
            "fichier `INDUSTRIAL_AI_Infer.pkl` — pret pour evaluation par le "
            "maitre de stage."
        )
        st.download_button(
            "Telecharger le CSV des predictions",
            data=pred_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions_Y_v3.csv",
            mime="text/csv",
        )
        st.dataframe(pred_df.round(4), width='stretch', height=350)

        st.markdown("#### Explorer une piece")
        idx = st.number_input(
            "Indice de la piece (ligne du jeu de test aveugle)",
            min_value=0,
            max_value=len(pred_df) - 1,
            value=0,
            step=1,
        )
        row = pred_df.iloc[int(idx)]
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.bar(Y_COLS, row.values, color="#1F7A72")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Distorsion optique predite")
        ax.set_title(f"Piece #{int(idx)} — distorsion predite par point (0 = conforme)")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig, width='stretch')

        if infer_X is not None:
            with st.expander("Voir les mesures de deformation en entree pour cette piece"):
                st.dataframe(infer_X.iloc[[int(idx)]].T.rename(columns={int(idx): "valeur"}))
