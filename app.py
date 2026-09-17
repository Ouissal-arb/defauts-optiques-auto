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
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        # Le plus souvent : fichier suivi par Git LFS laisse sous forme de
        # pointeur texte (pas telecharge) plutot que le vrai binaire.
        return None


@st.cache_resource
def load_preprocessing():
    imputer_path = ARTIFACTS_DIR / "imputer.joblib"
    scaler_path = ARTIFACTS_DIR / "scaler.joblib"
    if not (imputer_path.exists() and scaler_path.exists()):
        return None, None
    try:
        return joblib.load(imputer_path), joblib.load(scaler_path)
    except Exception:
        return None, None


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
        "Tester une piece",
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

# =====================================================================
elif page == "Tester une piece":
    st.title("🧪 Diagnostic d'une piece — saisie manuelle")
    st.markdown(
        """
        Saisissez les **29 mesures de deformation physique** d'une piece (ou
        chargez un exemple existant), puis lancez la prediction : le modele
        estime la distorsion optique aux 12 points de mesure et l'appli rend
        un verdict **Conforme / Defaut** en comparant chaque point a une
        tolerance de conformite.
        """
    )

    models = load_models()
    imputer, scaler = load_preprocessing()
    infer_X = load_infer_X()

    if models is None or imputer is None or scaler is None:
        st.error(
            "Artefacts de modele introuvables ou illisibles "
            "(`models.joblib`, `imputer.joblib`, `scaler.joblib`).\n\n"
            "Si ces fichiers sont suivis par **Git LFS**, verifiez qu'ils ont "
            "bien ete telecharges (`git lfs pull`) et pas laisses sous forme "
            "de simple pointeur texte."
        )
        st.stop()

    default_tolerance = float(metadata.get("tolerance", 2.0))

    st.markdown("#### 1. Pre-remplir les mesures")
    preset = st.radio(
        "Point de depart",
        ["Valeurs medianes (reference)", "Exemple du jeu de test aveugle"],
        horizontal=True,
    )

    if preset == "Exemple du jeu de test aveugle" and infer_X is not None:
        col_a, col_b = st.columns([3, 1])
        with col_b:
            if st.button("🎲 Tirer un exemple au hasard"):
                st.session_state["piece_idx"] = int(np.random.randint(len(infer_X)))
        with col_a:
            idx = st.number_input(
                "Indice de la piece",
                min_value=0,
                max_value=len(infer_X) - 1,
                value=int(st.session_state.get("piece_idx", 0)),
                step=1,
                key="piece_idx",
            )
        preset_values = infer_X.iloc[int(idx)].to_dict()
        st.caption(f"Mesures pre-remplies depuis la piece #{int(idx)} du jeu de test aveugle.")
    elif infer_X is not None:
        preset_values = infer_X.median().to_dict()
    else:
        preset_values = {c: 0.0 for c in X_COLS}

    st.markdown("#### 2. Verifier / ajuster les mesures et predire")
    groups = {
        "Capteurs de surface (s)": [c for c in X_COLS if c.startswith("s")],
        "Mesures v": [c for c in X_COLS if c.startswith("v")],
        "Mesures oh": [c for c in X_COLS if c.startswith("oh")],
    }

    input_values = {}
    with st.form("piece_form"):
        for group_name, cols in groups.items():
            st.markdown(f"**{group_name}**")
            n_per_row = 4
            for i in range(0, len(cols), n_per_row):
                row_cols = cols[i : i + n_per_row]
                widget_cols = st.columns(len(row_cols))
                for wc, feat in zip(widget_cols, row_cols):
                    default_val = float(preset_values.get(feat, 0.0))
                    input_values[feat] = wc.number_input(
                        feat,
                        value=round(default_val, 4),
                        format="%.4f",
                        key=f"feat_{feat}_{preset}",
                    )

        tolerance = st.slider(
            "Tolerance de conformite (|distorsion| max acceptee par point)",
            min_value=0.5,
            max_value=25.0,
            value=round(default_tolerance, 1),
            step=0.5,
            help="Valeur par defaut = mediane de |distorsion| observee sur les "
            "donnees d'entrainement (voir metadata.json). A remplacer par la "
            "vraie tolerance metier du cahier des charges des qu'elle est connue.",
        )

        submitted = st.form_submit_button("🔍 Predire la conformite")

    if submitted:
        x_row = pd.DataFrame([input_values])[X_COLS]
        x_imputed = imputer.transform(x_row)
        x_scaled = scaler.transform(x_imputed)

        preds = {col: float(models[col].predict(x_scaled)[0]) for col in Y_COLS}
        pred_series = pd.Series(preds)[Y_COLS]
        conforme_mask = pred_series.abs() <= tolerance
        n_defauts = int((~conforme_mask).sum())

        st.markdown("#### 3. Verdict")
        if n_defauts == 0:
            st.success(
                f"✅ **PIECE CONFORME** — les {len(Y_COLS)} points respectent "
                f"la tolerance de {tolerance:.2f}."
            )
        else:
            bad_points = ", ".join(
                p.replace("value_center_", "") for p in pred_series.index[~conforme_mask]
            )
            st.error(
                f"❌ **DEFAUT DETECTE** — {n_defauts} point(s) hors tolerance "
                f"({tolerance:.2f}) : {bad_points}"
            )

        fig, ax = plt.subplots(figsize=(9, 3.5))
        colors = ["#1F7A72" if ok else "#C9524A" for ok in conforme_mask]
        ax.bar(Y_COLS, pred_series.values, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(tolerance, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(-tolerance, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel("Distorsion optique predite")
        ax.set_title("Distorsion predite par point (pointilles = tolerance)")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig, width='stretch')

        result_table = pred_series.rename("Distorsion predite").to_frame()
        result_table["Statut"] = np.where(conforme_mask, "Conforme", "Defaut")
        st.dataframe(result_table.round(4), width='stretch')
