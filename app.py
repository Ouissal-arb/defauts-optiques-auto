"""
Interface Streamlit - Prediction de defauts optiques (composants automobiles)
================================================================================
Application d'inspection : elle lit les artefacts produits par le notebook
`pipeline3_finale.ipynb` (dossier `artifacts/`) pour presenter la demarche ML,
et permet en plus de tester de nouvelles pieces (CSV de mesures X), de
determiner leur conformite, d'enregistrer l'historique des controles et de
suivre un dashboard qualite.

Lancer avec :  streamlit run app.py
(a executer depuis le dossier qui contient a la fois app.py et artifacts/)
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ARTIFACTS_DIR = Path("artifacts")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
HISTORY_DB = DATA_DIR / "historique.db"

st.set_page_config(
    page_title="Defauts optiques - Composants automobiles",
    page_icon="🔎",
    layout="wide",
)

# ------------------------------------------------------------------ palette
# Couleurs centralisees, reutilisees dans toute l'appli (CSS + graphiques) :
# un accent d'identite (teal), deux couleurs de statut reservees (jamais
# utilisees pour autre chose que conforme/non conforme), une couleur neutre
# pour les valeurs "informatives", et une palette categorique fixe pour les
# graphiques a plusieurs series (comparaison de modeles).
COLOR_ACCENT = "#1F7A72"
COLOR_GOOD = "#2F9E44"
COLOR_CRITICAL = "#E03131"
COLOR_NEUTRAL = "#516361"
CATEGORICAL_PALETTE = ["#1F7A72", "#C97B4A", "#4C72B0", "#8172B2", "#C44E52", "#55A868", "#937860"]

# ------------------------------------------------------------------ styling
st.markdown(
    """
    <style>
    html, body, [class*="css"]  { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8E7;
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        box-shadow: 0 1px 3px rgba(15, 61, 56, 0.08);
    }
    .metric-card .metric-label {
        color: #516361;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 600;
    }
    .metric-card .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        line-height: 1.5;
    }
    .accent { color: #1F7A72; }
    .status-pill {
        display: inline-block;
        padding: 0.15rem 0.65rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-pill.good { background-color: #E5F6E9; color: #1F7A34; }
    .status-pill.critical { background-color: #FCE8E8; color: #B02525; }
    .page-caption { color: #516361; font-size: 1rem; margin-bottom: 1rem; }
    h1, h2, h3 { letter-spacing: -0.01em; }
    </style>
    """,
    unsafe_allow_html=True,
)


def stat_tile(label, value, color=COLOR_ACCENT):
    """Petite carte KPI HTML (label en majuscules + valeur coloree)."""
    return (
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{color}">{value}</div></div>'
    )


def status_pill(is_good, text_good="Conforme", text_bad="Non conforme"):
    cls = "good" if is_good else "critical"
    text = text_good if is_good else text_bad
    icon = "🟢" if is_good else "🔴"
    return f'<span class="status-pill {cls}">{icon} {text}</span>'


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
def load_model():
    path = ARTIFACTS_DIR / "models.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_resource
def load_preprocessing():
    imputer_path = ARTIFACTS_DIR / "imputer.joblib"
    scaler_path = ARTIFACTS_DIR / "scaler.joblib"
    if not (imputer_path.exists() and scaler_path.exists()):
        return None, None
    return joblib.load(imputer_path), joblib.load(scaler_path)


# ------------------------------------------------------------------ historique (SQLite)
# Une base de donnees locale (fichier unique, pas de serveur) suffit largement
# pour ce volume de donnees et permet de rechercher/filtrer/trier facilement,
# ce qu'un simple CSV rendrait plus penible a mesure que l'historique grandit.
def get_connection():
    conn = sqlite3.connect(HISTORY_DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS controles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            horodatage TEXT NOT NULL,
            lot TEXT NOT NULL,
            piece_id TEXT NOT NULL,
            distorsion_max REAL NOT NULL,
            distorsion_moyenne REAL NOT NULL,
            statut TEXT NOT NULL,
            cibles_hors_tolerance TEXT NOT NULL,
            tolerance REAL NOT NULL,
            y_json TEXT NOT NULL
        )
        """
    )
    return conn


def save_batch_to_history(lot_name, result_df, y_cols, tolerance):
    conn = get_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for _, row in result_df.iterrows():
        y_vals = row[y_cols].astype(float)
        hors = [c for c in y_cols if abs(y_vals[c]) > tolerance]
        statut = "Conforme" if not hors else "Non conforme"
        rows.append((
            now, str(lot_name), str(row.get("id", "")),
            float(y_vals.abs().max()), float(y_vals.abs().mean()),
            statut, json.dumps(hors), float(tolerance), json.dumps(y_vals.to_dict()),
        ))
    conn.executemany(
        "INSERT INTO controles (horodatage, lot, piece_id, distorsion_max, distorsion_moyenne, "
        "statut, cibles_hors_tolerance, tolerance, y_json) VALUES (?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


def load_history():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM controles ORDER BY horodatage DESC", conn)
    conn.close()
    if not df.empty:
        df["horodatage"] = pd.to_datetime(df["horodatage"])
    return df


# ------------------------------------------------------------------ pipeline de prediction
def predict_pieces(raw_df, x_cols, y_cols, imputer, scaler, model):
    X_imp = pd.DataFrame(imputer.transform(raw_df[x_cols]), columns=x_cols)
    X_scaled = pd.DataFrame(scaler.transform(X_imp), columns=x_cols)
    y_pred = model.predict(X_scaled)
    result = raw_df.reset_index(drop=True).copy()
    for j, col in enumerate(y_cols):
        result[col] = y_pred[:, j]
    return result


def classify_pieces(result_df, y_cols, tolerance):
    result_df = result_df.copy()
    abs_y = result_df[y_cols].abs()
    result_df["distorsion_max"] = abs_y.max(axis=1)
    result_df["distorsion_moyenne"] = abs_y.mean(axis=1)
    result_df["cibles_hors_tolerance"] = result_df[y_cols].apply(
        lambda row: [c for c in y_cols if abs(row[c]) > tolerance], axis=1
    )
    result_df["conforme"] = result_df["cibles_hors_tolerance"].apply(lambda x: len(x) == 0)
    return result_df


metadata = load_metadata()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🔎 Defauts optiques")
st.sidebar.caption("Pieces automobiles transparentes")
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔍 Tester des pieces",
        "📋 Historique",
        "📊 Exploration des donnees",
        "🤖 Benchmark des modeles",
        "📈 Resultats & performance",
        "📁 Predictions (jeu aveugle)",
    ],
)

if metadata is None:
    st.warning(
        "Aucun artefact trouve dans le dossier `artifacts/`.\n\n"
        "Lancez d'abord le notebook **pipeline3_finale.ipynb** en entier : il "
        "entraine le modele et sauvegarde tout ce dont cette application a "
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
if page == "🏠 Dashboard":
    st.title("🏠 Dashboard qualite")
    st.markdown(
        '<p class="page-caption">Vue d\'ensemble des controles effectues sur les pieces '
        "automobiles (donnees enregistrees depuis la page <b>Tester des pieces</b>).</p>",
        unsafe_allow_html=True,
    )

    history_df = load_history()

    if history_df.empty:
        st.info(
            "Aucun controle enregistre pour le moment. Va dans **🔍 Tester des pieces** "
            "pour importer un fichier CSV et lancer ton premier controle : il apparaitra "
            "ici automatiquement une fois enregistre."
        )
    else:
        n_total = len(history_df)
        n_conforme = int((history_df["statut"] == "Conforme").sum())
        n_non_conforme = n_total - n_conforme
        taux_conformite = n_conforme / n_total * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(stat_tile("Pieces testees", n_total), unsafe_allow_html=True)
        c2.markdown(stat_tile("Conformes", n_conforme, COLOR_GOOD), unsafe_allow_html=True)
        c3.markdown(stat_tile("Non conformes", n_non_conforme, COLOR_CRITICAL), unsafe_allow_html=True)
        c4.markdown(stat_tile("Taux de conformite", f"{taux_conformite:.1f}%"), unsafe_allow_html=True)

        c5, c6 = st.columns(2)
        c5.markdown(
            stat_tile("Distorsion moyenne", f"{history_df['distorsion_moyenne'].mean():.3f}", COLOR_NEUTRAL),
            unsafe_allow_html=True,
        )
        c6.markdown(
            stat_tile("Distorsion maximale observee", f"{history_df['distorsion_max'].max():.3f}", COLOR_NEUTRAL),
            unsafe_allow_html=True,
        )

        st.markdown("### Repartition des controles")
        col1, col2 = st.columns([1, 2])
        with col1:
            fig, ax = plt.subplots(figsize=(4, 4))
            wedges, _, autotexts = ax.pie(
                [n_conforme, n_non_conforme],
                labels=["Conforme", "Non conforme"],
                autopct="%1.0f%%",
                colors=[COLOR_GOOD, COLOR_CRITICAL],
                wedgeprops={"edgecolor": "white", "linewidth": 2},
                startangle=90,
            )
            for t in autotexts:
                t.set_color("white")
                t.set_fontweight("bold")
            ax.set_title("Conformite globale")
            st.pyplot(fig, width='stretch')

        with col2:
            evo = history_df.copy()
            evo["date"] = evo["horodatage"].dt.date
            daily = evo.groupby("date")["statut"].value_counts().unstack(fill_value=0)
            for col in ["Conforme", "Non conforme"]:
                if col not in daily.columns:
                    daily[col] = 0
            daily["taux_conformite"] = daily["Conforme"] / (daily["Conforme"] + daily["Non conforme"]) * 100

            if len(daily) >= 2:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(daily.index, daily["taux_conformite"], marker="o", color=COLOR_ACCENT, linewidth=2)
                ax.fill_between(daily.index, daily["taux_conformite"], color=COLOR_ACCENT, alpha=0.08)
                ax.set_ylabel("Taux de conformite (%)")
                ax.set_ylim(0, 105)
                ax.set_title("Evolution du taux de conformite")
                ax.grid(axis="y", alpha=0.25)
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                st.pyplot(fig, width='stretch')
            else:
                st.info(
                    "Pas encore assez de journees distinctes pour tracer une evolution — "
                    "reviens quand tu auras fait des controles sur plusieurs jours."
                )

        st.markdown("### Derniers controles")
        st.dataframe(
            history_df[["horodatage", "lot", "piece_id", "distorsion_max", "statut"]].head(10),
            width='stretch',
        )

# =====================================================================
elif page == "🔍 Tester des pieces":
    st.title("🔍 Tester des pieces")
    st.markdown(
        '<p class="page-caption">Importe un fichier CSV de nouvelles pieces '
        "(colonne <code>id</code> + les 29 variables X) pour predire leurs distorsions "
        "optiques et determiner leur conformite.</p>",
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ Format attendu du fichier CSV"):
        st.write(f"**{len(X_COLS)} colonnes X requises**, en plus d'une colonne `id` (optionnelle) :")
        st.code(", ".join(X_COLS), language=None)

    uploaded_file = st.file_uploader("Fichier CSV des pieces a tester", type=["csv"])

    st.markdown("#### Tolerance de conformite")
    st.warning(
        "⚠️ La vraie tolerance industrielle doit etre validee avec ton encadrant — "
        "la valeur ci-dessous est un point de depart modifiable, pas une valeur officielle."
    )
    tolerance = st.number_input(
        "Tolerance |distorsion| maximale acceptee, par point de mesure (k1 a k12)",
        min_value=0.0, value=1.0, step=0.1, format="%.2f",
    )

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Impossible de lire le fichier : {e}")
            st.stop()

        missing_cols = [c for c in X_COLS if c not in raw_df.columns]
        if missing_cols:
            st.error(f"Colonnes manquantes dans le CSV : {missing_cols}")
            st.stop()
        if "id" not in raw_df.columns:
            raw_df.insert(0, "id", range(1, len(raw_df) + 1))

        imputer, scaler = load_preprocessing()
        model = load_model()
        if imputer is None or scaler is None or model is None:
            st.error(
                "Artefacts de modele introuvables (`artifacts/imputer.joblib`, `scaler.joblib`, "
                "`models.joblib`). Lance d'abord le notebook `pipeline3_finale.ipynb`."
            )
            st.stop()

        result_df = predict_pieces(raw_df, X_COLS, Y_COLS, imputer, scaler, model)
        result_df = classify_pieces(result_df, Y_COLS, tolerance)

        n_total = len(result_df)
        n_conforme = int(result_df["conforme"].sum())
        n_non_conforme = n_total - n_conforme

        c1, c2, c3 = st.columns(3)
        c1.markdown(stat_tile("Pieces testees", n_total), unsafe_allow_html=True)
        c2.markdown(stat_tile("Conformes", n_conforme, COLOR_GOOD), unsafe_allow_html=True)
        c3.markdown(stat_tile("Non conformes", n_non_conforme, COLOR_CRITICAL), unsafe_allow_html=True)

        st.markdown("#### Resultats")
        display_df = result_df[["id"] + Y_COLS + ["distorsion_max"]].round(4).copy()
        display_df["statut"] = result_df["conforme"].map({True: "🟢 Conforme", False: "🔴 Non conforme"})
        st.dataframe(display_df, width='stretch', height=350)

        st.markdown("#### Explorer une piece")
        piece_ids = result_df["id"].astype(str).tolist()
        selected_id = st.selectbox("Choisir une piece", piece_ids)
        piece_row = result_df[result_df["id"].astype(str) == selected_id].iloc[0]

        fig, ax = plt.subplots(figsize=(9, 3.5))
        bar_colors = [COLOR_CRITICAL if abs(piece_row[c]) > tolerance else COLOR_GOOD for c in Y_COLS]
        ax.bar(Y_COLS, piece_row[Y_COLS].values, color=bar_colors)
        ax.axhline(tolerance, color=COLOR_CRITICAL, linestyle="--", linewidth=1, label=f"Tolerance (±{tolerance:.2f})")
        ax.axhline(-tolerance, color=COLOR_CRITICAL, linestyle="--", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Distorsion predite")
        ax.set_title(f"Piece {selected_id}")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8, loc="upper right")
        plt.tight_layout()
        st.pyplot(fig, width='stretch')

        if piece_row["conforme"]:
            st.markdown(status_pill(True) + " — tous les points sont dans la tolerance.", unsafe_allow_html=True)
        else:
            details = ", ".join(f"{c} = {piece_row[c]:.3f}" for c in piece_row["cibles_hors_tolerance"])
            st.markdown(status_pill(False) + f" — points hors tolerance : {details}", unsafe_allow_html=True)

        st.markdown("#### Enregistrer et exporter")
        col_a, col_b = st.columns(2)
        with col_a:
            lot_name = st.text_input("Nom du lot (pour l'historique)", value=Path(uploaded_file.name).stem)
            if st.button("💾 Enregistrer dans l'historique"):
                save_batch_to_history(lot_name, result_df, Y_COLS, tolerance)
                st.success(f"{n_total} piece(s) enregistree(s) dans l'historique.")
        with col_b:
            export_df = result_df[["id"] + X_COLS + Y_COLS + ["distorsion_max"]].copy()
            export_df["statut"] = result_df["conforme"].map({True: "Conforme", False: "Non conforme"})
            st.download_button(
                "📥 Telecharger le CSV des resultats",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name=f"resultats_{Path(uploaded_file.name).stem}.csv",
                mime="text/csv",
            )

# =====================================================================
elif page == "📋 Historique":
    st.title("📋 Historique des controles")

    history_df = load_history()
    if history_df.empty:
        st.info(
            "Aucun controle enregistre pour le moment — va dans **🔍 Tester des pieces** "
            "puis clique sur **Enregistrer dans l'historique** apres un controle."
        )
    else:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search = st.text_input("Rechercher (id de piece ou nom de lot)")
        with col2:
            statut_filter = st.multiselect(
                "Statut", ["Conforme", "Non conforme"], default=["Conforme", "Non conforme"]
            )
        with col3:
            tri = st.selectbox(
                "Trier par", ["Date (recent)", "Date (ancien)", "Distorsion (desc)", "Distorsion (asc)"]
            )

        filtered = history_df[history_df["statut"].isin(statut_filter)]
        if search:
            mask = (
                filtered["piece_id"].astype(str).str.contains(search, case=False, na=False)
                | filtered["lot"].astype(str).str.contains(search, case=False, na=False)
            )
            filtered = filtered[mask]

        sort_map = {
            "Date (recent)": ("horodatage", False),
            "Date (ancien)": ("horodatage", True),
            "Distorsion (desc)": ("distorsion_max", False),
            "Distorsion (asc)": ("distorsion_max", True),
        }
        sort_col, ascending = sort_map[tri]
        filtered = filtered.sort_values(sort_col, ascending=ascending)

        st.caption(f"{len(filtered)} controle(s) affiche(s) sur {len(history_df)} au total.")
        display_hist = filtered[["horodatage", "lot", "piece_id", "distorsion_max", "distorsion_moyenne", "statut"]].copy()
        display_hist[["distorsion_max", "distorsion_moyenne"]] = display_hist[["distorsion_max", "distorsion_moyenne"]].round(4)
        st.dataframe(display_hist, width='stretch', height=380)

        if not filtered.empty:
            st.markdown("#### Consulter le detail d'une piece")
            options = [
                f"{r.lot} — {r.piece_id} — {r.horodatage:%Y-%m-%d %H:%M}" for r in filtered.itertuples()
            ]
            selected = st.selectbox("Controle", options)
            selected_row = filtered.iloc[options.index(selected)]
            y_values = json.loads(selected_row["y_json"])
            hors = json.loads(selected_row["cibles_hors_tolerance"])

            fig, ax = plt.subplots(figsize=(9, 3.5))
            cols_k = list(y_values.keys())
            vals = list(y_values.values())
            bar_colors = [COLOR_CRITICAL if c in hors else COLOR_GOOD for c in cols_k]
            ax.bar(cols_k, vals, color=bar_colors)
            ax.axhline(selected_row["tolerance"], color=COLOR_CRITICAL, linestyle="--", linewidth=1)
            ax.axhline(-selected_row["tolerance"], color=COLOR_CRITICAL, linestyle="--", linewidth=1)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.tick_params(axis="x", rotation=45)
            ax.set_title(f"{selected_row['piece_id']} ({selected_row['lot']})")
            plt.tight_layout()
            st.pyplot(fig, width='stretch')

            is_good = selected_row["statut"] == "Conforme"
            st.markdown(status_pill(is_good), unsafe_allow_html=True)

        st.download_button(
            "📥 Telecharger l'historique filtre (CSV)",
            data=filtered.drop(columns=["y_json"]).to_csv(index=False).encode("utf-8"),
            file_name="historique_controles.csv",
            mime="text/csv",
        )

# =====================================================================
elif page == "📊 Exploration des donnees":
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
elif page == "🤖 Benchmark des modeles":
    st.title("Comparaison des modeles")
    st.markdown(
        "Chaque famille de modele est entrainee et evaluee **sur le meme split** "
        "80/20 pour une comparaison equitable, avec un modele independant par "
        "point (k1 a k12). Le modele final retenu peut differer du meilleur R2 "
        "ci-dessous : c'est un choix de conception (modele unique multi-output), "
        "documente ici pour le rapport de stage."
    )

    if "model_comparison" in metadata:
        comparison = pd.DataFrame(metadata["model_comparison"]).set_index("modele")
    else:
        # compatibilite avec un ancien metadata.json (benchmark RF vs GB uniquement)
        comparison = pd.DataFrame(metadata["benchmark"]).T.rename(
            columns={"R2": "R2_moyen", "MAE": "MAE_moyen"}
        )
    comparison = comparison.sort_values("R2_moyen", ascending=False)
    winner = metadata["winner_algo"]

    col1, col2 = st.columns([1, 2])
    with col1:
        cols_to_show = [c for c in ["R2_moyen", "MAE_moyen", "RMSE_moyen", "temps_entrainement_s"] if c in comparison.columns]
        st.dataframe(comparison[cols_to_show].round(4), width='stretch')
        st.success(f"Modele retenu : **{winner}**")

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = [COLOR_ACCENT if name == winner else "#C9D6D4" for name in comparison.index]
        ax.barh(comparison.index, comparison["R2_moyen"], color=colors)
        ax.set_xlabel("R2 moyen (validation, moyenne des 12 cibles)")
        ax.axvline(0, color="black", linewidth=0.8)
        for i, v in enumerate(comparison["R2_moyen"]):
            ax.text(v, i, f" {v:.3f}", va="center")
        ax.invert_yaxis()
        st.pyplot(fig, width='stretch')

    st.markdown("#### Hyperparametres retenus (modele final)")
    st.json(metadata["best_params"] if metadata.get("best_params") else
            {"info": "parametres par defaut (pas d'optimisation dediee a ce modele)"})

# =====================================================================
elif page == "📈 Resultats & performance":
    st.title("Performance du modele final")

    c1, c2 = st.columns(2)
    c1.markdown(stat_tile("R2 global (validation)", f"{global_row['R2']:.3f}"), unsafe_allow_html=True)
    c2.markdown(stat_tile("MAE global (validation)", f"{global_row['MAE']:.3f}"), unsafe_allow_html=True)

    st.markdown(
        "Suivi granulaire par point de mesure, pour identifier les eventuelles "
        "faiblesses locales du modele :"
    )
    st.dataframe(per_target.round(4), width='stretch')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(per_target.index, per_target["R2"], color=COLOR_ACCENT)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("R2 par point")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(per_target.index, per_target["MAE"], color="#C9524A")
    axes[1].set_title("MAE par point")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    st.pyplot(fig, width='stretch')

# =====================================================================
elif page == "📁 Predictions (jeu aveugle)":
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
        ax.bar(Y_COLS, row.values, color=COLOR_ACCENT)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Distorsion optique predite")
        ax.set_title(f"Piece #{int(idx)} — distorsion predite par point (0 = conforme)")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig, width='stretch')

        if infer_X is not None:
            with st.expander("Voir les mesures de deformation en entree pour cette piece"):
                st.dataframe(infer_X.iloc[[int(idx)]].T.rename(columns={int(idx): "valeur"}))
