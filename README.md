# Prediction de defauts optiques — composants automobiles

## Contenu
- `pipeline3_finale.ipynb` — le pipeline complet (EDA -> pretraitement ->
  split 80/20 -> comparaison de 6 familles de modeles [Random Forest,
  Extra Trees, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM]
  -> optimisation de Random Forest -> export des predictions), conforme a
  la feuille de route de l'encadrant. **Modele final retenu : Random
  Forest multi-output** (un seul modele qui predit les 12 cibles a la
  fois) — un choix de conception (un seul modele a deployer, correlations
  entre points prises en compte), documente par le benchmark complet des
  7 approches meme si une autre n'est pas toujours premiere au R2.
- `app.py` — interface Streamlit en 7 pages (elle ne reentraine rien : elle
  lit les artefacts sauvegardes par le notebook) :
  - 🏠 **Dashboard** — KPIs de production (pieces testees, taux de
    conformite, distorsion moyenne/max), repartition conforme/non conforme,
    evolution du taux de conformite dans le temps.
  - 🔍 **Tester des pieces** — importer un CSV de nouvelles pieces
    (`id` + les 29 X), prediction des 12 Y, classement conforme/non
    conforme selon une tolerance ajustable, detail piece par piece,
    enregistrement dans l'historique et export CSV des resultats.
  - 📋 **Historique** — recherche, filtre (conforme/non conforme), tri
    (date, distorsion), detail d'un controle passe, export CSV.
  - 📊 Exploration des donnees, 🤖 Benchmark des modeles, 📈 Resultats &
    performance, 📁 Predictions (jeu aveugle) — les pages du projet ML
    d'origine, conservees telles quelles.
- `.streamlit/config.toml` — theme visuel de l'application.
- `requirements.txt` — dependances Python.
- `data/historique.db` — base **SQLite** (creee automatiquement au premier
  controle enregistre) qui stocke l'historique des controles. Un simple
  fichier local suffit pour ce volume de donnees (pas de serveur a
  installer) tout en permettant une vraie recherche/filtre/tri en SQL ;
  ce dossier n'est pas versionne (voir `.gitignore`), chaque poste a son
  propre historique local.

## Comment executer

### Option A — Google Colab

1. Ouvrez `pipeline3_finale.ipynb` dans Colab (File > Upload notebook, ou
   glisser-deposer).
2. Executez les cellules dans l'ordre. A la section **"Upload des donnees"**,
   une fenetre de selection de fichiers s'ouvre : choisissez
   `INDUSTRIAL_AI_TrainValTest.pkl` et `INDUSTRIAL_AI_Infer.pkl` sur votre
   ordinateur (gardez ces noms exacts).
3. Laissez tourner jusqu'a la derniere section **"11. Telechargement des
   resultats"** : une archive `resultats_projet.zip` se telecharge
   automatiquement dans votre dossier de telechargements. Elle contient :
   - `artifacts/` (modeles, scaler, imputer, metriques, `best_pipeline.joblib`
     [bundle unique imputer+scaler+modeles pret a charger depuis l'appli],
     `model_comparison_full.csv` / `model_comparison_summary.csv` [tableau
     comparatif complet des 7 experiences, a reutiliser dans le rapport de
     stage] — necessaire a l'interface Streamlit)
   - `predictions_Y_v3.csv` (le livrable a remettre a votre encadrant)
4. **En local** : extrayez le zip, placez le dossier `artifacts/` a cote de
   `app.py`, puis :
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

### Option B — Local (Jupyter / VS Code)

1. **Placer les donnees** `INDUSTRIAL_AI_TrainValTest.pkl` et
   `INDUSTRIAL_AI_Infer.pkl` dans le meme dossier que `pipeline3_finale.ipynb`.

2. **Installer les dependances** :
   ```bash
   pip install -r requirements.txt
   ```

3. **Executer le notebook du debut a la fin.** Il va :
   - afficher l'EDA (valeurs manquantes, valeurs aberrantes) ;
   - entrainer et comparer Random Forest, Extra Trees, Gradient Boosting,
     HistGradientBoosting, XGBoost et LightGBM (parametres par defaut,
     un modele independant par cible) ;
   - **regulariser Random Forest multi-output par validation croisee 5-fold** :
     recherche sur `max_depth` (reduit), `min_samples_leaf` / `min_samples_split`
     (augmentes), `max_features` (limite) et `n_estimators`, en comparant
     explicitement R2 train vs R2 validation pour chaque combinaison (et vs
     la configuration par defaut, qui memorise parfaitement le train —
     R2=1.0 — signe de sur-apprentissage) ;
   - retenir ce **Random Forest multi-output regularise** comme modele final
     (un seul modele pour les 12 cibles) — ce choix est documente par le
     tableau comparatif de la section 4, meme s'il n'est pas toujours celui
     qui a le R2 independant le plus eleve ;
   - exporter `predictions_Y_v3.csv` (predictions sur le jeu de test aveugle,
     dans l'ordre original des lignes) ;
   - sauvegarder un dossier `artifacts/` (modele unique, scaler, imputer,
     metriques, `best_pipeline.joblib`, tableau comparatif complet).

   ⚠️ Cette recherche d'hyperparametres (section 5) est plus longue qu'avant
   (validation croisee 5-fold x 15 combinaisons + la reference = 80
   entrainements) : comptez plus de temps que lors d'un run precedent,
   surtout sur le vrai jeu de donnees (~5000+ lignes). Reduisez `N_ITER`
   dans la section 5 si besoin.

4. **Lancer l'interface Streamlit** (depuis le meme dossier, une fois
   `artifacts/` genere) :
   ```bash
   streamlit run app.py
   ```

## Points a verifier / ajuster avec vos vraies donnees
- `OUTLIER_Y_MAX` (section 2.2 du notebook) : seuil au-dela duquel une
  valeur de distorsion est consideree comme une erreur capteur. Un point de
  depart (50) est propose, a confirmer visuellement sur les histogrammes
  affiches par le notebook.
- `N_ITER` (section 5, recherche d'hyperparametres) : nombre de combinaisons
  testees. 15 est un bon compromis qualite/temps de calcul ; augmentez-le si
  votre machine le permet.
- **Tolerance de conformite** (page "Tester des pieces" de `app.py`) : la
  valeur par defaut (1.0) est **fictive**, a remplacer par la vraie
  tolerance industrielle une fois validee avec l'encadrant. Elle est
  ajustable directement dans l'interface (pas besoin de modifier le code).
