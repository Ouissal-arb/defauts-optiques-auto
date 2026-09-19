# Prediction de defauts optiques — composants automobiles

## Contenu
- `pipeline3_finale.ipynb` — le pipeline complet (EDA -> pretraitement ->
  split 80/20 -> comparaison de 6 familles de modeles [Random Forest,
  Extra Trees, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM]
  -> optimisation de Random Forest -> independant vs multi-output ->
  export des predictions), conforme a la feuille de route de l'encadrant.
- `app.py` — interface Streamlit qui presente le projet (elle ne reentraine
  rien : elle lit les resultats sauvegardes par le notebook).
- `.streamlit/config.toml` — theme visuel de l'application.
- `requirements.txt` — dependances Python.

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
     HistGradientBoosting, XGBoost et LightGBM (parametres par defaut) ;
   - optimiser les hyperparametres de Random Forest (RandomizedSearch) ;
   - comparer approche independante (1 modele par cible) et multi-output
     (1 seul modele pour les 12 cibles) ;
   - choisir le modele final sur la base du **R2 moyen** (metrique
     principale), en gardant MAE et RMSE comme indicateurs complementaires ;
   - exporter `predictions_Y_v3.csv` (predictions sur le jeu de test aveugle,
     dans l'ordre original des lignes) ;
   - sauvegarder un dossier `artifacts/` (modeles, scaler, imputer, metriques,
     `best_pipeline.joblib`, tableau comparatif complet).

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
