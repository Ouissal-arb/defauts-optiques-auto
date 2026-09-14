# Prediction de defauts optiques — composants automobiles

## Contenu
- `pipeline3_finale.ipynb` — le pipeline complet (EDA -> pretraitement ->
  split 80/20 -> benchmark RF vs Gradient Boosting -> optimisation ->
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
3. Laissez tourner jusqu'a la derniere section **"9. Telechargement des
   resultats"** : une archive `resultats_projet.zip` se telecharge
   automatiquement dans votre dossier de telechargements. Elle contient :
   - `artifacts/` (modeles, scaler, imputer, metriques — necessaire a
     l'interface Streamlit)
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
   - entrainer et comparer Random Forest et Gradient Boosting ;
   - optimiser les hyperparametres du modele gagnant ;
   - calculer le R2 (global + par point k1-k12) et le MAE ;
   - exporter `predictions_Y_v3.csv` (predictions sur le jeu de test aveugle,
     dans l'ordre original des lignes) ;
   - sauvegarder un dossier `artifacts/` (modeles, scaler, imputer, metriques).

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
