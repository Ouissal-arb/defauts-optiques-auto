# Prediction de defauts optiques — composants automobiles

## Contenu
- `pipeline3_finale.ipynb` — le pipeline complet (EDA -> pretraitement ->
  split 80/20 -> benchmark RF vs Gradient Boosting -> optimisation ->
  export des predictions), conforme a la feuille de route de l'encadrant.
- `app.py` — interface Streamlit qui presente le projet. Elle ne reentraine
  rien pour les pages de reporting (elle lit les resultats sauvegardes par
  le notebook), mais l'onglet **"Tester une piece"** appelle directement les
  modeles sauvegardes (`artifacts/models.joblib`) pour predire en direct la
  conformite d'une piece saisie manuellement.
- `.streamlit/config.toml` — theme visuel de l'application.
- `requirements.txt` — dependances Python.

## Onglet interactif "Tester une piece"

Cet onglet permet de saisir les 29 mesures de deformation d'une piece (ou
de partir d'un exemple du jeu de test aveugle / des valeurs medianes) et
d'obtenir instantanement :
- la distorsion optique predite aux 12 points de mesure (k1 a k12) ;
- un verdict **Conforme / Defaut** obtenu en comparant chaque point a une
  **tolerance de conformite** (curseur ajustable dans l'interface).

La tolerance par defaut (`metadata["tolerance"]`) est calculee comme la
mediane de `|distorsion|` observee sur les donnees d'entrainement — c'est
un **place-holder statistique**, pas une vraie specification qualite.
**A remplacer par la tolerance metier reelle** (cahier des charges /
maitre de stage) des qu'elle est connue, dans `pipeline3_finale.ipynb`
(section 8, variable `TOLERANCE`) ou directement via le curseur de l'appli.

Cet onglet a besoin des fichiers `artifacts/models.joblib`,
`artifacts/imputer.joblib` et `artifacts/scaler.joblib`. Ces fichiers sont
suivis par **Git LFS** (`*.joblib` dans `.gitattributes`) : assurez-vous de
faire `git lfs pull` (ou d'installer Git LFS avant le clone) pour recuperer
les vrais binaires plutot que de simples pointeurs texte — sinon l'appli
affiche un message d'erreur explicite dans cet onglet.

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
