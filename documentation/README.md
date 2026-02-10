# 📊 Projet BAAC – Analyse des accidents en France

Ce projet traite des données BAAC des accidents en France (2019–2024) pour explorer, analyser et modéliser la gravité des accidents.
Les données sont disponibles ici : [Data](data.md)


Le projet comprend deux notebooks principaux et un rapport écrit résumant l’ensemble des analyses.

# Contenu du projet
1. Notebook : Analyse exploratoire & modèles de machine learning 
    |
    |-> [Accident](../notebooks/accident.ipynb)

Fonctionnalités principales :

* Analyse exploratoire

* Découpage temporel Train/Test : train = 2019–2023, test = 2024 (pour éviter toute fuite d’information et tester la généralisation sur une année future).

* Échantillonnage contrôlé : sous-échantillon d’entraînement N_SAMPLE = 30 000 pour compatibilité avec GridSearchCV, tout en conservant la distribution de la cible grav_acc.

* Pipelines et prétraitement reproductible : Pipeline + ColumnTransformer (imputation médiane pour les numériques, standardisation, imputation catégorielle “most_frequent”, OneHotEncoding).

* Comparaison multi-modèles : Logistic Regression, Random Forest, Gradient Boosting, avec focus métier sur le rappel (recall) de la classe grave.

* Analyse post-résultats : rapports de classification, matrices de confusion, ROC-AUC, faux négatifs mis à jour pour 2019–2024.

* Seuil de décision métier : ajustement des probabilités sur la régression logistique (t ∈ {0.5, 0.4, 0.3, 0.2}) pour limiter les faux négatifs (seuil_final = 0.4).  
  
2. Notebook : Démarche MLflow
    |
    |-> [Mlflow](../notebooks/mlflow.ipynb)

Fonctionnalités principales :

* Tracking local standardisé : mlflow.set_tracking_uri("file:../mlruns") et définition de l’expérience mlflow.set_experiment("BAAC").

* Run MLflow complet et traçable : encapsulation de l’évaluation du modèle dans mlflow.start_run(...).

* Logging enrichi :

    - Paramètres : hyperparamètres (best_params_) et tailles des datasets (n_train_total, n_train_sample, n_test_2024).

    - Métriques : recall_2024 (principal), precision_2024, f1_2024, roc_auc_2024.

    - Artefacts MLflow :

    - Export du classification_report en .txt.

    - Sauvegarde de la matrice de confusion en .png.

    - Logging du modèle complet (pipeline preprocessing + classifier) via mlflow.sklearn.log_model(...) pour réutilisation et reproductibilité.

3. Rapport écrit

Résumé complet du projet, analyses et conclusions : [Rapport](rapport.md)

