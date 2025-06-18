# Fichier utilitaire pour partager les variables entre alchemy_api.py et mlflow_utils.py

settings = {
    # source de données de l'entrainnement
    "training_data": "df_data_all_complete_cleaned.csv",
    "wanted_train_cycle": 1,
    "epochs": 50,
    "train_seed": 42,
    # les valeurs numériques
    "numerical_cols": ["age", "taille", "poids", "nb_enfants", "quotient_caf", "revenu_estime_mois", "risque_personnel", "loyer_mensuel"],
    # les colonnes catégorielles
    "categorical_cols": ["sport_licence", "niveau_etude", "region", "smoker", "situation_familiale"],
    # les valeurs manquantes à compléter à la moyenne et à ajouter dans une colonne boolean : _missing
    "partials_num_cols": ["score_credit", "historique_credits", "loyer_mensuel"]
}

wanted_train_cycle = settings.get("wanted_train_cycle", 1)
artifact_path = "sequential_model"
