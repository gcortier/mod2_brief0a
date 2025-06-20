# Fichier utilitaire pour partager les variables entre alchemy_api.py et mlflow_utils.py

settings = {
    # source de données de l'entrainnement : Original
    "source_data": "data-all-complete-684bf9cd92797851623245.csv",
    # source de données de l'entrainnement : Cleaned
    "training_data": "df_data_all_complete_cleaned.csv",
    "wanted_train_cycle": 1,
    "epochs": 50,
    "train_seed": 42,
    # les valeurs numériques
    "numerical_cols": ["age", "taille", "poids", "nb_enfants", "quotient_caf", "revenu_estime_mois", "risque_personnel", "loyer_mensuel"],
    # les colonnes catégorielles
    "categorical_cols": ["sport_licence", "niveau_etude", "region", "smoker", "situation_familiale"],
    # les valeurs manquantes à compléter à la moyenne et à ajouter dans une colonne boolean : _missing
    "partials_num_cols": ["score_credit", "historique_credits", "loyer_mensuel"],
    "transfert_weights": {
        "active": True,  # Indique si le transfert de poids est activé
        "run_id": "0eceb24c95c94737a29e6b50b848b0c2"  # URI du modèle à partir duquel les poids seront transférés
    }  
}

wanted_train_cycle = settings.get("wanted_train_cycle", 1)
artifact_path = "sequential_model"
