# Fichier utilitaire pour partager les variables entre alchemy_api.py et mlflow_utils.py

settings = {
    "dataversion": "df_data_all_cleaned.csv",
    "wanted_train_cycle": 1,
    "epochs": 50,
    "train_seed": 4242,
}

wanted_train_cycle = settings.get("wanted_train_cycle", 1)
artifact_path = "sequential_model"
