import sys
import mlflow

from sklearn.model_selection import train_test_split
import json
from modules.preprocess import preprocessing_v2
from models.models import model_predict
import pandas as pd
import joblib
from os.path import join as join
from pydantic import BaseModel
from typing import List, Any
import numpy as np
import os

import pandas as pd
from os.path import join as join
import missingno as msno
    
# from mlflow_shared import settings, artifact_path, wanted_train_cycle
from mlflow_shared_brief2 import settings, artifact_path, wanted_train_cycle


## Base initialisation for Loguru and FastAPI
from myapp_base import setup_loguru, app, Request, HTTPException
logger = setup_loguru("logs/alchemy_api.log")

from datetime import datetime
today_str = datetime.now().strftime("%Y%m%d_%H%M")

# Force url for MLFlow
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment(artifact_path)


# Utilitaire pour lire le run_id courant

def set_last_run_id(run_id):
    try:
        result = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "settings": settings,
            # futur : ajouter le r²
        }
        with open("models/current_model.json", "w") as f:
            json.dump(result, f)
        
        logger.info(f"Nouveau modèle de prédiction mis à jour avec run_id: {run_id} (sauvegardé dans models/current_model.json)")
                     
    except Exception:
        return ""
    
    
def get_last_run_id():
    try:
        with open("models/current_model.json", encoding="utf-8") as f:
            data = json.load(f)
            run_id = data.get("run_id", None)
            return None if run_id == "" else run_id
    except Exception:
        return None


prediction_model = get_last_run_id()  # Variable to hold the prediction model


### Function to train and log a model iteratively in MLFlow
def backup_preprocessor(settings):
    from mlflow_utils import train_and_log_model
    
    """
    Entraîne un modèle et le log dans MLFlow, en utilisant un run_id pour charger un modèle précédent si disponible.
    """
    df = pd.read_csv(join('data', settings["training_data"]))
    logger.info(f"Proceed: {settings['training_data']} with shape {df.shape}")
    
    X_processed, y, preprocessor = preprocessing_v2(df, settings)
    
    joblib.dump(preprocessor, join('models', f'preprocessor_latest.pkl'))



    return preprocessor




### Function to train and log a model iteratively in MLFlow
def train_and_log_iterative(run_idx, settings, run_id=None):
    from mlflow_utils import train_and_log_model
    
    """
    Entraîne un modèle et le log dans MLFlow, en utilisant un run_id pour charger un modèle précédent si disponible.
    """
    df = pd.read_csv(join('data', settings["training_data"]))
    logger.info(f"Training data loaded: {settings['training_data']} with shape {df.shape}")
    
    X_train, X_test, y_train, y_test = prepare_data(df, settings, run_idx)
    
    run_desc = f"Performance for run {run_idx}/{wanted_train_cycle}"
    
    run_id = train_and_log_model(X_train, y_train, X_test, y_test, run_desc, run_id, run_idx, artifact_path)
    return run_id

def prepare_data(df, settings, run_idx=0):
    """
    Prend un DataFrame, applique le préprocessing et split en train/test.
    Retourne X_train, X_test, y_train, y_test
    """
    X, y = preprocessing_v2(df, settings)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42+run_idx)  # Ajout de run_idx pour la reproductibilité
    return X_train, X_test, y_train, y_test

       
def outliers_traitments(df, dimension="revenu_estime_mois"):
    """
    Traite les valeurs aberrantes dans le DataFrame.
    """
    Q1 = df[dimension].quantile(0.25)
    Q3 = df[dimension].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[dimension] >= (Q1 - 1.5 * IQR)) & (df[dimension] <= (Q3 + 1.5 * IQR))]
    return df     


def modelize_study_level(df):
    """
    Convertit la colonne 'niveau_etude' en valeurs numériques.
    """
    if 'niveau_etude' not in df.columns:
        logger.warning("Colonne 'niveau_etude' non trouvée dans le DataFrame.")
        return df

    df['niveau_etude'] = df['niveau_etude'].map({
        'aucun': 0,
        'bac': 1,
        'bac+2': 2,
        'master': 3,
        'doctorat': 4
    })
    return df

   
def fillna_with_missing_indicator(df, columns):
    """
    Pour chaque colonne de la liste, ajoute une colonne <col>_missing (1 si NaN, 0 sinon)
    et remplit les NaN par la moyenne de la colonne.
    """
    for col in columns:
        missing_col = f"{col}_missing"
        df[missing_col] = df[col].isna().astype(int)
        df[col] = df[col].fillna(df[col].mean())
    return df


def clean_dataset(csv_path, csv_target):
    """
    Analyse le dataset, nettoie et sauvegarde le résultat.
    """
    
    collisions = pd.read_csv(csv_path)

    # Suppression des doublons
    collisions = collisions.drop_duplicates()

    # On vire car pas RGPD et pas éthique
    collisions.drop(columns=['nom'], inplace=True, errors='ignore')
    collisions.drop(columns=['prenom'], inplace=True, errors='ignore')
    # On vire car pas éthique
    collisions.drop(columns=['sexe'], inplace=True, errors='ignore')
    collisions.drop(columns=['nationalité_francaise'], inplace=True, errors='ignore')
    collisions.drop(columns=['orientation_sexuelle'], inplace=True, errors='ignore')
    collisions.drop(columns=['date_creation_compte'], inplace=True, errors='ignore')


    # Ajout indicateur de valeurs manquantes(key_missing) + remplissage par la moyenne
    collisions = fillna_with_missing_indicator(collisions, ['loyer_mensuel', 'score_credit', 'historique_credits'])

    # on remplace par la valeur la plus courante
    collisions['situation_familiale'] = collisions['situation_familiale'].fillna(collisions['situation_familiale'].mode()[0])

    # filter outlers values
    collisions = collisions[(collisions['poids'] > 42) & (collisions['loyer_mensuel'] > 300)]

    # process category : niveau_etude to int levels
    modelize_study_level(collisions)


    logger.info(f"Dataset shape after cleaning: {collisions.shape}")

    # Sauvegarde du dataset nettoyé
    collisions.to_csv(csv_target, index=False)
    
     


             
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # python alchemy_api.py train
        # run_id = None
        run_id = prediction_model
        
        for i in range(wanted_train_cycle):
            logger.info(f"Starting training iteration {i} of {wanted_train_cycle}")
            run_id = train_and_log_iterative(i, settings, run_id)
        # Mettre à jour le modèle de prédiction avec le dernier run_id
        set_last_run_id(run_id)
    elif len(sys.argv) > 1 and sys.argv[1] == "clean_dataset":
        # python alchemy_api.py clean_dataset
        logger.info("Analyse des données en cours...")
        
        # Charger les paramètres de configuration
        source_data = settings.get("source_data", "data-all-684bf775c031b265646213.csv")
        training_data = settings.get("training_data", "df_data_all_cleaned.csv")
                
        csv_path = join('data', source_data)
        csv_target = join('data', training_data)
        clean_dataset(csv_path, csv_target)
    elif len(sys.argv) > 1 and sys.argv[1] == "backup_preprocessor":
        # python alchemy_api.py backup_preprocessor
        logger.info("backup_preprocessor en cours...")
        backup_preprocessor(settings)
        
    else:
        print("Aucune action lancée. Pour entraîner, lancez : python alchemy_api.py train")

class PredictRequest(BaseModel):
    data: List[Any]  # Liste des features pour une seule instance

# Modèle Pydantic explicite pour LoanData
class LoanDataBase(BaseModel):
    age: int
    taille: float
    poids: float
    nb_enfants: int
    quotient_caf: float
    sport_licence: str
    niveau_etude: str
    region: str
    smoker: str
    revenu_estime_mois: int
    situation_familiale: str
    risque_personnel: float
    loyer_mensuel: float

    class Config:
        orm_mode = True


# Nouvelle route de prédiction basée sur LoanDataSchema
@app.post("/predict_loandata42")
async def predict_loandata(request: Request, payload: LoanDataBase):
    """
    Endpoint pour faire une prédiction à partir d'un modèle MLflow sauvegardé,
    en utilisant un payload conforme à LoanDataSchema (clé/valeur).
    """
    logger.info(f"Route '{request.url.path}' appelée avec données: {payload.dict()}")
    try:
        run_id = get_last_run_id()
        if not run_id:
            raise HTTPException(status_code=404, detail="Aucun run_id trouvé. Veuillez entraîner un modèle d'abord.")
        from mlflow_utils import MLFlow_load_model
        model = MLFlow_load_model(run_id, artifact_path)
        logger.info(f"Model loaded from MLflow run ID: {run_id}")

        # Charger le préprocesseur
        preprocessor = joblib.load(join('models','preprocessor_latest.pkl'))

        # Récupérer la config des colonnes
        numerical_cols = settings.get("numerical_cols", [])    
        categorical_cols = settings.get("categorical_cols", [])
        partials_num_cols = settings.get("partials_num_cols", [])
        # on ajoute les colonnes missing + la colonne _missing aux colonnes numériques
        numerical_cols += [col for col in partials_num_cols]
        numerical_cols += [col + "_missing" for col in partials_num_cols]
        columns = numerical_cols + categorical_cols

        # Construction du dictionnaire d'entrée
        payload_dict = payload.dict()
        # Compléter les colonnes _missing à 0 si absentes
        for col in partials_num_cols:
            missing_col = col + "_missing"
            if missing_col not in payload_dict:
                payload_dict[missing_col] = 0
        # Compléter toute colonne manquante à None ou 0
        for col in columns:
            if col not in payload_dict:
                payload_dict[col] = 0

        # Construction du DataFrame dans le bon ordre
        X_input = pd.DataFrame([{col: payload_dict[col] for col in columns}])
        logger.info(f"X_input columns: {X_input.columns.tolist()}")
        logger.info(f"X_input values: {X_input.values.tolist()}")

        X_processed = preprocessor.transform(X_input)
        y_pred = model_predict(model, X_processed)
        predict = float(np.asarray(y_pred).squeeze().item())
        logger.info(f"Prediction made: {y_pred}, prediction value: {predict}")
        return {"prediction": predict}
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class RetrainRequest(BaseModel):
    data_path: str  # Chemin du fichier CSV à utiliser comme nouvelle source de données
    from_existing_model: bool = True  # True: fine-tuning, False: nouveau modèle

class RetrainResponse(BaseModel):
    status: str
    run_id: str




@app.get("/current_model")
async def current_model(request: Request):
    """
    Endpoint pour obtenir le run_id du modèle courant.
    """
    logger.info(f"Route '{request.url.path}' called by {request.client.host}")
    run_id = get_last_run_id()
    if not run_id:
        raise HTTPException(status_code=404, detail="Aucun modèle courant trouvé.")
    
    return {"run_id": run_id}
