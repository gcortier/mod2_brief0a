import sys
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

import json

from modules.preprocess import preprocessing, split
from modules.evaluate import evaluate_performance
from modules.print_draw import print_data, draw_loss
from models.models import create_nn_model, train_model, model_predict
import pandas as pd
import joblib
from os.path import join as join
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Any
import numpy as np
import os

import pandas as pd
from os.path import join as join
import missingno as msno
    
from datetime import datetime

## Base initialisation for Loguru and FastAPI
from myapp_base import setup_loguru, create_app
logger = setup_loguru("logs/alchemy_api.log")

app = create_app()
today_str = datetime.now().strftime("%Y%m%d_%H%M")

# Force url for MLFlow
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))


# Utilitaire pour lire le run_id courant

def set_last_run_id(run_id):
    try:
        result = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
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


settings = {
    "description": "not set ",
    "dataversion": "data-all-684bf775c031b265646213.csv", 
    "wanted_train_cycle": 3,  # nombre d'entraînements à effectuer 3 est le meilleur
    "epochs": 50,  
    "train_seed": 42,  
}

# Paramètres d'entraînement
wanted_train_cycle = settings.get("wanted_train_cycle", 1)  # nombre d'entraînements à effectuer
artifact_path = "linear_regression_model"

prediction_model = get_last_run_id()  # Variable to hold the prediction model

from mlflow_utils import MLFlow_train_model, MLFlow_load_model, MLFlow_make_prediction

### Function to train and log a model iteratively in MLFlow
def train_and_log_iterative(run_idx, info, run_id=None):
    """
    Entraîne un modèle et le log dans MLFlow, en utilisant un run_id pour charger un modèle précédent si disponible.
    """
    df = pd.read_csv(join('data', info["dataversion"]))
    X_train, X_test, y_train, y_test = prepare_data(df, run_idx)
    
    run_desc = f"Performance for run {run_idx}/{wanted_train_cycle}"
    
    run_id = train_and_log_model(X_train, y_train, X_test, y_test, run_desc, run_id, run_idx, artifact_path)
    return run_id

def prepare_data(df, run_idx=0):
    """
    Prend un DataFrame, applique le préprocessing et split en train/test.
    Retourne X_train, X_test, y_train, y_test
    """
    X, y, _ = preprocessing(df)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42+run_idx)  # Ajout de run_idx pour la reproductibilité
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)  # Ajout de run_idx pour la reproductibilité
    return X_train, X_test, y_train, y_test

def train_and_log_model(X_train, y_train, X_test, y_test, run_desc, model_id=None, run_idx=0, artifact_path="linear_regression_model"):
    """
    Entraîne un modèle, loggue dans MLflow, retourne le model_id.
    """
     # Charger le modèle du run précédent ou créer un nouveau modèle
    if model_id is not None:
        logger.info(f"Loading model from previous model_id: {model_id}")
        model = MLFlow_load_model(model_id, artifact_path)
    else:
        logger.info("No previous model_id, creating new model.")
        model = create_nn_model(X_train.shape[1])
        model_id = "None"  # Reset model_id if no previous model is loaded
    
    step_base_name = f"model_{today_str}_{run_idx}_{model_id}"
    model, hist = MLFlow_train_model({
        "save_model": False, # should be False when tests are finished
        "save_cost": True, # should be False when tests are finished
        "step_base_name": step_base_name,
        "step": run_idx
    }, model, X_train, y_train, X_val=X_test, y_val=y_test, epochs=50, batch_size=32, verbose=0)
    
    preds = MLFlow_make_prediction(model, X_test)
    
    perf = evaluate_performance(y_test, preds)
    print_data(perf, exp_name=run_desc)
    logger.info(f"Model performances: {perf}")
    

    with mlflow.start_run() as run:
        mlflow.log_param("description", run_desc)
        mlflow.log_param("data_version", settings.get("dataversion", "df_old.csv"))
        mlflow.log_param("random_state", settings.get("train_seed", 42))
        mlflow.log_param("previous_run_id", model_id if model_id else "None")
        mlflow.log_metric("mse", perf['MSE'])
        mlflow.log_metric("mae", perf['MAE'])
        mlflow.log_metric("r2", perf['R²'])
        mlflow.sklearn.log_model(model, artifact_path)
        logger.info(f"Run {run_idx + 1} terminé, run_id={run.info.run_id}")
        return run.info.run_id
       
       
       
def outliers_traitments(df, dimension="revenu_estime_mois"):
    """
    Traite les valeurs aberrantes dans le DataFrame.
    Utilise missingno pour visualiser les données manquantes et les valeurs aberrantes.
    """
    Q1 = df[dimension].quantile(0.25)
    Q3 = df[dimension].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[dimension] >= (Q1 - 1.5 * IQR)) & (df[dimension] <= (Q3 + 1.5 * IQR))]
    return df     


def modelize_study_level(df):
    df['niveau_etude'] = df['niveau_etude'].map({
        'aucun': 0,
        'bac': 1,
        'bac+2': 2,
        'master': 3,
        'doctorat': 4
    })
    return df

   
       
def analyse_dataset():
    # collisions = pd.read_csv(join('data', "data_numeric_only.csv"))
    collisions = pd.read_csv(join('data', "data-all-68482f115ac04033078508.csv"))
    
    # collisions.info()
    # collisions.describe()
    
    # Suppression des doublons
    collisions = collisions.drop_duplicates()

    # On vire car pas RGPD et pas éthique
    collisions.drop(columns=['nom'], inplace=True, errors='ignore')
    collisions.drop(columns=['prenom'], inplace=True, errors='ignore')
    # On vire car pas éthique
    collisions.drop(columns=['sexe'], inplace=True, errors='ignore')
    collisions.drop(columns=['nationalité_francaise'], inplace=True, errors='ignore')
    # On vire car trop peu de données
    collisions.drop(columns=['score_credit'], inplace=True, errors='ignore')
    collisions.drop(columns=['historique_credits'], inplace=True, errors='ignore')

    # on rempli loyer mensuel avec la moyenne 
    collisions['loyer_mensuel'] = collisions['loyer_mensuel'].fillna(collisions['loyer_mensuel'].mean())

    # filter outlers values
    collisions = collisions[(collisions['poids'] > 30) & (collisions['loyer_mensuel'] > 0)]


    # process category
    modelize_study_level(collisions)


    logger.info(f"Dataset shape after cleaning: {collisions.shape}")

    # Sauvegarde du dataset nettoyé
    collisions.to_csv(join('data', 'df_cleaned.csv'), index=False)
     


             
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        
        # run_id = None
        run_id = prediction_model
        
        for i in range(wanted_train_cycle):
            logger.info(f"Starting training iteration {i} of {wanted_train_cycle}")
            run_id = train_and_log_iterative(i, settings, run_id)
        # Mettre à jour le modèle de prédiction avec le dernier run_id
        set_last_run_id(run_id)
    elif len(sys.argv) > 1 and sys.argv[1] == "analyse":
        # python mlFlow_api.py analyse
        logger.info("Analyse des données en cours...")
        analyse_dataset()
        
    else:
        print("Aucune action lancée. Pour entraîner, lancez : python mlFlow_api.py train")
        
        
@app.get("/health")
async def health(request: Request):
    """
    Endpoint de santé pour vérifier que l'application fonctionne.
    """
    logger.info(f"Route '{request.url.path}' called by {request.client.host}")
    return {"status": "healthy", "message": "API is running"}






class PredictRequest(BaseModel):
    data: List[Any]  # Liste des features pour une seule instance

@app.post("/predict")
async def predict(request: Request, payload: PredictRequest):
    """
    Endpoint pour faire une prédiction à partir d'un modèle MLflow sauvegardé.
    """
    logger.info(f"Route '{request.url.path}' called with data: {payload.data}")
    try:
        # Charger le run_id courant depuis le fichier local
        run_id = get_last_run_id()
        if not run_id:
            raise HTTPException(status_code=404, detail="Aucun run_id trouvé. Veuillez entraîner un modèle d'abord.")
        model = MLFlow_load_model(run_id, artifact_path)
        logger.info(f"Model loaded from MLflow run ID: {run_id}")
                
        # Charger le préprocesseur
        preprocessor = joblib.load(join('models','preprocessor.pkl'))
        # Colonnes attendues par le préprocesseur
        numerical_cols = ["age", "taille", "poids", "revenu_estime_mois"]
        categorical_cols = ["sexe", "sport_licence", "niveau_etude", "region", "smoker", "nationalité_francaise"]
        columns = numerical_cols + categorical_cols
        
        
        # Transformer les données d'entrée en DataFrame
        X_input = pd.DataFrame([payload.data], columns=columns)
        X_processed = preprocessor.transform(X_input)
        # Prédiction
        y_pred = model_predict(model, X_processed)
        
        predict = np.asarray(y_pred).squeeze().item()
        
        logger.info(f"Prediction made: {y_pred}, prediction value: {predict}")
        return {"prediction": float(predict)}
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class RetrainRequest(BaseModel):
    data_path: str  # Chemin du fichier CSV à utiliser comme nouvelle source de données
    from_existing_model: bool = True  # True: fine-tuning, False: nouveau modèle

class RetrainResponse(BaseModel):
    status: str
    run_id: str


@app.post("/retrain",
    response_model=RetrainResponse)
async def retrain(request: Request, payload: RetrainRequest):
    """
    Réentraîne le modèle à partir d'un fichier CSV fourni (data_path) et d'une option pour fine-tuning ou nouveau modèle.
    """
    
    settings = {
        "description": "retrain model with new data",
        "wanted_train_cycle": 3,  # nombre d'entraînements à effectuer 3 est le meilleur
        "epochs": 50,  
        "train_seed": 42,  
    }
    
    logger.info(f"Route '{request.url.path}' called for retraining with data_path={payload.data_path}, from_existing_model={payload.from_existing_model}")
    
    
    try:
        # Charger le dataset fourni
        df = pd.read_csv(payload.data_path)
        logger.info(f"Données chargées pour réentraînement: shape={df.shape}, colonnes={df.columns.tolist()}")

        run_id = get_last_run_id() if payload.from_existing_model else None
        
        # Mettre à jour la version des données avec seulement le nom du fichier
        settings["dataversion"] = os.path.basename(payload.data_path)
        for i in range(wanted_train_cycle):
            logger.info(f"Starting training iteration {i} of {wanted_train_cycle} (from_existing_model={payload.from_existing_model})")
            run_id = train_and_log_iterative(i, settings, run_id)
            
        # Mettre à jour le modèle de prédiction avec le dernier run_id
        # Sauvegarder le run_id et les scores dans un fichier local
        set_last_run_id(run_id)
        
        return {"status": "success", "run_id": run_id}
    
    except Exception as e:
        logger.error(f"Erreur lors du réentraînement: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
