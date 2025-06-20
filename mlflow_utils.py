import mlflow
import mlflow.sklearn
import joblib
from os.path import join as join
from datetime import datetime
from models.models import create_nn_model, train_model
from modules.evaluate import evaluate_performance
from modules.print_draw import draw_loss, print_data
import pandas as pd

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input


## Base initialisation for Loguru and FastAPI
from myapp_base import setup_loguru
logger = setup_loguru("logs/mlflow_utils.log")


from alchemy_api import settings, artifact_path, wanted_train_cycle

today_str = datetime.now().strftime("%Y%m%d_%H%M")


def MLFlow_train_model(options, model, X, y, X_val=None, y_val=None, epochs=50, batch_size=32, verbose=0):
    # entrainnement du modèle
    model, hist = train_model(model, X, y, X_val, y_val, epochs, batch_size, verbose)
    
    # Options de sauvegarde model et cost picture
    step_base_name = options.get("step_base_name", f"model_{today_str}_ml_{options.get('step', 'default')}")
    if options.get("save_model", False):
        joblib.dump(model, join('models', f'{step_base_name}.pkl'))
        logger.info(f"Model saved as {step_base_name}.pkl")
    if options.get("save_cost", False):
        draw_loss(hist, join('figures',f'{step_base_name}.jpg'))
    return model, hist, step_base_name

def MLFlow_load_model(runId, artifactPath="linear_regression_model"):
    model_uri = f"runs:/{runId}/{artifactPath}"
    model = mlflow.sklearn.load_model(model_uri)
    return model

def MLFlow_make_prediction(model, X):
    preds = model.predict(X)
    return preds

def train_and_log_model(X_train, y_train, X_test, y_test, run_desc, model_id=None, run_idx=0, artifact_path="linear_regression_model"):
    """
    Entraîne un modèle, loggue dans MLflow, retourne le model_id.
    """
     # Charger le modèle du run précédent ou créer un nouveau modèle
    if model_id is not None:
        logger.info(f"Loading model from previous model_id: {model_id}")
        model = MLFlow_load_model(model_id, artifact_path)
    else:
        logger.info(f"No previous model_id, Trainning column : {X_train.shape[1]}")
        model = create_nn_model(X_train.shape[1])
        model_id = "None"  # Reset model_id if no previous model is loaded
    
    
    # Bascule des poids de models si existants dans le settings
    if settings.get("transfert_weights", {}).get("active", False):
        logger.info(f"Transferring weights from previous model_id: {settings['transfert_weights']['run_id']}")
        if not model_id or model_id == "None":
            logger.warning("No previous model_id provided for weight transfer.")
        else:
            # Transfert des poids du modèle existant vers le nouveau modèle
            MLFlow_Transmit_weights(model, settings['transfert_weights']['run_id'], artifact_path)
    
    step_base_name = f"model_{today_str}_{run_idx}_{model_id}"
    model, hist, step_base_name = MLFlow_train_model({
        "save_model": True, # should be False when tests are finished
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
        mlflow.log_param("data_version", settings.get("training_data", "df_data_all_cleaned.csv"))
        mlflow.log_param("random_state", settings.get("train_seed", 42))
        mlflow.log_param("previous_run_id", model_id if model_id else "None")
        mlflow.log_metric("mse", perf['MSE'])
        mlflow.log_metric("mae", perf['MAE'])
        mlflow.log_metric("r2", perf['R²'])
        mlflow.sklearn.log_model(model, artifact_path)
        logger.info(f"Run {run_idx + 1} terminé, run_id={run.info.run_id}")
        return run.info.run_id
    
    
    
def MLFlow_Transmit_weights(model_new, run_id, artifact_path="linear_regression_model"):
    """ 
    Transmet les poids d'un modèle existant à un nouveau modèle depuis MLflow.
    Args:
        model_new: Le modèle Keras dans lequel les poids seront transférés.
        run_id: L'ID du run MLflow d'où les poids seront extraits.
        artifact_path: Le chemin de l'artéfact dans MLflow où le modèle est stocké.
        Returns:
        model_uri: L'URI du modèle dans MLflow après le transfert des poids.
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    if not mlflow.get_artifact_uri(model_uri):
        logger.error(f"Model URI {model_uri} not found in MLflow.")
        return None
    
    
    logger.info(f"Transmitting weights from model at {model_uri} to new model.")
    
    
    
    model_old = MLFlow_load_model(model_uri, artifact_path)
    
    # Transfert des poids (hors couche d'entrée)
    for layer in model_new.layers:
        try:
            old_layer = model_old.get_layer(layer.name)
            # Vérifie la compatibilité des poids
            if all([w1.shape == w2.shape for w1, w2 in zip(layer.get_weights(), old_layer.get_weights())]):
                layer.set_weights(old_layer.get_weights())
                print(f"✅ Poids transférés pour : {layer.name}")
            else:
                print(f"⚠️ Forme incompatible pour : {layer.name}, poids non transférés")
        except ValueError:
            print(f"⛔ Nouvelle couche ou nom absent : {layer.name}")
    
    logger.info(f"Model weights transmitted to MLflow run {run_id} at {artifact_path}.")
    return model_uri