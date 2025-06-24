import pandas as pd
import joblib
import pytest
from fastapi.testclient import TestClient
from mlFlow_experiment import app

client = TestClient(app)

def test_predict_route():
    # Exemple de features (adapter selon le modèle attendu)
    features = [30, 175, 70, 2500, "M", "Oui", "Bac+2", "Ile-de-France", "Non", "Oui"]
    response = client.post("/predict", json={"data": features})
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert isinstance(result["prediction"], float)

def test_streamlit_features_to_dataframe():
    # Simule les features envoyées par Streamlit (doivent correspondre à l'ordre attendu)
    features = [30, 175, 70, 2500, 'M', 'Oui', 'Bac+2', 'Île-de-France', 'Non', 'Oui']
    numerical_cols = ["age", "taille", "poids", "revenu_estime_mois"]
    categorical_cols = ["sexe", "sport_licence", "niveau_etude", "region", "smoker", "nationalité_francaise"]
    columns = numerical_cols + categorical_cols
    X_input = pd.DataFrame([features], columns=columns)
    preprocessor = joblib.load("models/preprocessor.pkl")
    # Doit fonctionner sans erreur
    try:
        X_processed = preprocessor.transform(X_input)
    except Exception as e:
        pytest.fail(f"Erreur lors du transform du DataFrame Streamlit : {e}")
    assert X_processed.shape[0] == 1
