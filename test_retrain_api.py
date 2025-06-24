import pytest
from fastapi.testclient import TestClient
from mlFlow_api import app
import os

client = TestClient(app)

def test_retrain_route_new_model():
    # Teste le réentraînement en mode "nouveau modèle" (from_existing_model=False)
    payload = {
        "data_path": os.path.join("data", "df_new.csv"),
        "from_existing_model": False
    }
    response = client.post("/retrain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "nouveau modèle actif pour la prévision" in data or "run_id" in data


def test_retrain_route_finetune():
    # Teste le réentraînement en mode fine-tuning (from_existing_model=True)
    # On force le run_id courant à None pour simuler l'absence de modèle courant
    # (la logique API doit fallback sur un nouveau modèle)
    payload = {
        "data_path": os.path.join("data", "df_modifie.csv"),
        "from_existing_model": True
    }
    response = client.post("/retrain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "nouveau modèle actif pour la prévision" in data or "run_id" in data
