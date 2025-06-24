import os
import mlflow
import pytest





def test_mlflow_basic():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    exp_name = "test_experiment"
    mlflow.set_experiment(exp_name)
    with mlflow.start_run():
        mlflow.log_param("param1", 5)
        mlflow.log_metric("metric1", 0.85)
    assert True


def test_mlflow_server_running():
    try:
        mlflow.server.start()
        assert True
    except Exception:
        assert False

def test_log_parameters():
    mlflow.start_run()
    mlflow.log_param("param1", 5)
    mlflow.end_run()
    assert mlflow.get_artifact_uri() is not None

def test_log_metrics():
    mlflow.start_run()
    mlflow.log_metric("metric1", 0.85)
    mlflow.end_run()
    assert mlflow.get_artifact_uri() is not None

def test_log_model():
    mlflow.start_run()
    mlflow.sklearn.log_model(model, "model")
    mlflow.end_run()
    assert mlflow.get_artifact_uri() is not None

def test_retrieve_logged_data():
    run_id = mlflow.active_run().info.run_id
    logged_params = mlflow.get_run(run_id).data.params
    assert "param1" in logged_params