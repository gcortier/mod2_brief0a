import luigi
import pandas as pd
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class RetrainModel(luigi.Task):
    date = luigi.DateParameter()


    def output(self):
        return luigi.LocalTarget(f"models/model_{self.date}.txt")


    def run(self):
        # préparation des données 
        # entrainement du modèle
        with mlflow.start_run():
            mlflow.log_metric("mse", mse)
            mlflow.sklearn.log_model(model, "model")


        with self.output().open("w") as f:
            f.write(f"Model trained on {self.date} with mse={mse}\n")
