import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import numpy as np


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)


def run_train(data_path: str):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("homework-1")
    mlflow.autolog()
    with mlflow.start_run():
        mlflow.set_tag("model", "randomforestregressor")
        X_train, y_train = np.load(os.path.join(data_path, "train.pkl"), allow_pickle=True)
        X_val, y_val = np.load(os.path.join(data_path, "val.pkl"), allow_pickle=True)

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        print("trying to fit")
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        print(rmse)
        mlflow.log_metric("rmse", rmse)

if __name__ == '__main__':
    run_train()
