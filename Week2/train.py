import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

#%%
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

#%%
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

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run():
        mlflow.sklearn.autolog()

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred) #, squared=False)

        # mlflow.set_tag("developer", "yang")

        # mlflow.log_param("train-data-path", "./output/green_tripdata_2023-01.csv")
        # mlflow.log_param("valid-data-path", "./output/green_tripdata_2023-02.csv")

        # alpha = 0.1
        # mlflow.log_param("alpha", alpha)
        # lr = Lasso(alpha)
        # lr.fit(X_train, y_train)
        # y_pred = lr.predict(X_val)
        # rmse = mean_squared_error(y_val, y_pred, squared=False)

        # mlflow.log_metric("rmse", rmse)

        # mlflow.log_artifact(local_path="./models/lin_reg.bin", artifact_path="models_pickle")

if __name__ == '__main__':
    run_train()
