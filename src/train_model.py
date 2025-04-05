import os
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
import warnings

warnings.filterwarnings("ignore")


def load_data():
    load_dotenv()
    raw_data_path = os.getenv("RAW_DATA_PATH")
    df = pd.read_csv(raw_data_path)
    return df


def load_config():
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
    return config


def preprocess_data(df, features, target):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def train_model(X_train, y_train, model_params):
    model = LogisticRegression(**model_params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, target_names):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", acc)
    return acc


def main():

    config = load_config()
    features = config["features"]
    model_params = config["model_params"]
    description = config.get("mlflow", {}).get("description", "")

    df = load_data()
    target = 'target'

    X_train, X_test, y_train, y_test = preprocess_data(df, features, target)

#### ml flow tracking ####
    mlflow.set_experiment("wine-classification")
    with mlflow.start_run():
        model = train_model(X_train, y_train, model_params)
        acc = evaluate_model(model, X_test, y_test, target)
        mlflow.set_tag("mlflow.note.content", description)
        # Log params from config
        for param, value in model_params.items():
            mlflow.log_param(param, value)
            mlflow.log_param("features", features)

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    mlflow.log_artifact("config.yml")

if __name__ == "__main__":
    main()
