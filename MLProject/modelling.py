# MLProject/modelling.py
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import dagshub
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    # Memuat dataset
    # Contoh sederhana: kita asumsikan dataset ada di direktori yang sama
    # Dalam implementasi nyata, sesuaikan dengan lokasi dataset Anda.
    X_train = pd.read_csv('../preprocessed_data/X_train_preprocessed.csv')
    X_test = pd.read_csv('../preprocessed_data/X_test_preprocessed.csv')
    y_train = pd.read_csv('../preprocessed_data/y_train_preprocessed.csv').values.ravel()
    y_test = pd.read_csv('../preprocessed_data/y_test_preprocessed.csv').values.ravel()

    # Konfigurasi DagsHub untuk menyimpan hasil secara online
    dagshub.init(repo_owner='DimasTawaqqal', repo_name='bankmarketing-mlflow', mlflow=True)
    mlflow.set_tracking_uri('https://dagshub.com/DimasTawaqqal/bankmarketing-mlflow.mlflow')

    with mlflow.start_run(run_name="CI_Run") as run:
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

        model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        mlflow.sklearn.log_model(model, "model")
        print(f"Model trained with accuracy: {accuracy}")

if __name__ == "__main__":
    main()