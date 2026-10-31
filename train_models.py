"""
BITS Pilani WILP - M.Tech (AIML) Machine Learning Assignment 2
Dataset: Kaggle Telco Customer Churn (7,043 instances, 19 features)
Models: Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM
"""

import os
import ssl
import json
import joblib
import certifi
import warnings
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

warnings.filterwarnings("ignore")

RANDOM_SEED = 20261108
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

ENCODING_MAPS = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "MultipleLines": {"No phone service": 0, "No": 1, "Yes": 2},
    "InternetService": {"No": 0, "DSL": 1, "Fiber optic": 2},
    "OnlineSecurity": {"No internet service": 0, "No": 1, "Yes": 2},
    "OnlineBackup": {"No internet service": 0, "No": 1, "Yes": 2},
    "DeviceProtection": {"No internet service": 0, "No": 1, "Yes": 2},
    "TechSupport": {"No internet service": 0, "No": 1, "Yes": 2},
    "StreamingTV": {"No internet service": 0, "No": 1, "Yes": 2},
    "StreamingMovies": {"No internet service": 0, "No": 1, "Yes": 2},
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "PaperlessBilling": {"No": 0, "Yes": 1},
    "PaymentMethod": {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3
    }
}


def load_and_preprocess_telco_data() -> pd.DataFrame:
    print(f"[INFO] Downloading Telco Customer Churn dataset...")
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(DATA_URL, context=ssl_context) as response:
        df = pd.read_csv(response)

    df.to_csv("telco_churn_raw.csv", index=False)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    for col, mapping in ENCODING_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)

    feature_cols = [c for c in df.columns if c != "Churn"]
    df = df[feature_cols + ["Churn"]]

    print(f"[SUCCESS] Preprocessed Dataset Shape: {df.shape[0]} instances, {len(feature_cols)} features.")
    return df


def calculate_six_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "AUC": round(float(roc_auc_score(y_true, y_prob)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4)
    }


def main():
    print("=" * 76)
    print("BITS PILANI M.TECH (AIML) - TELCO CHURN TRAINING & EVALUATION PIPELINE")
    print("=" * 76)

    df = load_and_preprocess_telco_data()
    target_col = "Churn"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )

    test_df = X_test.copy()
    test_df[target_col] = y_test.values
    test_df.to_csv("test_data.csv", index=False)
    print(f"[ARTIFACT] Saved test_data.csv ({test_df.shape[0]} rows, {test_df.shape[1]} columns)")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    classifiers = {
        "Logistic Regression": (
            "logistic_regression.joblib",
            LogisticRegression(C=0.8, penalty="l2", solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
        ),
        "Decision Tree": (
            "decision_tree.joblib",
            DecisionTreeClassifier(criterion="gini", max_depth=6, min_samples_split=20, min_samples_leaf=10, random_state=RANDOM_SEED)
        ),
        "KNN": (
            "knn_classifier.joblib",
            KNeighborsClassifier(n_neighbors=15, weights="uniform", metric="minkowski", p=2)
        ),
        "Naive Bayes": (
            "naive_bayes.joblib",
            GaussianNB(var_smoothing=1e-8)
        ),
        "Support Vector Machine": (
            "svm_classifier.joblib",
            SVC(C=1.2, kernel="rbf", gamma="scale", probability=True, random_state=RANDOM_SEED)
        )
    }

    evaluation_summary = {}

    print("\nTraining & Evaluating All 5 Classifiers on Holdout Test Set (N = 1,409)...")
    print("-" * 76)

    for model_name, (filename, model_obj) in classifiers.items():
        model_obj.fit(X_train_scaled, y_train)
        y_pred = model_obj.predict(X_test_scaled)
        y_prob = model_obj.predict_proba(X_test_scaled)[:, 1]

        metrics = calculate_six_metrics(y_test.values, y_pred, y_prob)
        evaluation_summary[model_name] = metrics

        joblib.dump(model_obj, os.path.join(MODEL_DIR, filename))
        print(
            f"✔ {model_name:<24} | Acc: {metrics['Accuracy']:.4f} | AUC: {metrics['AUC']:.4f} | "
            f"Prec: {metrics['Precision']:.4f} | Rec: {metrics['Recall']:.4f} | "
            f"F1: {metrics['F1']:.4f} | MCC: {metrics['MCC']:.4f}"
        )

    metadata = {
        "dataset_name": "Kaggle Telco Customer Churn Dataset",
        "dataset_url": "https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
        "feature_columns": feature_cols,
        "target_column": target_col,
        "n_total_instances": int(df.shape[0]),
        "n_train_instances": int(X_train.shape[0]),
        "n_test_instances": int(X_test.shape[0]),
        "n_features": int(len(feature_cols)),
        "encoding_maps": ENCODING_MAPS,
        "baseline_metrics": evaluation_summary
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)



if __name__ == "__main__":
    main()
