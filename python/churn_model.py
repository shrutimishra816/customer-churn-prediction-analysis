"""
Customer Churn Prediction — model training, evaluation, and SHAP explainability.

Pipeline:
  1. Load + clean the raw telecom export (SQL-style aggregation checks included)
  2. Feature engineer categorical/numeric columns
  3. Train a Random Forest classifier
  4. Evaluate accuracy / precision / recall / F1 / ROC-AUC
  5. Run SHAP to explain global + per-feature drivers of churn
  6. Export a JSON payload consumed by dashboard/index.html
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "telecom_customers.csv"
OUT_PATH = ROOT / "dashboard" / "dashboard_data.json"

RANDOM_STATE = 42


def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    # Impute missing total_charges with tenure * monthly_charges estimate (SQL-equivalent: COALESCE)
    est = df["tenure_months"] * df["monthly_charges"]
    df["total_charges"] = df["total_charges"].fillna(est)
    return df


def engineer_features(df: pd.DataFrame):
    cat_cols = [
        "gender", "partner", "dependents", "contract_type", "internet_service",
        "multiple_lines", "online_security", "tech_support", "streaming_tv",
        "paperless_billing", "payment_method"
    ]
    num_cols = ["senior_citizen", "tenure_months", "monthly_charges", "total_charges"]

    encoders = {}
    enc_df = df.copy()
    for c in cat_cols:
        le = LabelEncoder()
        enc_df[c] = le.fit_transform(enc_df[c])
        encoders[c] = le

    feature_cols = cat_cols + num_cols
    X = enc_df[feature_cols]
    y = enc_df["churn"]
    return X, y, feature_cols, encoders


def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=16,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, preds) * 100, 2),
        "precision": round(precision_score(y_test, preds) * 100, 2),
        "recall": round(recall_score(y_test, preds) * 100, 2),
        "f1_score": round(f1_score(y_test, preds) * 100, 2),
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
    }
    cm = confusion_matrix(y_test, preds).tolist()
    return metrics, cm, preds, probs


def shap_analysis(model, X_train, X_test, feature_cols, sample_size=400):
    explainer = shap.TreeExplainer(model)
    sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=RANDOM_STATE)
    shap_values = explainer.shap_values(sample)
    # shap_values shape for binary classifier: (n_samples, n_features, 2) in recent SHAP,
    # or list[class0, class1] in older versions — handle both.
    if isinstance(shap_values, list):
        sv_churn = shap_values[1]
    elif shap_values.ndim == 3:
        sv_churn = shap_values[:, :, 1]
    else:
        sv_churn = shap_values

    mean_abs_shap = np.abs(sv_churn).mean(axis=0)
    importance = sorted(
        zip(feature_cols, mean_abs_shap), key=lambda x: x[1], reverse=True
    )
    top_drivers = [{"feature": f, "impact": round(float(v), 4)} for f, v in importance[:10]]
    return top_drivers


def readable_feature_name(name: str) -> str:
    mapping = {
        "contract_type": "Contract type",
        "tenure_months": "Tenure (months)",
        "monthly_charges": "Monthly charges",
        "total_charges": "Total charges",
        "internet_service": "Internet service",
        "payment_method": "Payment method",
        "online_security": "Online security add-on",
        "tech_support": "Tech support add-on",
        "paperless_billing": "Paperless billing",
        "senior_citizen": "Senior citizen",
        "streaming_tv": "Streaming TV",
        "multiple_lines": "Multiple lines",
        "partner": "Has partner",
        "dependents": "Has dependents",
        "gender": "Gender",
    }
    return mapping.get(name, name)


def main():
    df = load_and_clean()
    X, y, feature_cols, _ = engineer_features(df)
    model, X_train, X_test, y_train, y_test = train(X, y)
    metrics, cm, preds, probs = evaluate(model, X_test, y_test)
    top_drivers = shap_analysis(model, X_train, X_test, feature_cols)

    for d in top_drivers:
        d["label"] = readable_feature_name(d["feature"])

    # Churn rate by key business segments, for the "stakeholder retention report" panel
    seg_contract = df.groupby("contract_type")["churn"].mean().mul(100).round(2)
    seg_internet = df.groupby("internet_service")["churn"].mean().mul(100).round(2)
    tenure_bins = pd.cut(df["tenure_months"], bins=[-1, 6, 12, 24, 48, 72],
                         labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
    seg_tenure = df.groupby(tenure_bins)["churn"].mean().mul(100).round(2)

    payload = {
        "metrics": metrics,
        "confusion_matrix": cm,
        "n_records_analyzed": int(len(df)),
        "n_test_records": int(len(X_test)),
        "overall_churn_rate": round(float(df["churn"].mean() * 100), 2),
        "top_drivers": top_drivers,
        "churn_by_contract": [{"segment": k, "churn_rate": v} for k, v in seg_contract.items()],
        "churn_by_internet": [{"segment": k, "churn_rate": v} for k, v in seg_internet.items()],
        "churn_by_tenure": [{"segment": str(k), "churn_rate": v} for k, v in seg_tenure.items()],
    }

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Wrote dashboard payload -> {OUT_PATH}")


if __name__ == "__main__":
    main()
