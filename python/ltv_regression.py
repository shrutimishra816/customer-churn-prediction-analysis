"""
python/ltv_regression.py

Regression model — predicts a customer's expected lifetime value (LTV),
a continuous target, complementing the classification (churn) and
clustering (segmentation) work already in this repo so all three core
ML task types (Regression / Classification / Clustering) are covered.

LTV here is engineered as a simple, defensible proxy:
    LTV = monthly_charges * expected_remaining_tenure_months
where expected_remaining_tenure_months uses churn as a rough survival
signal (churned customers assumed near end-of-life; retained customers
assumed to continue for a horizon beyond their current tenure). This
mirrors how a first-pass LTV model is often built before a full survival
analysis is justified.

Pipeline:
  1. Load + clean data (same as churn_model.py)
  2. Engineer the LTV target
  3. Train/compare Linear Regression and Random Forest Regressor
  4. Evaluate with MAE, RMSE, R²
  5. Plot actual vs. predicted LTV

Usage:
    python python/ltv_regression.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "telecom_customers.csv"
CHARTS_DIR = ROOT / "charts"
OUT_JSON = ROOT / "python" / "ltv_regression_metrics.json"

RANDOM_STATE = 42
RETENTION_HORIZON_MONTHS = 24  # assumed additional months for a retained customer


def load_and_engineer_target():
    df = pd.read_csv(DATA_PATH)
    est = df["tenure_months"] * df["monthly_charges"]
    df["total_charges"] = df["total_charges"].fillna(est)

    expected_remaining = np.where(
        df["churn"] == 1,
        1,  # churned: ~1 more billing cycle before they leave
        RETENTION_HORIZON_MONTHS,
    )
    df["ltv"] = df["total_charges"] + df["monthly_charges"] * expected_remaining
    return df


def build_features(df):
    cat_cols = [
        "gender", "partner", "dependents", "contract_type", "internet_service",
        "multiple_lines", "online_security", "tech_support", "streaming_tv",
        "paperless_billing", "payment_method",
    ]
    num_cols = ["senior_citizen", "tenure_months", "monthly_charges"]

    enc_df = df.copy()
    for c in cat_cols:
        enc_df[c] = LabelEncoder().fit_transform(enc_df[c])

    feature_cols = cat_cols + num_cols
    X = enc_df[feature_cols]
    y = enc_df["ltv"]
    return X, y, feature_cols


def evaluate(y_true, y_pred):
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def plot_actual_vs_predicted(y_test, preds, model_name, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, preds, alpha=0.3, s=10, color="#2563eb")
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, "--", color="#dc2626", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual LTV ($)")
    ax.set_ylabel("Predicted LTV ($)")
    ax.set_title(f"Actual vs. Predicted Customer LTV — {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    df = load_and_engineer_target()
    X, y, feature_cols = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    results = {}

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    results["linear_regression"] = evaluate(y_test, lr_preds)

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results["random_forest_regressor"] = evaluate(y_test, rf_preds)

    CHARTS_DIR.mkdir(exist_ok=True)
    plot_actual_vs_predicted(y_test, lr_preds, "Linear Regression", CHARTS_DIR / "ltv_linear_regression.png")
    plot_actual_vs_predicted(y_test, rf_preds, "Random Forest Regressor", CHARTS_DIR / "ltv_random_forest.png")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nSaved metrics -> {OUT_JSON}")
    print(f"Saved charts -> {CHARTS_DIR}/ltv_linear_regression.png, {CHARTS_DIR}/ltv_random_forest.png")


if __name__ == "__main__":
    main()
