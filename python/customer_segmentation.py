"""
python/customer_segmentation.py

Unsupervised customer segmentation via K-Means clustering — separate from
the supervised churn classifier, this groups customers by behavior
(tenure, monthly spend, total spend) so a retention team can target
segments rather than only individual high-risk customers.

Pipeline:
  1. Load the same telecom dataset used by the churn model
  2. Scale tenure / monthly_charges / total_charges
  3. Use the elbow method to pick a cluster count, then fit KMeans
  4. Profile each cluster (size, avg tenure/spend, churn rate)
  5. Save an elbow-curve chart + a 2D cluster scatter chart

Usage:
    python python/customer_segmentation.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "telecom_customers.csv"
CHARTS_DIR = ROOT / "charts"
OUT_JSON = ROOT / "python" / "segment_profiles.json"

RANDOM_STATE = 42
FEATURES = ["tenure_months", "monthly_charges", "total_charges"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    est = df["tenure_months"] * df["monthly_charges"]
    df["total_charges"] = df["total_charges"].fillna(est)
    return df


def elbow_curve(X_scaled, k_range=range(2, 9)):
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(list(k_range), inertias, marker="o", color="#2563eb")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method — Choosing k for Customer Segmentation")
    fig.tight_layout()
    CHARTS_DIR.mkdir(exist_ok=True)
    fig.savefig(CHARTS_DIR / "clustering_elbow.png", dpi=150)
    plt.close(fig)
    return inertias


def fit_kmeans(X_scaled, k=4):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def plot_clusters(df, labels, out_path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    scatter = ax.scatter(
        df["tenure_months"], df["monthly_charges"],
        c=labels, cmap="tab10", alpha=0.5, s=12,
    )
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Monthly charges")
    ax.set_title("Customer Segments (K-Means)")
    legend = ax.legend(*scatter.legend_elements(), title="Segment", loc="upper right")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def profile_segments(df, labels):
    df = df.copy()
    df["segment"] = labels
    profiles = []
    for seg_id, group in df.groupby("segment"):
        profiles.append({
            "segment": int(seg_id),
            "n_customers": int(len(group)),
            "avg_tenure_months": round(float(group["tenure_months"].mean()), 1),
            "avg_monthly_charges": round(float(group["monthly_charges"].mean()), 2),
            "avg_total_charges": round(float(group["total_charges"].mean()), 2),
            "churn_rate_pct": round(float(group["churn"].mean() * 100), 2),
        })
    return sorted(profiles, key=lambda p: p["churn_rate_pct"], reverse=True)


def main():
    df = load_data()
    X = df[FEATURES]
    X_scaled = StandardScaler().fit_transform(X)

    elbow_curve(X_scaled)
    km, labels = fit_kmeans(X_scaled, k=4)
    plot_clusters(df, labels, CHARTS_DIR / "customer_segments.png")

    profiles = profile_segments(df, labels)
    OUT_JSON.write_text(json.dumps(profiles, indent=2))

    print(json.dumps(profiles, indent=2))
    print(f"\nSaved cluster profiles -> {OUT_JSON}")
    print(f"Saved charts -> {CHARTS_DIR}/clustering_elbow.png, {CHARTS_DIR}/customer_segments.png")


if __name__ == "__main__":
    main()
