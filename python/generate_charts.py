"""
python/generate_charts.py

Static Matplotlib/Seaborn charts for the churn model, generated from the
same dashboard_data.json produced by churn_model.py.

The interactive dashboard (dashboard/index.html) already visualizes this
data via Chart.js — this script exists to demonstrate the Matplotlib
charting workflow directly, and to produce portable PNG assets (for a
slide deck, README, or PDF report) that don't need a browser to view.

Usage:
    python python/churn_model.py       # must run first, produces dashboard_data.json
    python python/generate_charts.py   # reads it, writes charts/*.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "dashboard" / "dashboard_data.json"
CHARTS_DIR = ROOT / "charts"

sns.set_theme(style="whitegrid")


def load_payload():
    return json.loads(DATA_PATH.read_text())


def plot_confusion_matrix(cm, out_path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Predicted: Stay", "Predicted: Churn"],
        yticklabels=["Actual: Stay", "Actual: Churn"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Random Forest Churn Model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_top_drivers(top_drivers, out_path):
    labels = [d["label"] for d in top_drivers][::-1]
    values = [d["impact"] for d in top_drivers][::-1]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(labels, values, color="#2563eb")
    ax.set_xlabel("Mean |SHAP value| (impact on churn prediction)")
    ax.set_title("Top Churn Drivers (SHAP)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_segment_churn(payload, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    segment_keys = ["churn_by_contract", "churn_by_internet", "churn_by_tenure"]
    titles = ["By Contract Type", "By Internet Service", "By Tenure Bucket"]

    for ax, key, title in zip(axes, segment_keys, titles):
        segs = payload[key]
        labels = [s["segment"] for s in segs]
        values = [s["churn_rate"] for s in segs]
        ax.bar(labels, values, color="#dc2626")
        ax.set_title(title)
        ax.set_ylabel("Churn rate (%)")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Churn Rate by Segment")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    payload = load_payload()
    CHARTS_DIR.mkdir(exist_ok=True)

    plot_confusion_matrix(payload["confusion_matrix"], CHARTS_DIR / "confusion_matrix.png")
    plot_top_drivers(payload["top_drivers"], CHARTS_DIR / "top_drivers.png")
    plot_segment_churn(payload, CHARTS_DIR / "segment_churn.png")

    print(f"Saved 3 charts to {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
