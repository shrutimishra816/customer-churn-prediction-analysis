# Customer Churn Prediction & Analysis

**Python · Scikit-Learn · SQL · Interactive HTML Dashboard**

An explainable churn prediction system for a telecom customer base — from SQL-based exploration through a tuned Random Forest classifier to SHAP-driven business insight, delivered as a stakeholder-facing retention report and live dashboard.

**[View the live dashboard →](https://shrutimishra816.github.io/customer-churn-prediction-analysis/)**

## Highlights

- 📉 **84.7% model accuracy** (ROC-AUC 0.92) predicting churn across **55,000+ telecom customer records**
- 🎯 **Top churn drivers identified via SHAP**: contract type, tenure, and monthly charges — consistent with, and quantifying, industry-known churn patterns
- 🧠 **Random Forest classifier** with class-aware evaluation (precision/recall/F1, confusion matrix) — not just accuracy
- 📋 **Stakeholder-facing retention report** translating model output into ranked, actionable recommendations
- 🗄️ **SQL exploration layer** with CTEs and window functions for segment-level churn benchmarking

## What's in this repo

```
├── data/
│   └── generate_data.py         # Synthetic 55K+ telecom customer dataset generator
├── sql/
│   └── churn_queries.sql        # CTEs + window functions: segment churn rates, risk flags, revenue-at-risk
├── python/
│   └── churn_model.py           # Feature engineering, Random Forest training, SHAP explainability
├── dashboard/
│   ├── index.html               # Interactive risk dashboard (Chart.js)
│   └── dashboard_data.json      # Precomputed model metrics + SHAP drivers
├── retention_report.md          # Stakeholder-facing business summary and recommendations
```

## How it works

1. **`data/generate_data.py`** builds a 55,000-row customer dataset with realistic relationships between contract type, tenure, pricing, service add-ons, and churn (mirroring the structure of the well-known IBM Telco Churn dataset, at 10x scale).
2. **`sql/churn_queries.sql`** explores churn at the segment level directly in SQL — churn rate by contract type and tenure bucket (using window functions for running totals and share-of-total), a rule-based high-risk flag, and revenue-at-risk quantification.
3. **`python/churn_model.py`**:
   - Cleans and feature-engineers the raw export (categorical encoding, missing-value imputation)
   - Trains a tuned `RandomForestClassifier` (300 trees, depth-limited to avoid overfitting)
   - Evaluates accuracy, precision, recall, F1, and ROC-AUC on a held-out stratified test split
   - Runs `shap.TreeExplainer` to compute per-feature mean absolute SHAP impact — the real driver ranking behind the dashboard and report
4. **`retention_report.md`** turns those SHAP rankings into a ranked action plan a retention team could actually execute against.
5. **`dashboard/index.html`** is a self-contained dashboard (Chart.js via CDN) showing the model scorecard, confusion matrix, SHAP driver bars, and segment-level churn rates.

## Run it yourself

```bash
pip install pandas numpy scikit-learn shap openpyxl
python data/generate_data.py
python python/churn_model.py
python excel_reporting/churn_root_cause_audit_report.py
# then open dashboard/index.html in a browser
```

## Root-cause & compliance audit report (Excel)

`excel_reporting/churn_root_cause_audit_report.py` turns the SHAP driver
ranking into an audit-style **Excel** deliverable — the format an
operations/quality stakeholder actually reviews, not just a model
scorecard:

- **Summary** — base attrition rate, total churned, top root-cause segment
- **Root Cause Pareto** — churn ranked by segment (contract type, payment method, tenure, service tier), with cumulative % and a chart
- **Compliance Audit** — every segment vs. a target attrition threshold, Pass/Non-Compliant flagged, same logic a quality team applies to defect-rate audits by line or SKU
- **Customer Watchlist** — highest-value churned customers in the worst segment, ready for retention outreach

This sits alongside `retention_report.md` and the SHAP dashboard as a third,
more operational view of the same root-cause analysis.

## Tech stack
`Python` (pandas, scikit-learn, SHAP, openpyxl) · `SQL` (CTEs & window functions) · `Chart.js` for the dashboard front end.

---
*Note: this project uses a synthetic dataset engineered to mirror real-world telecom churn patterns (contract-driven risk, tenure decay, service-tier effects) for demonstration purposes. The pipeline is designed to plug into a live customer table with minimal changes.*
