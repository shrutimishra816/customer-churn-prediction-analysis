# Customer Churn Prediction & Analysis

**Python · Scikit-Learn · SQL · Interactive HTML Dashboard**

An explainable churn prediction system for a telecom customer base — from SQL-based exploration through a tuned Random Forest classifier to SHAP-driven business insight, delivered as a stakeholder-facing retention report and live dashboard. Also includes customer segmentation (clustering), lifetime-value prediction (regression), web scraping (BeautifulSoup + Scrapy), a SQLite-backed query layer, an EDA notebook, and a pytest suite with CI — covering the full data-to-decision pipeline, not just the model.

**[View the live dashboard →](https://shrutimishra816.github.io/customer-churn-prediction-analysis/)**

## Highlights

- 📉 **84.7% model accuracy** (ROC-AUC 0.92) predicting churn across **55,000+ telecom customer records**
- 🎯 **Top churn drivers identified via SHAP**: contract type, tenure, and monthly charges — consistent with, and quantifying, industry-known churn patterns
- 🧠 **Random Forest classifier** with class-aware evaluation (precision/recall/F1, confusion matrix) — not just accuracy
- 🧩 **K-Means customer segmentation** (clustering) — 4 behavior-based segments, from 15.2% to 55.8% churn rate
- 💰 **Customer lifetime value regression** — Random Forest Regressor, R² 0.93
- 🕷️ **Web scraping** — BeautifulSoup + Selenium fallback, and a separate Scrapy spider, both hitting a live, robots.txt-permitted target
- 🗄️ **SQL exploration layer** with CTEs and window functions, wired up to a real running SQLite database (not just reference `.sql`)
- 📓 **EDA notebook** with executed outputs
- ✅ **Pytest suite + GitHub Actions CI** running on every push
- 📋 **Stakeholder-facing retention report** translating model output into ranked, actionable recommendations

## What's in this repo

```
├── data/
│   └── generate_data.py             # Synthetic 55K+ telecom customer dataset generator
├── sql/
│   ├── churn_queries.sql            # CTEs + window functions: segment churn rates, risk flags, revenue-at-risk
│   └── run_queries_sqlite.py        # Loads the CSV into SQLite and runs churn_queries.sql against it
├── python/
│   ├── churn_model.py               # Feature engineering, Random Forest training, SHAP explainability
│   ├── generate_charts.py           # Matplotlib/Seaborn static charts (confusion matrix, drivers, segments)
│   ├── customer_segmentation.py     # K-Means clustering + elbow method + cluster profiling
│   └── ltv_regression.py            # Linear + Random Forest regression predicting customer LTV
├── notebooks/
│   └── churn_eda.ipynb              # Exploratory data analysis notebook (executed, outputs included)
├── dashboard/
│   ├── index.html                   # Interactive risk dashboard (Chart.js)
│   └── dashboard_data.json          # Precomputed model metrics + SHAP drivers
├── web_scraping/
│   ├── scrape_market_signal.py      # BeautifulSoup/Selenium scraper (requests-based)
│   └── market_scraper/              # Scrapy project — same task, framework-based version
├── tests/
│   └── test_churn_model.py          # Pytest unit tests (feature engineering + scraping parser)
├── .github/workflows/tests.yml      # CI: runs the test suite on every push
├── retention_report.md              # Stakeholder-facing business summary and recommendations
```

## How it works

1. **`data/generate_data.py`** builds a 55,000-row customer dataset with realistic relationships between contract type, tenure, pricing, service add-ons, and churn (mirroring the structure of the well-known IBM Telco Churn dataset, at 10x scale).
2. **`sql/churn_queries.sql`** explores churn at the segment level directly in SQL — churn rate by contract type and tenure bucket (using window functions for running totals and share-of-total), a rule-based high-risk flag, and revenue-at-risk quantification. **`sql/run_queries_sqlite.py`** loads the CSV into a real SQLite database and executes these queries against it, unmodified.
3. **`python/churn_model.py`**:
   - Cleans and feature-engineers the raw export (categorical encoding, missing-value imputation)
   - Trains a tuned `RandomForestClassifier` (300 trees, depth-limited to avoid overfitting)
   - Evaluates accuracy, precision, recall, F1, and ROC-AUC on a held-out stratified test split
   - Runs `shap.TreeExplainer` to compute per-feature mean absolute SHAP impact — the real driver ranking behind the dashboard and report
4. **`python/generate_charts.py`** renders the same model output as static Matplotlib/Seaborn PNGs (confusion matrix heatmap, SHAP driver bar chart, segment churn bar charts) for use outside the browser dashboard.
5. **`python/customer_segmentation.py`** clusters customers by tenure/spend behavior via K-Means (elbow method to choose *k*), profiling each segment's size, average spend, and churn rate.
6. **`python/ltv_regression.py`** engineers a customer lifetime value target and predicts it with Linear Regression and a Random Forest Regressor, evaluated with MAE/RMSE/R².
7. **`retention_report.md`** turns the SHAP rankings into a ranked action plan a retention team could actually execute against.
8. **`dashboard/index.html`** is a self-contained dashboard (Chart.js via CDN) showing the model scorecard, confusion matrix, SHAP driver bars, and segment-level churn rates.

## Run it yourself

```bash
pip install -r requirements.txt
python data/generate_data.py
python python/churn_model.py
python python/generate_charts.py
python python/customer_segmentation.py
python python/ltv_regression.py
python sql/run_queries_sqlite.py
python excel_reporting/churn_root_cause_audit_report.py
python web_scraping/scrape_market_signal.py
pytest tests/ -v
# then open dashboard/index.html or notebooks/churn_eda.ipynb
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

## Customer segmentation (clustering)

`python/customer_segmentation.py` groups customers with K-Means instead of
classifying them — a different question ("what kinds of customers do we
have?") than the churn model's ("will this customer leave?"):

- Elbow method (`charts/clustering_elbow.png`) to justify the chosen *k*
- 4 segments profiled by size, average tenure, average spend, and churn rate
- Highest-risk segment: short-tenure, high-monthly-spend customers at 55.8%
  churn — a clear, actionable retention target distinct from any single
  SHAP feature

## Customer lifetime value (regression)

`python/ltv_regression.py` predicts a continuous target (LTV) rather than
a class label, using the same customer features:

- LTV engineered as spend-to-date plus an expected-remaining-value term
  conditioned on churn status
- Linear Regression baseline vs. Random Forest Regressor — R² 0.84 vs. 0.93
- Actual-vs-predicted scatter plots for both models saved to `charts/`

## Web scraping

Two implementations of the same task, both hitting a **live**,
robots.txt-permitted target (each package's own PyPI project page — a
self-referential "dependency intelligence" report on this project's own
`requirements.txt`):

- **`web_scraping/scrape_market_signal.py`** — `requests` + `BeautifulSoup`,
  with retry/backoff and a `SeleniumFallbackScraper` class for pages that
  need JS rendering
- **`web_scraping/market_scraper/`** — a full **Scrapy** project (spider +
  settings), for cases where a crawling framework (concurrency, retries,
  pipelines, multi-page crawls) is the better fit than a one-off script

```bash
pip install requests beautifulsoup4 selenium scrapy
python web_scraping/scrape_market_signal.py
# or, the Scrapy version:
cd web_scraping/market_scraper && scrapy crawl pypi_deps -o ../market_signal_scrapy.csv
```

## SQL, wired to a real database

`sql/churn_queries.sql` is written against a `customers` table (MySQL/SQLite-
compatible CTEs and window functions). `sql/run_queries_sqlite.py` loads
`telecom_customers.csv` into a local SQLite database and runs every query
in that file against it — so the SQL layer is demonstrably executable, not
just reference text. The same query file runs unchanged against MySQL with
`mysql-connector-python` (see the `get_mysql_connection` stub in that script).

## Notebook

`notebooks/churn_eda.ipynb` is an executed EDA notebook — churn split,
churn rate by contract type, a tenure-vs-spend scatter colored by churn,
and a correlation heatmap — run before jumping into the classifier.

## Tests

`tests/test_churn_model.py` covers the feature-engineering pipeline
(`engineer_features`, `readable_feature_name`) and the scraping parser
(`parse_package_page`), including an edge case for missing HTML fields.
`.github/workflows/tests.yml` runs the suite on every push via GitHub
Actions.

```bash
pytest tests/ -v
```

## Tech stack
`Python` (pandas, scikit-learn, SHAP, openpyxl, matplotlib, seaborn, requests, BeautifulSoup, Selenium, Scrapy, pytest) · `SQL` (CTEs & window functions, SQLite) · `Chart.js` for the dashboard front end · GitHub Actions for CI.

---
*Note: this project uses a synthetic dataset engineered to mirror real-world telecom churn patterns (contract-driven risk, tenure decay, service-tier effects) for demonstration purposes. The pipeline is designed to plug into a live customer table with minimal changes.*
