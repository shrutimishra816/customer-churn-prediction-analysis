"""
tests/test_churn_model.py

Unit tests for the data-cleaning and feature-engineering functions in
python/churn_model.py, and for the HTML-parsing logic in
web_scraping/scrape_market_signal.py — the two places bad input is most
likely to silently break the pipeline.

Run:
    pytest tests/ -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "web_scraping"))

from churn_model import engineer_features, readable_feature_name  # noqa: E402
from scrape_market_signal import parse_package_page  # noqa: E402


# ---------------------------------------------------------------------
# churn_model.py
# ---------------------------------------------------------------------

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "gender": ["Male", "Female", "Male"],
        "partner": ["Yes", "No", "Yes"],
        "dependents": ["No", "No", "Yes"],
        "contract_type": ["Month-to-month", "Two year", "One year"],
        "internet_service": ["Fiber optic", "DSL", "No"],
        "multiple_lines": ["Yes", "No", "No"],
        "online_security": ["No", "Yes", "No"],
        "tech_support": ["No", "Yes", "Yes"],
        "streaming_tv": ["Yes", "No", "No"],
        "paperless_billing": ["Yes", "No", "Yes"],
        "payment_method": ["Electronic check", "Mailed check", "Credit card (automatic)"],
        "senior_citizen": [0, 1, 0],
        "tenure_months": [2, 60, 24],
        "monthly_charges": [70.5, 45.0, 55.0],
        "total_charges": [141.0, 2700.0, 1320.0],
        "churn": [1, 0, 0],
    })


def test_engineer_features_shapes(sample_df):
    X, y, feature_cols, encoders = engineer_features(sample_df)

    assert len(X) == len(sample_df)
    assert len(y) == len(sample_df)
    assert set(feature_cols) == set(X.columns)
    # every categorical column should have a fitted LabelEncoder
    assert "contract_type" in encoders


def test_engineer_features_encodes_categoricals_as_numeric(sample_df):
    X, _, _, _ = engineer_features(sample_df)
    for col in X.columns:
        assert pd.api.types.is_numeric_dtype(X[col]), f"{col} was not encoded to numeric"


def test_engineer_features_target_matches_churn_column(sample_df):
    _, y, _, _ = engineer_features(sample_df)
    assert list(y) == list(sample_df["churn"])


@pytest.mark.parametrize("raw,expected", [
    ("contract_type", "Contract type"),
    ("tenure_months", "Tenure (months)"),
    ("monthly_charges", "Monthly charges"),
])
def test_readable_feature_name_known_columns(raw, expected):
    assert readable_feature_name(raw) == expected


def test_readable_feature_name_unknown_column_passthrough():
    assert readable_feature_name("some_new_column") == "some_new_column"


# ---------------------------------------------------------------------
# web_scraping/scrape_market_signal.py
# ---------------------------------------------------------------------

SAMPLE_PYPI_HTML = """
<html>
  <h1 class="package-header__name">pandas 3.0.3</h1>
  <p class="package-description__summary">
    Powerful data structures for data analysis
  </p>
  <a href="/user/jorisvandenbossche/">jorisvandenbossche</a>
</html>
"""


def test_parse_package_page_extracts_expected_fields():
    record = parse_package_page(SAMPLE_PYPI_HTML, "pandas")
    assert record.package == "pandas"
    assert record.latest_version == "3.0.3"
    assert "data structures" in record.summary
    assert record.author == "jorisvandenbossche"


def test_parse_package_page_handles_missing_summary_gracefully():
    html = '<html><h1 class="package-header__name">numpy 2.5.1</h1></html>'
    record = parse_package_page(html, "numpy")
    assert record.latest_version == "2.5.1"
    assert record.summary == ""
    assert record.author == "unknown"
