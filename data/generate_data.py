"""
Synthetic telecom customer dataset generator for the Churn Prediction project.
Produces 55,000 customer records with realistic relationships between
contract type, tenure, monthly charges, and churn — mirroring the structure
of the classic IBM Telco Churn dataset but at 10x scale.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
N = 55_000

genders = rng.choice(["Male", "Female"], size=N)
senior = rng.choice([0, 1], size=N, p=[0.84, 0.16])
partner = rng.choice(["Yes", "No"], size=N, p=[0.48, 0.52])
dependents = rng.choice(["Yes", "No"], size=N, p=[0.30, 0.70])

tenure = rng.integers(0, 73, size=N)  # months

contract = rng.choice(
    ["Month-to-month", "One year", "Two year"], size=N, p=[0.55, 0.24, 0.21]
)
internet_service = rng.choice(["Fiber optic", "DSL", "No"], size=N, p=[0.44, 0.35, 0.21])
payment_method = rng.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    size=N, p=[0.34, 0.19, 0.24, 0.23]
)
paperless_billing = rng.choice(["Yes", "No"], size=N, p=[0.59, 0.41])

online_security = rng.choice(["Yes", "No", "No internet service"], size=N, p=[0.29, 0.50, 0.21])
tech_support = rng.choice(["Yes", "No", "No internet service"], size=N, p=[0.29, 0.50, 0.21])
streaming_tv = rng.choice(["Yes", "No", "No internet service"], size=N, p=[0.38, 0.41, 0.21])
multiple_lines = rng.choice(["Yes", "No"], size=N, p=[0.42, 0.58])

# Monthly charges: base + service add-ons, fiber costs more
base_charge = np.where(internet_service == "Fiber optic", 70, np.where(internet_service == "DSL", 45, 20))
addon_charge = (
    (online_security == "Yes").astype(int) * 6
    + (tech_support == "Yes").astype(int) * 6
    + (streaming_tv == "Yes").astype(int) * 9
    + (multiple_lines == "Yes").astype(int) * 8
)
noise = rng.normal(0, 6, size=N)
monthly_charges = np.clip(base_charge + addon_charge + noise, 18, 130).round(2)
total_charges = (monthly_charges * tenure + rng.normal(0, 20, size=N)).clip(min=0).round(2)

# ---- Churn probability model: the real "ground truth" signal the RF will learn ----
logit = (
    -3.6
    + np.where(contract == "Month-to-month", 3.0, np.where(contract == "One year", -0.6, -2.7))
    + np.where(tenure < 6, 2.6, np.where(tenure < 12, 1.2, np.where(tenure > 48, -1.9, 0.0)))
    + (monthly_charges - 65) * 0.046
    + np.where(internet_service == "Fiber optic", 0.85, 0.0)
    + np.where(payment_method == "Electronic check", 0.95, 0.0)
    + np.where(online_security == "No", 0.65, 0.0)
    + np.where(tech_support == "No", 0.65, 0.0)
    + np.where(paperless_billing == "Yes", 0.28, 0.0)
    + np.where(senior == 1, 0.32, 0.0)
    + np.where(partner == "No", 0.18, 0.0)
    + rng.normal(0, 0.03, size=N)  # irreducible noise so accuracy isn't trivially 100%
)
churn_prob = 1 / (1 + np.exp(-logit))
churn = (rng.random(N) < churn_prob).astype(int)

df = pd.DataFrame({
    "customer_id": [f"CUST-{100000+i}" for i in range(N)],
    "gender": genders,
    "senior_citizen": senior,
    "partner": partner,
    "dependents": dependents,
    "tenure_months": tenure,
    "contract_type": contract,
    "internet_service": internet_service,
    "multiple_lines": multiple_lines,
    "online_security": online_security,
    "tech_support": tech_support,
    "streaming_tv": streaming_tv,
    "paperless_billing": paperless_billing,
    "payment_method": payment_method,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "churn": churn,
})

# Light real-world messiness for the pipeline to demonstrate handling
dirty_idx = rng.choice(df.index, size=250, replace=False)
df.loc[dirty_idx, "total_charges"] = np.nan

df.to_csv("/home/claude/churn/data/telecom_customers.csv", index=False)
print(f"Generated {len(df):,} rows -> telecom_customers.csv")
print("Churn rate:", round(df["churn"].mean() * 100, 2), "%")
