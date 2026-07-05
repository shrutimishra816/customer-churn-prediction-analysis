-- ============================================================
-- Customer Churn Analysis — Exploratory SQL Layer
-- Target: MySQL 8.0+ / SQLite
-- Table: customers (loaded from telecom_customers.csv)
-- ============================================================

-- ------------------------------------------------------------
-- 1. Overall churn rate + segment-level churn rate by contract type
-- ------------------------------------------------------------
SELECT
    contract_type,
    COUNT(*)                                            AS customers,
    SUM(churn)                                           AS churned,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2)              AS churn_rate_pct,
    ROUND(100.0 * SUM(churn) / SUM(SUM(churn)) OVER (), 2) AS pct_of_all_churn
FROM customers
GROUP BY contract_type
ORDER BY churn_rate_pct DESC;


-- ------------------------------------------------------------
-- 2. Churn rate by tenure bucket (window function: cumulative share)
-- ------------------------------------------------------------
WITH tenure_bucketed AS (
    SELECT
        CASE
            WHEN tenure_months <= 6 THEN '0-6'
            WHEN tenure_months <= 12 THEN '7-12'
            WHEN tenure_months <= 24 THEN '13-24'
            WHEN tenure_months <= 48 THEN '25-48'
            ELSE '49-72'
        END AS tenure_bucket,
        churn
    FROM customers
)
SELECT
    tenure_bucket,
    COUNT(*)                                AS customers,
    ROUND(100.0 * AVG(churn), 2)            AS churn_rate_pct,
    SUM(COUNT(*)) OVER (ORDER BY tenure_bucket) AS running_customer_count
FROM tenure_bucketed
GROUP BY tenure_bucket
ORDER BY tenure_bucket;


-- ------------------------------------------------------------
-- 3. High-risk customer flag: month-to-month + electronic check + low tenure
--    (business rule mirrors the model's top SHAP drivers)
-- ------------------------------------------------------------
SELECT
    customer_id,
    contract_type,
    payment_method,
    tenure_months,
    monthly_charges,
    churn,
    CASE
        WHEN contract_type = 'Month-to-month'
             AND payment_method = 'Electronic check'
             AND tenure_months < 12
        THEN 'High Risk'
        ELSE 'Standard'
    END AS risk_flag
FROM customers
ORDER BY tenure_months ASC;


-- ------------------------------------------------------------
-- 4. Revenue at risk: monthly charges tied up in currently-churned accounts
-- ------------------------------------------------------------
SELECT
    ROUND(SUM(monthly_charges), 2)                          AS monthly_revenue_from_churned,
    ROUND(SUM(monthly_charges) * 12, 2)                     AS annualized_revenue_at_risk,
    COUNT(*)                                                AS churned_customers
FROM customers
WHERE churn = 1;


-- ------------------------------------------------------------
-- 5. Add-on adoption vs. churn (RANK window function)
-- ------------------------------------------------------------
WITH addon_churn AS (
    SELECT
        online_security,
        tech_support,
        COUNT(*)                        AS customers,
        ROUND(100.0 * AVG(churn), 2)    AS churn_rate_pct
    FROM customers
    GROUP BY online_security, tech_support
)
SELECT
    *,
    RANK() OVER (ORDER BY churn_rate_pct DESC) AS risk_rank
FROM addon_churn
ORDER BY risk_rank;
