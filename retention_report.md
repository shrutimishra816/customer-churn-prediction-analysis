# Customer Retention Report — Stakeholder Summary

**Prepared from**: Random Forest churn classifier (84.7% test accuracy, ROC-AUC 0.92), trained on 55,000 customer records, with SHAP explainability applied to translate model internals into business drivers.

## Headline finding
Churn is not evenly distributed — it is concentrated in a small number of identifiable, addressable segments. The model's top three SHAP drivers are **contract type**, **monthly charge level**, and **tenure**, together accounting for the majority of predictive signal.

## What's driving churn

1. **Contract type is the single strongest lever.** Month-to-month customers churn at **58%**, versus **12.5%** for one-year and **3%** for two-year contracts. Converting even a fraction of month-to-month customers to annual contracts (via a discount or bundled incentive) is the highest-leverage retention action available.

2. **The first 6 months are the danger zone.** New customers churn at **65%**, decaying steadily to **19%** by month 49+. A structured onboarding or early-tenure check-in program targeting the 0–6 month cohort would intervene at the point of highest risk.

3. **Fiber customers churn ~2x more than DSL customers** (51% vs. 28%), despite paying a premium — likely reflecting higher price sensitivity or service expectations at that tier. Worth investigating fiber-specific service quality or pricing friction.

4. **Missing add-ons correlate with elevated risk.** Customers without online security or tech support add-ons show meaningfully higher churn — these products may function as retention tools, not just revenue lines.

5. **Electronic check payers churn more than automatic-payment customers**, consistent with lower switching friction / lower engagement — a candidate segment for auto-pay migration campaigns.

## Recommended actions, ranked by expected impact
| Priority | Action | Rationale |
|---|---|---|
| 1 | Incentivize contract upgrades for month-to-month customers | Largest single driver; ~4–19x churn multiplier vs. longer contracts |
| 2 | Launch a 0–6 month onboarding / retention touchpoint | Highest-risk tenure window |
| 3 | Audit fiber service experience & pricing | Unexplained 2x churn premium vs. DSL |
| 4 | Bundle security/support add-ons into standard plans | Correlated with materially lower churn |
| 5 | Migrate electronic-check payers to autopay | Lower-friction payment methods show lower churn |

## Model performance & confidence
- **Accuracy: 84.7%** | Precision: 80.0% | Recall: 76.0% | ROC-AUC: 0.92
- The model correctly identifies roughly 3 in 4 customers who go on to churn (recall), giving retention teams a workable target list rather than acting blind.
- Full confusion matrix and SHAP driver rankings are available in the interactive dashboard (`dashboard/index.html`).

*Methodology note: this report is generated from a synthetic dataset built to mirror the structure and relationships of a real telecom churn dataset, for portfolio/demonstration purposes. The pipeline (SQL exploration → feature engineering → Random Forest → SHAP) is production-ready and designed to be pointed at a live customer table.*
