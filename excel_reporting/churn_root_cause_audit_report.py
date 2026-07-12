"""
Customer Attrition Root-Cause & Compliance Audit Report
---------------------------------------------------------
The Random Forest + SHAP model (python/churn_model.py) already tells us
*which features* drive churn. This script turns that into the kind of
audit-style Excel report an operations/quality analyst would publish:
segment-level attrition rates, a root-cause Pareto ranking, and a
compliance-style flagging of segments that breach an acceptable-attrition
threshold — plus a customer-level watchlist log for follow-up.

Sheets:
    1. Summary          - headline KPIs
    2. Root Cause Pareto - churn count/rate by top-driver segment, ranked
    3. Compliance Audit  - every segment vs. a target threshold, Pass/Fail
    4. Customer Watchlist - highest-risk active customers for retention outreach

Run: python excel_reporting/churn_root_cause_audit_report.py
"""
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.chart import BarChart, Reference
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "telecom_customers.csv"
OUT_PATH = Path(__file__).parent / "churn_root_cause_audit_report.xlsx"

# Target: no single segment should have an attrition rate more than this
# many percentage points above the overall base rate. Anything above is
# flagged "Non-Compliant" for follow-up -- same logic a process-compliance
# audit applies to defect rates by line/shift/SKU.
THRESHOLD_DELTA_PP = 8.0

HEADER_FILL = PatternFill("solid", fgColor="232F3E")
HEADER_FONT = Font(color="FFFFFF", bold=True, size=11)
TITLE_FONT = Font(bold=True, size=14, color="232F3E")
THIN = Side(style="thin", color="D9D9D9")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
PASS_FILL = PatternFill("solid", fgColor="C6EFCE")
FAIL_FILL = PatternFill("solid", fgColor="F8696B")


def style_header(ws, row, ncols):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = BORDER


def autosize(ws, ncols, width=22):
    for c in range(1, ncols + 1):
        ws.column_dimensions[get_column_letter(c)].width = width


def write_df(ws, df, start_row=1, start_col=1):
    for j, col in enumerate(df.columns):
        ws.cell(row=start_row, column=start_col + j, value=col)
    style_header(ws, start_row, len(df.columns))
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        for j, val in enumerate(row):
            cell = ws.cell(row=start_row + i, column=start_col + j, value=val)
            cell.border = BORDER
    return start_row + len(df) + 1


def tenure_bucket(months):
    if months <= 6:
        return "0-6 mo"
    if months <= 12:
        return "7-12 mo"
    if months <= 24:
        return "13-24 mo"
    return "24+ mo"


def build_report():
    df = pd.read_csv(DATA_PATH)
    df["tenure_bucket"] = df["tenure_months"].apply(tenure_bucket)

    total = len(df)
    churned = int(df["churn"].sum())
    base_rate = churned / total * 100

    # --- Root cause pareto across the three strongest known drivers ---
    segments = []
    for dim, label in [("contract_type", "Contract Type"),
                        ("payment_method", "Payment Method"),
                        ("tenure_bucket", "Tenure"),
                        ("internet_service", "Internet Service"),
                        ("tech_support", "Tech Support")]:
        g = df.groupby(dim)["churn"].agg(["count", "sum"]).reset_index()
        g.columns = ["segment_value", "customers", "churned"]
        g["driver"] = label
        g["churn_rate_pct"] = (g["churned"] / g["customers"] * 100).round(1)
        segments.append(g[["driver", "segment_value", "customers", "churned", "churn_rate_pct"]])
    all_segments = pd.concat(segments, ignore_index=True)

    pareto = all_segments.sort_values("churned", ascending=False).head(15).reset_index(drop=True)
    pareto["pct_of_total_churn"] = (pareto["churned"] / churned * 100).round(1)
    pareto["cumulative_pct"] = pareto["pct_of_total_churn"].cumsum().round(1)

    # --- Compliance audit: every segment vs threshold ---
    audit = all_segments.copy()
    audit["delta_vs_base_pp"] = (audit["churn_rate_pct"] - base_rate).round(1)
    audit["status"] = audit["delta_vs_base_pp"].apply(
        lambda d: "Non-Compliant" if d > THRESHOLD_DELTA_PP else "Pass"
    )
    audit = audit.sort_values("delta_vs_base_pp", ascending=False).reset_index(drop=True)

    # --- Watchlist: churned customers in the worst segments, highest bill first ---
    worst_contract = pareto.loc[pareto["driver"] == "Contract Type", "segment_value"].iloc[0] \
        if (pareto["driver"] == "Contract Type").any() else df["contract_type"].mode()[0]
    watchlist = (
        df[(df["churn"] == 1) & (df["contract_type"] == worst_contract)]
        [["customer_id", "contract_type", "payment_method", "tenure_months",
          "monthly_charges", "total_charges"]]
        .sort_values("monthly_charges", ascending=False)
        .head(300)
    )

    wb = Workbook()

    # Summary
    ws = wb.active
    ws.title = "Summary"
    ws["B2"] = "Customer Attrition Root-Cause & Compliance Audit"
    ws["B2"].font = TITLE_FONT
    ws["B3"] = "Segment-level churn breakdown, root-cause ranking, and audit-style compliance flags"
    ws["B3"].font = Font(italic=True, color="595959")
    kpis = [
        ("Total Customers", f"{total:,}"),
        ("Total Churned", f"{churned:,}"),
        ("Base Attrition Rate", f"{base_rate:.1f}%"),
        ("Compliance Threshold", f"base rate + {THRESHOLD_DELTA_PP:.0f}pp"),
        ("Segments Flagged Non-Compliant", int((audit['status'] == 'Non-Compliant').sum())),
        ("Top Root-Cause Segment", f"{pareto.iloc[0]['driver']}: {pareto.iloc[0]['segment_value']}"),
    ]
    r = 5
    for label, val in kpis:
        ws.cell(row=r, column=2, value=label).font = Font(bold=True)
        ws.cell(row=r, column=3, value=val)
        ws.cell(row=r, column=2).fill = PatternFill("solid", fgColor="F2F2F2")
        r += 1
    autosize(ws, 4, width=36)

    # Root cause pareto
    ws2 = wb.create_sheet("Root Cause Pareto")
    ws2["A1"] = "Top Attrition Root-Cause Segments (Pareto)"
    ws2["A1"].font = TITLE_FONT
    next_row = write_df(ws2, pareto, start_row=3)
    autosize(ws2, len(pareto.columns), width=18)

    chart = BarChart()
    chart.title = "Churned Customers by Segment"
    chart.y_axis.title = "Churned Customers"
    data_ref = Reference(ws2, min_col=4, min_row=3, max_row=2 + len(pareto))
    cats_ref = Reference(ws2, min_col=2, min_row=4, max_row=3 + len(pareto))
    chart.add_data(data_ref, titles_from_data=True)
    chart.set_categories(cats_ref)
    chart.width = 24
    chart.height = 10
    ws2.add_chart(chart, f"A{next_row + 1}")

    # Compliance audit
    ws3 = wb.create_sheet("Compliance Audit")
    ws3["A1"] = f"Segment Compliance Audit (threshold: base + {THRESHOLD_DELTA_PP:.0f}pp)"
    ws3["A1"].font = TITLE_FONT
    write_df(ws3, audit, start_row=3)
    autosize(ws3, len(audit.columns), width=18)
    status_col = list(audit.columns).index("status") + 1
    for i in range(len(audit)):
        cell = ws3.cell(row=4 + i, column=status_col)
        cell.fill = FAIL_FILL if cell.value == "Non-Compliant" else PASS_FILL
        if cell.value == "Non-Compliant":
            cell.font = Font(color="FFFFFF", bold=True)

    # Watchlist
    ws4 = wb.create_sheet("Customer Watchlist")
    ws4["A1"] = f"Retention Watchlist — Churned, {worst_contract} contract, highest bill first"
    ws4["A1"].font = TITLE_FONT
    write_df(ws4, watchlist, start_row=3)
    autosize(ws4, len(watchlist.columns), width=18)

    wb.save(OUT_PATH)
    print(f"Saved report -> {OUT_PATH}")
    print(f"Base attrition rate: {base_rate:.1f}% | Non-compliant segments: {(audit['status'] == 'Non-Compliant').sum()}")
    print(f"Top root cause: {pareto.iloc[0]['driver']} = {pareto.iloc[0]['segment_value']} ({pareto.iloc[0]['churn_rate_pct']}% churn rate)")


if __name__ == "__main__":
    build_report()
