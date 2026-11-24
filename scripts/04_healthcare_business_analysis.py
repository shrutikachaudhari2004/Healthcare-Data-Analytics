
# =========================================================
# DAY 4: HEALTHCARE BUSINESS ANALYSIS AND KPI DEVELOPMENT
# Technologies: Python, Pandas, NumPy, Matplotlib, Seaborn
# =========================================================

from pathlib import Path
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# 1. PROJECT PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "healthcare_dataset_cleaned.csv"
)

OUTPUT_DIR = ROOT / "reports" / "day4"
FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. LOAD DATASET
# =========================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\n========== DAY 4: HEALTHCARE BUSINESS ANALYSIS ==========")

print("\nTotal records:", len(df))
print("Total columns:", len(df.columns))

# Standardize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(r"[^a-z0-9]+", "_", regex=True)
    .str.strip("_")
)


# =========================================================
# 3. PREPARE ANALYSIS COLUMNS
# =========================================================

if "age" in df.columns:
    df["age"] = pd.to_numeric(
        df["age"], errors="coerce"
    )

if "billing_amount" in df.columns:
    df["billing_amount"] = pd.to_numeric(
        df["billing_amount"], errors="coerce"
    )

if "date_of_admission" in df.columns:
    df["date_of_admission"] = pd.to_datetime(
        df["date_of_admission"], errors="coerce"
    )

if "discharge_date" in df.columns:
    df["discharge_date"] = pd.to_datetime(
        df["discharge_date"], errors="coerce"
    )

# Calculate length of stay in days.
# Negative durations are invalid and excluded from this KPI.

if {
    "date_of_admission",
    "discharge_date"
}.issubset(df.columns):

    duration = (
        df["discharge_date"] -
        df["date_of_admission"]
    ).dt.total_seconds() / 86400

    negative_duration_count = int(
        (duration < 0).sum()
    )

    df["length_of_stay_days"] = duration.where(
        duration >= 0
    )

else:
    negative_duration_count = None
    df["length_of_stay_days"] = np.nan


# =========================================================
# 4. GENERATE OVERALL HEALTHCARE KPIs
# =========================================================

print("\n========== HEALTHCARE KPIs ==========")

kpis = {}

kpis["total_records"] = int(len(df))

kpis["unique_hospitals"] = (
    int(df["hospital"].nunique())
    if "hospital" in df.columns
    else None
)

kpis["average_age"] = (
    round(float(df["age"].mean()), 2)
    if "age" in df.columns and df["age"].notna().any()
    else None
)

kpis["average_billing_amount"] = (
    round(float(df["billing_amount"].mean()), 2)
    if "billing_amount" in df.columns
    and df["billing_amount"].notna().any()
    else None
)

kpis["total_recorded_billing_amount"] = (
    round(float(df["billing_amount"].sum()), 2)
    if "billing_amount" in df.columns
    and df["billing_amount"].notna().any()
    else None
)

kpis["average_length_of_stay_days"] = (
    round(float(df["length_of_stay_days"].mean()), 2)
    if df["length_of_stay_days"].notna().any()
    else None
)

kpis["median_length_of_stay_days"] = (
    round(float(df["length_of_stay_days"].median()), 2)
    if df["length_of_stay_days"].notna().any()
    else None
)

kpis["negative_date_durations_flagged"] = (
    negative_duration_count
)

if "medical_condition" in df.columns:
    kpis["unique_medical_conditions"] = int(
        df["medical_condition"].nunique()
    )

if "insurance_provider" in df.columns:
    kpis["unique_insurance_providers"] = int(
        df["insurance_provider"].nunique()
    )

if "admission_type" in df.columns:
    kpis["unique_admission_types"] = int(
        df["admission_type"].nunique()
    )

for key, value in kpis.items():
    print(f"{key}: {value}")


# =========================================================
# 5. CREATE REUSABLE GROUP ANALYSIS
# =========================================================

def analyze_category(data, column, output_name):

    if column not in data.columns:
        print(f"\nSkipping {column}: column not found.")
        return None

    aggregations = {
        "record_count": (column, "size")
    }

    if "billing_amount" in data.columns:
        aggregations["average_billing_amount"] = (
            "billing_amount", "mean"
        )

        aggregations["total_recorded_billing"] = (
            "billing_amount", "sum"
        )

    if data["length_of_stay_days"].notna().any():
        aggregations["average_stay_days"] = (
            "length_of_stay_days", "mean"
        )

    result = (
        data.groupby(column, dropna=False)
        .agg(**aggregations)
        .reset_index()
    )

    if "average_billing_amount" in result.columns:
        result["average_billing_amount"] = (
            result["average_billing_amount"].round(2)
        )

    if "average_stay_days" in result.columns:
        result["average_stay_days"] = (
            result["average_stay_days"].round(2)
        )

    result.to_csv(
        OUTPUT_DIR / output_name,
        index=False
    )

    return result


# =========================================================
# 6. HOSPITAL-WISE ANALYSIS
# =========================================================

print("\n========== HOSPITAL ANALYSIS ==========")

hospital_analysis = analyze_category(
    df,
    "hospital",
    "day4_hospital_analysis.csv"
)

if hospital_analysis is not None:

    hospital_analysis = hospital_analysis.sort_values(
        "record_count",
        ascending=False
    )

    print(hospital_analysis.head(10).to_string(index=False))

    top_hospitals = hospital_analysis.head(10)

    plt.figure(figsize=(11, 6))

    sns.barplot(
        data=top_hospitals,
        x="record_count",
        y="hospital"
    )

    plt.title("Top 10 Hospitals by Record Count")
    plt.xlabel("Number of Records")
    plt.ylabel("Hospital")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "top_hospitals.png",
        dpi=150
    )

    plt.close()


# =========================================================
# 7. INSURANCE PROVIDER ANALYSIS
# =========================================================

print("\n========== INSURANCE ANALYSIS ==========")

insurance_analysis = analyze_category(
    df,
    "insurance_provider",
    "day4_insurance_analysis.csv"
)

if insurance_analysis is not None:
    insurance_analysis = insurance_analysis.sort_values(
        "record_count",
        ascending=False
    )

    print(insurance_analysis.to_string(index=False))


# =========================================================
# 8. MEDICAL CONDITION ANALYSIS
# =========================================================

print("\n========== MEDICAL CONDITION ANALYSIS ==========")

condition_analysis = analyze_category(
    df,
    "medical_condition",
    "day4_condition_analysis.csv"
)

if condition_analysis is not None:

    condition_analysis = condition_analysis.sort_values(
        "record_count",
        ascending=False
    )

    print(condition_analysis.to_string(index=False))

    top_conditions = condition_analysis.head(10)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=top_conditions,
        x="record_count",
        y="medical_condition"
    )

    plt.title("Top Medical Conditions by Record Count")
    plt.xlabel("Number of Records")
    plt.ylabel("Medical Condition")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "medical_condition_business_analysis.png",
        dpi=150
    )

    plt.close()


# =========================================================
# 9. ADMISSION TYPE ANALYSIS
# =========================================================

print("\n========== ADMISSION TYPE ANALYSIS ==========")

admission_analysis = analyze_category(
    df,
    "admission_type",
    "day4_admission_analysis.csv"
)

if admission_analysis is not None:

    admission_analysis = admission_analysis.sort_values(
        "record_count",
        ascending=False
    )

    print(admission_analysis.to_string(index=False))

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=admission_analysis,
        x="admission_type",
        y="record_count"
    )

    plt.title("Admission Type Analysis")
    plt.xlabel("Admission Type")
    plt.ylabel("Number of Records")

    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "admission_type_business_analysis.png",
        dpi=150
    )

    plt.close()


# =========================================================
# 10. MEDICATION ANALYSIS
# =========================================================

print("\n========== MEDICATION ANALYSIS ==========")

medication_analysis = analyze_category(
    df,
    "medication",
    "day4_medication_analysis.csv"
)

if medication_analysis is not None:

    medication_analysis = medication_analysis.sort_values(
        "record_count",
        ascending=False
    )

    print(medication_analysis.head(10).to_string(index=False))


# =========================================================
# 11. MONTHLY ADMISSION ANALYSIS
# =========================================================

print("\n========== MONTHLY ADMISSION ANALYSIS ==========")

monthly_analysis = None

if "date_of_admission" in df.columns:

    monthly_data = df.dropna(
        subset=["date_of_admission"]
    ).copy()

    monthly_data["month"] = (
        monthly_data["date_of_admission"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly_analysis = (
        monthly_data.groupby("month")
        .agg(
            record_count=("month", "size"),
            average_billing=("billing_amount", "mean")
            if "billing_amount" in monthly_data.columns
            else ("month", "size")
        )
        .reset_index()
    )

    if "billing_amount" in monthly_data.columns:
        monthly_analysis["average_billing"] = (
            monthly_analysis["average_billing"].round(2)
        )
    else:
        monthly_analysis = monthly_analysis.drop(
            columns=["average_billing"]
        )

    monthly_analysis.to_csv(
        OUTPUT_DIR / "day4_monthly_analysis.csv",
        index=False
    )

    print(monthly_analysis.to_string(index=False))

    if not monthly_analysis.empty:

        plt.figure(figsize=(12, 6))

        sns.lineplot(
            data=monthly_analysis,
            x="month",
            y="record_count",
            marker="o"
        )

        plt.title("Monthly Healthcare Admission Records")
        plt.xlabel("Month")
        plt.ylabel("Number of Records")

        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / "monthly_healthcare_trend.png",
            dpi=150
        )

        plt.close()


# =========================================================
# 12. LENGTH OF STAY ANALYSIS
# =========================================================

print("\n========== LENGTH OF STAY ANALYSIS ==========")

if df["length_of_stay_days"].notna().any():

    valid_stays = df[
        "length_of_stay_days"
    ].dropna()

    stay_summary = pd.DataFrame({
        "metric": [
            "Average stay days",
            "Median stay days",
            "Minimum stay days",
            "Maximum stay days"
        ],
        "value": [
            round(float(valid_stays.mean()), 2),
            round(float(valid_stays.median()), 2),
            round(float(valid_stays.min()), 2),
            round(float(valid_stays.max()), 2)
        ]
    })

    stay_summary.to_csv(
        OUTPUT_DIR / "day4_length_of_stay.csv",
        index=False
    )

    print(stay_summary.to_string(index=False))

    plt.figure(figsize=(10, 6))

    sns.histplot(
        valid_stays,
        bins=20,
        kde=True
    )

    plt.title("Length of Hospital Stay Distribution")
    plt.xlabel("Length of Stay (Days)")
    plt.ylabel("Number of Records")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "length_of_stay_analysis.png",
        dpi=150
    )

    plt.close()

else:
    print("No valid admission and discharge dates available.")


# =========================================================
# 13. CREATE BUSINESS INSIGHTS
# =========================================================

print("\n========== BUSINESS INSIGHTS ==========")

insights = []

if condition_analysis is not None and not condition_analysis.empty:

    top_condition = condition_analysis.iloc[0]

    insights.append(
        "Most frequent medical condition category: "
        f"{top_condition['medical_condition']} "
        f"({int(top_condition['record_count'])} records)."
    )

if hospital_analysis is not None and not hospital_analysis.empty:

    top_hospital = hospital_analysis.iloc[0]

    insights.append(
        "Hospital with the highest record count: "
        f"{top_hospital['hospital']} "
        f"({int(top_hospital['record_count'])} records)."
    )

if admission_analysis is not None and not admission_analysis.empty:

    top_admission = admission_analysis.iloc[0]

    insights.append(
        "Most frequent admission category: "
        f"{top_admission['admission_type']} "
        f"({int(top_admission['record_count'])} records)."
    )

if insurance_analysis is not None and not insurance_analysis.empty:

    top_insurance = insurance_analysis.sort_values(
        "record_count",
        ascending=False
    ).iloc[0]

    insights.append(
        "Insurance provider with the most records: "
        f"{top_insurance['insurance_provider']} "
        f"({int(top_insurance['record_count'])} records)."
    )

if "average_length_of_stay_days" in kpis and kpis[
    "average_length_of_stay_days"
] is not None:

    insights.append(
        "Average recorded length of stay: "
        f"{kpis['average_length_of_stay_days']} days."
    )

if "average_billing_amount" in kpis and kpis[
    "average_billing_amount"
] is not None:

    insights.append(
        "Average recorded billing amount: "
        f"{kpis['average_billing_amount']}."
    )

# Save insights as a text file
INSIGHTS_FILE = OUTPUT_DIR / "day4_business_insights.txt"

with open(
    INSIGHTS_FILE,
    "w",
    encoding="utf-8"
) as file:

    for number, insight in enumerate(insights, start=1):
        file.write(f"{number}. {insight}\n")

for insight in insights:
    print("-", insight)


# =========================================================
# 14. SAVE KPI SUMMARY
# =========================================================

KPI_FILE = OUTPUT_DIR / "day4_kpi_summary.csv"

kpi_df = pd.DataFrame(
    list(kpis.items()),
    columns=["kpi_name", "kpi_value"]
)

kpi_df.to_csv(KPI_FILE, index=False)


# =========================================================
# 15. SAVE JSON SUMMARY
# =========================================================

json_summary = {
    "project": "Healthcare Data Analytics",
    "day": 4,
    "analysis": "Business Analysis and KPI Development",
    "kpis": kpis,
    "insights": insights,
    "limitations": [
        "Record counts are not necessarily unique patient counts.",
        "Billing amounts are recorded amounts, not confirmed payments or revenue.",
        "Hospital record volume does not measure hospital quality.",
        "These are descriptive dataset-level findings, not clinical recommendations."
    ]
}

JSON_FILE = OUTPUT_DIR / "day4_business_summary.json"

with open(
    JSON_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(json_summary, file, indent=4, default=str)


# =========================================================
# 16. FINAL OUTPUT
# =========================================================

print("\n========== DAY 4 COMPLETED ==========")

print("\nReports generated in:")
print(OUTPUT_DIR)

print("\nCharts generated in:")
print(FIGURE_DIR)

print("\nTotal business insights:", len(insights))

print("\n========== SUCCESS ==========")
