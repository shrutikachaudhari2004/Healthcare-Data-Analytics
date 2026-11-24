
# ==================================================
# DAY 3: HEALTHCARE DATA EXPLORATORY DATA ANALYSIS
# ==================================================

from pathlib import Path
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "healthcare_dataset_cleaned.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "day3"
FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD DATASET
# ==================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Cleaned dataset not found. "
        "Complete Day 2 cleaning first."
    )

df = pd.read_csv(INPUT_FILE)

print("\n========== DAY 3: HEALTHCARE EDA ==========")


# ==================================================
# 3. BASIC DATASET INFORMATION
# ==================================================

print("\n1. DATASET OVERVIEW")

print("Total rows:", df.shape[0])
print("Total columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDuplicate records:", df.duplicated().sum())


# ==================================================
# 4. MISSING VALUE ANALYSIS
# ==================================================

print("\n2. MISSING VALUE ANALYSIS")

missing_report = pd.DataFrame({
    "column": df.columns,
    "missing_values": df.isnull().sum().values,
    "missing_percentage": (
        df.isnull().mean().values * 100
    ).round(2)
})

missing_report = missing_report.sort_values(
    by="missing_values",
    ascending=False
)

print(missing_report.to_string(index=False))

missing_report.to_csv(
    OUTPUT_DIR / "day3_missing_values.csv",
    index=False
)


# ==================================================
# 5. STATISTICAL ANALYSIS
# ==================================================

print("\n3. STATISTICAL ANALYSIS")

numeric_df = df.select_dtypes(include="number")

if not numeric_df.empty:

    statistical_summary = numeric_df.describe().T

    print(statistical_summary)

    statistical_summary.to_csv(
        OUTPUT_DIR / "day3_statistical_summary.csv"
    )

else:
    statistical_summary = pd.DataFrame()
    print("No numeric columns available.")


# ==================================================
# 6. AGE DISTRIBUTION
# ==================================================

print("\n4. AGE ANALYSIS")

if "age" in df.columns:

    age = pd.to_numeric(df["age"], errors="coerce").dropna()

    print("Average age:", round(age.mean(), 2))
    print("Median age:", round(age.median(), 2))
    print("Minimum age:", age.min())
    print("Maximum age:", age.max())

    if not age.empty:

        plt.figure(figsize=(10, 6))

        sns.histplot(
            age,
            bins=20,
            kde=True
        )

        plt.title("Healthcare Patient Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Number of Records")

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / "age_distribution.png",
            dpi=150
        )

        plt.close()


# ==================================================
# 7. GENDER ANALYSIS
# ==================================================

print("\n5. GENDER ANALYSIS")

if "gender" in df.columns:

    gender_counts = df["gender"].value_counts()

    print(gender_counts)

    plt.figure(figsize=(8, 5))

    gender_counts.plot(kind="bar")

    plt.title("Healthcare Records by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Number of Records")

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "gender_distribution.png",
        dpi=150
    )

    plt.close()


# ==================================================
# 8. MEDICAL CONDITION ANALYSIS
# ==================================================

print("\n6. MEDICAL CONDITION ANALYSIS")

if "medical_condition" in df.columns:

    condition_counts = (
        df["medical_condition"]
        .value_counts()
        .head(10)
    )

    print(condition_counts)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=condition_counts.values,
        y=condition_counts.index
    )

    plt.title("Top Medical Conditions")
    plt.xlabel("Number of Records")
    plt.ylabel("Medical Condition")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "medical_condition_analysis.png",
        dpi=150
    )

    plt.close()


# ==================================================
# 9. ADMISSION TYPE ANALYSIS
# ==================================================

print("\n7. ADMISSION TYPE ANALYSIS")

if "admission_type" in df.columns:

    admission_counts = df["admission_type"].value_counts()

    print(admission_counts)

    plt.figure(figsize=(8, 5))

    admission_counts.plot(kind="bar")

    plt.title("Healthcare Admission Types")
    plt.xlabel("Admission Type")
    plt.ylabel("Number of Records")

    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "admission_type_analysis.png",
        dpi=150
    )

    plt.close()


# ==================================================
# 10. BILLING AMOUNT ANALYSIS
# ==================================================

print("\n8. BILLING AMOUNT ANALYSIS")

if "billing_amount" in df.columns:

    billing = pd.to_numeric(
        df["billing_amount"],
        errors="coerce"
    ).dropna()

    print("Average billing:", round(billing.mean(), 2))
    print("Minimum billing:", round(billing.min(), 2))
    print("Maximum billing:", round(billing.max(), 2))
    print("Total billing:", round(billing.sum(), 2))

    if not billing.empty:

        plt.figure(figsize=(10, 6))

        sns.histplot(
            billing,
            bins=30,
            kde=True
        )

        plt.title("Healthcare Billing Amount Distribution")
        plt.xlabel("Billing Amount")
        plt.ylabel("Number of Records")

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / "billing_distribution.png",
            dpi=150
        )

        plt.close()

    if "medical_condition" in df.columns:

        billing_by_condition = (
            df.assign(billing_amount=pd.to_numeric(
                df["billing_amount"],
                errors="coerce"
            ))
            .groupby("medical_condition")["billing_amount"]
            .mean()
            .dropna()
            .sort_values(ascending=False)
        )

        print("\nAverage billing by medical condition:")
        print(billing_by_condition)

        plt.figure(figsize=(10, 6))

        billing_by_condition.plot(kind="bar")

        plt.title("Average Billing by Medical Condition")
        plt.xlabel("Medical Condition")
        plt.ylabel("Average Billing")

        plt.xticks(rotation=30)
        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / "billing_by_condition.png",
            dpi=150
        )

        plt.close()


# ==================================================
# 11. TEST RESULTS ANALYSIS
# ==================================================

print("\n9. TEST RESULTS ANALYSIS")

if "test_results" in df.columns:

    test_counts = df["test_results"].value_counts()

    print(test_counts)

    plt.figure(figsize=(8, 5))

    test_counts.plot(kind="bar")

    plt.title("Healthcare Test Results Distribution")
    plt.xlabel("Test Results")
    plt.ylabel("Number of Records")

    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "test_results_analysis.png",
        dpi=150
    )

    plt.close()


# ==================================================
# 12. MONTHLY ADMISSION TREND
# ==================================================

print("\n10. MONTHLY ADMISSION ANALYSIS")

if "date_of_admission" in df.columns:

    admission_dates = pd.to_datetime(
        df["date_of_admission"],
        errors="coerce"
    )

    monthly_admissions = (
        admission_dates.dropna()
        .dt.to_period("M")
        .value_counts()
        .sort_index()
    )

    monthly_admissions.index = (
        monthly_admissions.index.astype(str)
    )

    print(monthly_admissions)

    if not monthly_admissions.empty:

        plt.figure(figsize=(12, 6))

        monthly_admissions.plot(
            kind="line",
            marker="o"
        )

        plt.title("Monthly Healthcare Admission Trend")
        plt.xlabel("Month")
        plt.ylabel("Number of Records")

        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / "monthly_admission_trend.png",
            dpi=150
        )

        plt.close()


# ==================================================
# 13. CORRELATION ANALYSIS
# ==================================================

print("\n11. CORRELATION ANALYSIS")

correlation_data = df.select_dtypes(include="number")

if correlation_data.shape[1] >= 2:

    correlation_matrix = correlation_data.corr()

    print(correlation_matrix)

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Healthcare Data Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "correlation_heatmap.png",
        dpi=150
    )

    plt.close()


# ==================================================
# 14. GENERATE EDA SUMMARY
# ==================================================

summary = {
    "total_records": int(len(df)),
    "total_columns": int(len(df.columns)),
    "duplicate_records": int(df.duplicated().sum()),
    "total_missing_cells": int(df.isnull().sum().sum()),
    "numeric_columns": list(
        df.select_dtypes(include="number").columns
    ),
    "categorical_columns": list(
        df.select_dtypes(
            include=["object", "string"]
        ).columns
    )
}

if "age" in df.columns:

    age = pd.to_numeric(df["age"], errors="coerce")

    summary["average_age"] = (
        round(float(age.mean()), 2)
        if age.notna().any() else None
    )

if "billing_amount" in df.columns:

    billing = pd.to_numeric(
        df["billing_amount"],
        errors="coerce"
    )

    summary["average_billing"] = (
        round(float(billing.mean()), 2)
        if billing.notna().any() else None
    )

if "medical_condition" in df.columns:

    summary["top_medical_conditions"] = (
        df["medical_condition"]
        .value_counts()
        .head(5)
        .to_dict()
    )

summary_path = OUTPUT_DIR / "day3_eda_summary.json"

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(summary, file, indent=4)


# ==================================================
# 15. FINAL RESULT
# ==================================================

print("\n========== DAY 3 EDA COMPLETED ==========")

print("Total records:", len(df))
print("Total columns:", len(df.columns))

print("\nReports saved in:")
print(OUTPUT_DIR)

print("\nCharts saved in:")
print(FIGURE_DIR)

print("\n========== SUCCESS ==========")
