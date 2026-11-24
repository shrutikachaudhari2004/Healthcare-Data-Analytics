
from pathlib import Path
import json
import re
import pandas as pd

# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "healthcare_dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CLEANED_FILE = PROCESSED_DIR / "healthcare_dataset_cleaned.csv"


# ==========================================
# 2. LOAD RAW DATASET
# ==========================================

if not RAW_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {RAW_FILE}")

df = pd.read_csv(RAW_FILE)

print("\n========== DAY 2: DATA CLEANING ==========")

original_rows = len(df)
original_columns = len(df.columns)

print("\nOriginal rows:", original_rows)
print("Original columns:", original_columns)

# Preserve original missing-value counts
initial_missing = df.isna().sum().to_dict()


# ==========================================
# 3. STANDARDIZE COLUMN NAMES
# ==========================================

def clean_column_name(column):
    column = str(column).strip().lower()
    column = re.sub(r"[^a-z0-9]+", "_", column)
    return column.strip("_")


new_columns = [clean_column_name(col) for col in df.columns]

if len(new_columns) != len(set(new_columns)):
    raise ValueError(
        "Some column names become duplicates after standardization. "
        "Please inspect the original column names."
    )

df.columns = new_columns

print("\nStandardized column names:")
print(df.columns.tolist())


# ==========================================
# 4. CLEAN TEXT VALUES
# ==========================================

text_columns = df.select_dtypes(
    include=["object", "string"]
).columns

for col in text_columns:
    df[col] = df[col].astype("string").str.strip()

    # Convert empty strings into missing values
    df[col] = df[col].replace("", pd.NA)

    # Standardize common missing-value markers
    df[col] = df[col].replace(
        to_replace=r"(?i)^(na|n/a|null|none)$",
        value=pd.NA,
        regex=True
    )

print("\nText cleaning completed.")


# ==========================================
# 5. STANDARDIZE GENDER VALUES IF PRESENT
# ==========================================

if "gender" in df.columns:

    gender_mapping = {
        "m": "Male",
        "male": "Male",
        "f": "Female",
        "female": "Female"
    }

    def standardize_gender(value):
        if pd.isna(value):
            return pd.NA

        value = str(value).strip()

        return gender_mapping.get(value.casefold(), value)

    df["gender"] = df["gender"].map(standardize_gender)


# Standardize blood type if this column exists
if "blood_type" in df.columns:
    df["blood_type"] = df["blood_type"].str.upper()


# ==========================================
# 6. CONVERT NUMERIC COLUMNS
# ==========================================

numeric_columns = [
    col for col in df.columns
    if col == "age"
    or col.endswith(("_amount", "_cost", "_charges"))
]

numeric_conversion_failures = {}

for col in numeric_columns:

    before_conversion = df[col].notna()

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    failures = before_conversion & df[col].isna()

    numeric_conversion_failures[col] = int(failures.sum())

print("\nNumeric columns converted:", numeric_columns)


# ==========================================
# 7. CONVERT DATE COLUMNS
# ==========================================

date_columns = [
    col for col in df.columns
    if "date" in col
]

date_conversion_failures = {}

for col in date_columns:

    before_conversion = df[col].notna()

    df[col] = pd.to_datetime(
        df[col],
        errors="coerce"
    )

    failures = before_conversion & df[col].isna()

    date_conversion_failures[col] = int(failures.sum())

print("\nDate columns converted:", date_columns)


# ==========================================
# 8. CHECK MISSING VALUES
# ==========================================

missing_values = df.isnull().sum()

print("\n========== MISSING VALUES ==========")
print(missing_values)

print(
    "\nTotal missing cells:",
    int(missing_values.sum())
)


# ==========================================
# 9. REMOVE EXACT DUPLICATE ROWS
# ==========================================

duplicate_count = int(df.duplicated().sum())

print("\nDuplicate records found:", duplicate_count)

df = df.drop_duplicates().copy()

print("Duplicate records removed:", duplicate_count)


# ==========================================
# 10. VALIDATE SUSPICIOUS VALUES
# ==========================================

invalid_age_count = 0

if "age" in df.columns:

    invalid_age = (
        df["age"].notna()
        & ~df["age"].between(0, 120)
    )

    invalid_age_count = int(invalid_age.sum())

    print("\nAge values outside 0–120:", invalid_age_count)


invalid_date_order_count = 0

if {
    "date_of_admission",
    "discharge_date"
}.issubset(df.columns):

    invalid_date_order = (
        df["date_of_admission"].notna()
        & df["discharge_date"].notna()
        & (
            df["discharge_date"]
            < df["date_of_admission"]
        )
    )

    invalid_date_order_count = int(
        invalid_date_order.sum()
    )

    print(
        "\nDischarge before admission:",
        invalid_date_order_count
    )

# Suspicious records are flagged, not automatically deleted.


# ==========================================
# 11. SAVE CLEANED DATASET
# ==========================================

df.to_csv(CLEANED_FILE, index=False)

print("\nCleaned dataset saved at:")
print(CLEANED_FILE)


# ==========================================
# 12. CREATE DATA QUALITY REPORT
# ==========================================

quality_report = pd.DataFrame({
    "column": df.columns,
    "data_type": [
        str(df[col].dtype) for col in df.columns
    ],
    "missing_values": [
        int(df[col].isna().sum()) for col in df.columns
    ],
    "missing_percentage": [
        round(float(df[col].isna().mean() * 100), 2)
        for col in df.columns
    ],
    "unique_values": [
        int(df[col].nunique(dropna=True))
        for col in df.columns
    ]
})

REPORT_CSV = REPORT_DIR / "day2_data_quality_report.csv"

quality_report.to_csv(REPORT_CSV, index=False)

# Create a summary containing counts, not patient records.
summary = {
    "source_file": "data/raw/healthcare_dataset.csv",
    "cleaned_file": "data/processed/healthcare_dataset_cleaned.csv",
    "rows_before_cleaning": int(original_rows),
    "rows_after_cleaning": int(len(df)),
    "columns": int(len(df.columns)),
    "exact_duplicates_removed": duplicate_count,
    "initial_missing_values": {
        str(k): int(v) for k, v in initial_missing.items()
    },
    "final_missing_values": {
        str(k): int(v) for k, v in df.isna().sum().items()
    },
    "numeric_conversion_failures": numeric_conversion_failures,
    "date_conversion_failures": date_conversion_failures,
    "invalid_age_rows_flagged": invalid_age_count,
    "invalid_date_order_rows_flagged": invalid_date_order_count
}

REPORT_JSON = REPORT_DIR / "day2_cleaning_summary.json"

with open(REPORT_JSON, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)


# ==========================================
# 13. FINAL RESULTS
# ==========================================

print("\n========== CLEANING SUMMARY ==========")

print("Original rows:", original_rows)
print("Cleaned rows:", len(df))
print("Original columns:", original_columns)
print("Final columns:", len(df.columns))

print("Duplicates removed:", duplicate_count)
print("Missing cells remaining:", int(df.isna().sum().sum()))

print("\nFirst five cleaned records:")
print(df.head().to_string(index=False))

print("\nGenerated files:")
print(CLEANED_FILE)
print(REPORT_CSV)
print(REPORT_JSON)

print("\n========== DAY 2 COMPLETED ==========")
