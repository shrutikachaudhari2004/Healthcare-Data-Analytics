
from pathlib import Path
import pandas as pd

# Locate the project directory
project_root = Path(__file__).resolve().parents[1]

# Load healthcare dataset
file_path = project_root / "data" / "raw" / "healthcare_dataset.csv"

if not file_path.exists():
    raise FileNotFoundError(f"Dataset not found: {file_path}")

df = pd.read_csv(file_path)

# 1. Dataset information
print("\n===== DATASET OVERVIEW =====")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# 2. Column names
print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

# 3. First five records
print("\n===== FIRST FIVE RECORDS =====")
print(df.head())

# 4. Data types
print("\n===== DATA TYPES =====")
print(df.dtypes)

# 5. Missing values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# 6. Duplicate records
print("\n===== DUPLICATE RECORDS =====")
print(df.duplicated().sum())

# 7. Statistical summary
print("\n===== STATISTICAL SUMMARY =====")
print(df.describe(include="all").transpose())

print("\n===== EXPLORATION COMPLETED =====")
