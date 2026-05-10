import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

# -------------------------------------------------------------------------
# Set random seed for reproducibility
# -------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Ensure the charts directory exists
os.makedirs("charts", exist_ok=True)

print("="*60)
print("STARTING STUDENT PERFORMANCE FACTORS ANALYTICS SYSTEM")
print("="*60)

# =========================================================================
# Task 1: Data Loading and Validation
# =========================================================================
print("\n--- Task 1: Data Loading and Validation ---")
csv_path = "StudentPerformanceFactors.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset '{csv_path}' not found in the workspace directory.")

df_raw = pd.read_csv(csv_path)
print(f"Dataset loaded successfully from: {csv_path}")
print(f"Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# Expected Columns validation
expected_cols = [
    "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
    "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores", "Motivation_Level",
    "Internet_Access", "Tutoring_Sessions", "Family_Income", "Teacher_Quality",
    "School_Type", "Peer_Influence", "Physical_Activity", "Learning_Disabilities",
    "Parental_Education_Level", "Distance_from_Home", "Gender", "Exam_Score"
]

missing_expected = [col for col in expected_cols if col not in df_raw.columns]
extra_cols = [col for col in df_raw.columns if col not in expected_cols]

if len(missing_expected) == 0:
    print("SUCCESS: All expected columns are present.")
else:
    print(f"WARNING: The following expected columns are missing: {missing_expected}")
    print("Continuing only with available compatible columns.")

# Report datatypes and exam score range
print("\nRaw Data Types:")
print(df_raw.dtypes)

exam_min = df_raw["Exam_Score"].min()
exam_max = df_raw["Exam_Score"].max()
print(f"\nExam_Score Range in raw data: [{exam_min}, {exam_max}]")

# =========================================================================
# Task 2: Data Preprocessing
# =========================================================================
print("\n--- Task 2: Data Preprocessing ---")

# Deep copy of raw df
df = df_raw.copy()

# Record missing values before cleaning
missing_before = df.isnull().sum()
print("\nMissing values BEFORE imputation:")
for col, val in missing_before.items():
    if val > 0:
        print(f"  - {col}: {val} missing values")

# Clean text columns: remove extra spaces and standardize category names
text_cols = df.select_dtypes(include=['object', 'str']).columns
for col in text_cols:
    df[col] = df[col].astype(str).str.strip()
    # If the column has missing values represented as nan or empty strings, map them back to NaN
    df[col] = df[col].replace({'nan': np.nan, 'None': np.nan, '': np.nan})

# Impute Missing Values:
# Numeric: use median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  - Imputed numeric column '{col}' with median: {median_val}")

# Categorical: use mode
categorical_cols = df.select_dtypes(include=['object', 'str']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"  - Imputed categorical column '{col}' with mode: {mode_val}")

# Record missing values after cleaning
missing_after = df.isnull().sum()
print("\nMissing values AFTER imputation (should be all 0):")
print(f"Total missing values in dataset: {missing_after.sum()}")

# Detect duplicate rows
duplicates_count = df.duplicated().sum()
print(f"\nDuplicate rows detected: {duplicates_count}")
# (We found 0 duplicates, which is great)

# Flag Exam_Score above 100 as a data quality issue (do not silently delete!)
above_100_records = df[df["Exam_Score"] > 100]
print(f"\nChecking for Exam_Scores above 100:")
if not above_100_records.empty:
    print(f"  - FOUND DATA QUALITY ISSUE: {len(above_100_records)} record(s) have Exam_Score > 100.")
    for idx, row in above_100_records.iterrows():
        print(f"    * Row Index {idx}: Student of Gender {row['Gender']} has score {row['Exam_Score']} "
              f"(Hours_Studied: {row['Hours_Studied']}, Attendance: {row['Attendance']}%).")
    print("  - NOTE: These records have NOT been deleted, but are flagged and documented.")
else:
    print("  - No Exam_Scores above 100 found.")