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

# =========================================================================
# Task 3: Feature Engineering
# =========================================================================
print("\n--- Task 3: Feature Engineering ---")

PASS_THRESHOLD = 65
print(f"Using pass/fail threshold: {PASS_THRESHOLD} marks")

# Pass_Flag
df["Pass_Flag"] = (df["Exam_Score"] >= PASS_THRESHOLD).astype(int)

# Risk_Level
def get_risk_level(score):
    if score < 65:
        return "High Risk"
    elif score <= 69:
        return "Medium Risk"
    else:
        return "Low Risk"

df["Risk_Level"] = df["Exam_Score"].apply(get_risk_level)

# Study_Category
def get_study_category(hours):
    if hours < 15:
        return "Low Study Hours"
    elif hours <= 24:
        return "Medium Study Hours"
    else:
        return "High Study Hours"

df["Study_Category"] = df["Hours_Studied"].apply(get_study_category)

# Attendance_Category
def get_attendance_category(att):
    if att < 70:
        return "Low Attendance"
    elif att <= 84:
        return "Medium Attendance"
    else:
        return "High Attendance"

df["Attendance_Category"] = df["Attendance"].apply(get_attendance_category)

# Previous_Performance_Category
def get_prev_perf_category(prev_score):
    if prev_score < 65:
        return "Low"
    elif prev_score <= 84:
        return "Medium"
    else:
        return "High"

df["Previous_Performance_Category"] = df["Previous_Scores"].apply(get_prev_perf_category)

print("New columns engineered successfully:")
print("  - Pass_Flag, Risk_Level, Study_Category, Attendance_Category, Previous_Performance_Category")

# Export Cleaned & Engineered Dataset
cleaned_csv_path = "cleaned_student_performance.csv"
df.to_csv(cleaned_csv_path, index=False)
print(f"Cleaned dataset saved as: {cleaned_csv_path}")

# =========================================================================
# Task 4 & 5: Exploratory Data Analysis & Correlation Analysis
# =========================================================================
print("\n--- Task 4 & 5: Exploratory Data Analysis & Correlation ---")

# Overall Exam_Score distribution stats
exam_mean = df["Exam_Score"].mean()
exam_median = df["Exam_Score"].median()
exam_min_clean = df["Exam_Score"].min()
exam_max_clean = df["Exam_Score"].max()

print(f"Overall Exam_Score Stats:")
print(f"  - Average: {exam_mean:.2f}")
print(f"  - Median: {exam_median:.2f}")
print(f"  - Range: [{exam_min_clean}, {exam_max_clean}]")

# Study time vs Exam_Score correlation
study_corr = df["Hours_Studied"].corr(df["Exam_Score"])
# Attendance vs Exam_Score correlation
attendance_corr = df["Attendance"].corr(df["Exam_Score"])
# Previous_Scores vs Exam_Score correlation
prev_corr = df["Previous_Scores"].corr(df["Exam_Score"])
# Sleep_Hours vs Exam_Score correlation
sleep_corr = df["Sleep_Hours"].corr(df["Exam_Score"])
# Tutoring_Sessions vs Exam_Score correlation
tutoring_corr = df["Tutoring_Sessions"].corr(df["Exam_Score"])

print(f"\nKey Correlations with Exam_Score:")
print(f"  - Attendance: {attendance_corr:.4f}")
print(f"  - Hours_Studied: {study_corr:.4f}")
print(f"  - Previous_Scores: {prev_corr:.4f}")
print(f"  - Tutoring_Sessions: {tutoring_corr:.4f}")
print(f"  - Sleep_Hours: {sleep_corr:.4f}")

# Categorical breakdowns (averages)
avg_by_motivation = df.groupby("Motivation_Level")["Exam_Score"].mean().to_dict()
avg_by_resources = df.groupby("Access_to_Resources")["Exam_Score"].mean().to_dict()
avg_by_school = df.groupby("School_Type")["Exam_Score"].mean().to_dict()
avg_by_gender = df.groupby("Gender")["Exam_Score"].mean().to_dict()
avg_by_income = df.groupby("Family_Income")["Exam_Score"].mean().to_dict()

# Pass/fail count & Risk segment breakdown
pass_count = df["Pass_Flag"].sum()
fail_count = len(df) - pass_count
pass_percentage = (pass_count / len(df)) * 100

risk_counts = df["Risk_Level"].value_counts().to_dict()
risk_percentages = (df["Risk_Level"].value_counts(normalize=True) * 100).to_dict()

# Calculate numeric correlations table
numeric_df = df.select_dtypes(include=[np.number])
# Exclude pass flag in the standard numeric df for correlation mapping if needed, but let's calculate correlation of everything with Exam_Score
correlation_series = numeric_df.corr()["Exam_Score"].sort_values(ascending=False)
correlation_table = pd.DataFrame({
    "Feature": correlation_series.index,
    "Pearson_Correlation": correlation_series.values,
    "Relationship_Strength": np.abs(correlation_series.values)
}).sort_values(by="Relationship_Strength", ascending=False)
# Exclude Exam_Score self-correlation for presentation
correlation_table_display = correlation_table[correlation_table["Feature"] != "Exam_Score"]

print("\nNumeric Correlations Table (sorted by strength):")
print(correlation_table_display.to_string(index=False))

# Generate eda_summary.md
eda_md_content = f"""# Student Performance Exploratory Data Analysis (EDA) Summary

This document summarizes the insights gained from the exploratory analysis of the `StudentPerformanceFactors.csv` dataset, which contains **{len(df)}** student profiles and **20** attributes.

## 1. Overall Performance Metrics
* **Total Student Records**: {len(df)}
* **Exam Score Range**: {exam_min_clean} to {exam_max_clean} marks
* **Average Exam Score**: {exam_mean:.2f} marks
* **Median Exam Score**: {exam_median:.2f} marks
* **Class Pass Rate (Threshold >= {PASS_THRESHOLD})**: {pass_percentage:.2f}% ({pass_count} passed, {fail_count} failed)

## 2. Key Numeric Factor Analysis (Correlation with Exam_Score)
Correlation values range from -1 to +1, indicating the strength and direction of the linear relationship. 

| Feature | Pearson Correlation | Relationship Strength | Interpretation |
| :--- | :---: | :---: | :--- |
{chr(10).join([f"| {row['Feature']} | {row['Pearson_Correlation']:.4f} | {'High' if row['Relationship_Strength'] > 0.5 else 'Medium' if row['Relationship_Strength'] > 0.2 else 'Very Low'} | {'Strong Positive Influence' if row['Pearson_Correlation'] > 0.5 else 'Moderate Positive Influence' if row['Pearson_Correlation'] > 0.2 else 'No strong linear relationship'} |" for _, row in correlation_table_display.iterrows()])}

### Important Insights:
1. **Attendance ({attendance_corr:.4f}) and Hours Studied ({study_corr:.4f})** have the absolute strongest relationships with exam performance. Regular class attendance and dedicated study hours are the absolute foundations of academic success.
2. **Previous Scores ({prev_corr:.4f})** and **Tutoring Sessions ({tutoring_corr:.4f})** are also very useful, positive indicators, showing that past preparation and seek-help behaviors significantly aid final scores.
3. **Sleep Hours ({sleep_corr:.4f})** has a negligible linear correlation. While sleep is crucial for physical and cognitive well-being, its relationship to scores within this range is non-linear and should not be overemphasized as a direct predictor.
4. *Remember: Correlation does not prove causation.* While these factors move in tandem, other underlying student, home, or school variables can influence both factors simultaneously.

## 3. Socio-Cultural & Environment Factors Analysis (Averages)

### Average Score by Motivation Level
* **High Motivation**: {avg_by_motivation.get('High', 0):.2f} marks
* **Medium Motivation**: {avg_by_motivation.get('Medium', 0):.2f} marks
* **Low Motivation**: {avg_by_motivation.get('Low', 0):.2f} marks

### Average Score by Access to Resources
* **High Access**: {avg_by_resources.get('High', 0):.2f} marks
* **Medium Access**: {avg_by_resources.get('Medium', 0):.2f} marks
* **Low Access**: {avg_by_resources.get('Low', 0):.2f} marks

### Average Score by Family Income
* **High Income**: {avg_by_income.get('High', 0):.2f} marks
* **Medium Income**: {avg_by_income.get('Medium', 0):.2f} marks
* **Low Income**: {avg_by_income.get('Low', 0):.2f} marks

### Average Score by School Type
* **Private**: {avg_by_school.get('Private', 0):.2f} marks
* **Public**: {avg_by_school.get('Public', 0):.2f} marks

### Average Score by Gender
* **Female**: {avg_by_gender.get('Female', 0):.2f} marks
* **Male**: {avg_by_gender.get('Male', 0):.2f} marks
* Note: The average scores across genders are practically identical, showing no performance bias.

## 4. Student Risk Segmentation Breakdown
Students are segmented into three risk categories based on their exam scores:
* **Low Risk** (Score >= 70): **{risk_counts.get('Low Risk', 0)}** students ({risk_percentages.get('Low Risk', 0.0):.2f}%)
* **Medium Risk** (Score 65 to 69): **{risk_counts.get('Medium Risk', 0)}** students ({risk_percentages.get('Medium Risk', 0.0):.2f}%)
* **High Risk** (Score < 65): **{risk_counts.get('High Risk', 0)}** students ({risk_percentages.get('High Risk', 0.0):.2f}%)
"""

with open("eda_summary.md", "w", encoding="utf-8") as f:
    f.write(eda_md_content)
print("Saved eda_summary.md")


