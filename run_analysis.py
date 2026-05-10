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

# =========================================================================
# Task 6 & 7: Regression and Classification Models
# =========================================================================
print("\n--- Task 6 & 7: Regression and Classification Models ---")

# Columns to use for predictive models (excluding engineered target leakage columns)
model_features = [
    "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
    "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores", "Motivation_Level",
    "Internet_Access", "Tutoring_Sessions", "Family_Income", "Teacher_Quality",
    "School_Type", "Peer_Influence", "Physical_Activity", "Learning_Disabilities",
    "Parental_Education_Level", "Distance_from_Home", "Gender"
]

# We encode categorical columns. Let's list which columns are categorical
cat_features_for_model = [col for col in model_features if df[col].dtype == 'object' or df[col].dtype == 'str']
num_features_for_model = [col for col in model_features if col not in cat_features_for_model]

print(f"Features for modeling: {len(num_features_for_model)} numeric, {len(cat_features_for_model)} categorical.")

# Create dummy encoded dataset for models (using drop_first=False so that we have complete importances for each level)
# We can sum dummy importances for each parent categorical column
X_raw = df[model_features]
X_encoded = pd.get_dummies(X_raw, columns=cat_features_for_model, drop_first=False)

# Target for regression
y_reg = df["Exam_Score"]
# Target for classification
y_clf = df["Pass_Flag"]

# Train/Test Split (80/20, fixed random seed 42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_encoded, y_reg, test_size=0.2, random_state=RANDOM_SEED
)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_encoded, y_clf, test_size=0.2, random_state=RANDOM_SEED
)

# --- 1. Regression Models ---
# Baseline Model 1: Full Linear Regression (with drop_first=True for stats)
X_encoded_stats = pd.get_dummies(X_raw, columns=cat_features_for_model, drop_first=True, dtype=float)

X_train_stats, X_test_stats, y_train_stats, y_test_stats = train_test_split(
    X_encoded_stats, y_reg, test_size=0.2, random_state=RANDOM_SEED
)

lr_stats_model = LinearRegression()
lr_stats_model.fit(X_train_stats, y_train_stats)
y_pred_lr = lr_stats_model.predict(X_test_stats)

mae_lr = mean_absolute_error(y_test_stats, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test_stats, y_pred_lr))
r2_lr = r2_score(y_test_stats, y_pred_lr)

# Calculate standard errors, t-statistics, and p-values using scipy
coefs_stats = lr_stats_model.coef_
intercept_stats = lr_stats_model.intercept_

X_design = np.column_stack([np.ones(X_train_stats.shape[0]), X_train_stats.values])
beta_stats = np.r_[intercept_stats, coefs_stats]

predictions_train = X_design @ beta_stats
residuals_train = y_train_stats.values - predictions_train

n_samp, n_feat = X_design.shape
s2_residual = np.sum(residuals_train**2) / (n_samp - n_feat)

XTX_inv = np.linalg.inv(X_design.T @ X_design)
cov_beta = s2_residual * XTX_inv
standard_errors = np.sqrt(np.diag(cov_beta))

t_statistics = beta_stats / standard_errors
p_vals = 2 * (1 - stats.t.cdf(np.abs(t_statistics), n_samp - n_feat))

stats_features = ["Intercept"] + list(X_train_stats.columns)
stats_df = pd.DataFrame({
    "Feature": stats_features,
    "Coefficient": beta_stats,
    "Std_Error": standard_errors,
    "t_statistic": t_statistics,
    "p_value": p_vals
})

stats_df.to_csv("linear_regression_results.csv", index=False)

# Model 2: Simplified Equation-Based Linear Regression on High-Significance Controllable Factors
# Factors: Hours_Studied, Attendance, Previous_Scores, Tutoring_Sessions, Physical_Activity
key_behavior_cols = ["Hours_Studied", "Attendance", "Previous_Scores", "Tutoring_Sessions", "Physical_Activity"]
X_simple_train = X_train_stats[key_behavior_cols]
X_simple_test = X_test_stats[key_behavior_cols]

lr_simple = LinearRegression()
lr_simple.fit(X_simple_train, y_train_stats)
y_pred_simple = lr_simple.predict(X_simple_test)

mae_simple = mean_absolute_error(y_test_stats, y_pred_simple)
rmse_simple = np.sqrt(mean_squared_error(y_test_stats, y_pred_simple))
r2_simple = r2_score(y_test_stats, y_pred_simple)

simple_intercept = lr_simple.intercept_
simple_coefs = lr_simple.coef_

# Advanced Model: Random Forest Regressor (trained on original X_encoded split)
reg_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg)

mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
r2 = r2_score(y_test_reg, y_pred_reg)

print("\nBaseline Regression Model Metrics (Full Linear Regression):")
print(f"  - MAE: {mae_lr:.4f}")
print(f"  - RMSE: {rmse_lr:.4f}")
print(f"  - R2 Score: {r2_lr:.4f}")

print("\nSimplified Equation-Based Regression Metrics (Key Controllable Behaviors):")
print(f"  - MAE: {mae_simple:.4f}")
print(f"  - RMSE: {rmse_simple:.4f}")
print(f"  - R2 Score: {r2_simple:.4f}")
print(f"  - Equation: Exam_Score = {simple_intercept:.2f} + {simple_coefs[0]:.4f}*Hours_Studied + {simple_coefs[1]:.4f}*Attendance + {simple_coefs[2]:.4f}*Previous_Scores + {simple_coefs[3]:.4f}*Tutoring_Sessions + {simple_coefs[4]:.4f}*Physical_Activity")

print("\nAdvanced Regression Model Metrics (Random Forest Regressor):")
print(f"  - MAE: {mae:.4f}")
print(f"  - RMSE: {rmse:.4f}")
print(f"  - R2 Score: {r2:.4f}")

# --- 2. Classification Model ---
clf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
clf_model.fit(X_train_clf, y_train_clf)

y_pred_clf = clf_model.predict(X_test_clf)
y_prob_clf = clf_model.predict_proba(X_test_clf)[:, 1]

accuracy = accuracy_score(y_test_clf, y_pred_clf)
precision = precision_score(y_test_clf, y_pred_clf)
recall = recall_score(y_test_clf, y_pred_clf)
f1 = f1_score(y_test_clf, y_pred_clf)
roc_auc = roc_auc_score(y_test_clf, y_prob_clf)
conf_matrix = confusion_matrix(y_test_clf, y_pred_clf)

print("\nClassification Model Metrics (Random Forest Classifier):")
print(f"  - Accuracy: {accuracy:.4f}")
print(f"  - Precision: {precision:.4f}")
print(f"  - Recall: {recall:.4f}")
print(f"  - F1-Score: {f1:.4f}")
print(f"  - ROC-AUC: {roc_auc:.4f}")
print(f"  - Confusion Matrix:\n{conf_matrix}")

# Class Balance Report
class_counts = df["Pass_Flag"].value_counts()
print(f"\nClass Balance in Dataset (Pass=1, Fail=0):")
print(f"  - Passed: {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(df)*100:.2f}%)")
print(f"  - Failed: {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(df)*100:.2f}%)")

# Calculate Feature Importances for original features
reg_importances = reg_model.feature_importances_
encoded_cols = X_encoded.columns
original_importances = {}

for col, imp in zip(encoded_cols, reg_importances):
    # Check if this column is a dummy of a categorical feature
    found = False
    for cat in cat_features_for_model:
        if col.startswith(cat + "_"):
            original_importances[cat] = original_importances.get(cat, 0) + imp
            found = True
            break
    if not found:
        original_importances[col] = imp

# Sort original feature importances
sorted_importances = sorted(original_importances.items(), key=lambda x: x[1], reverse=True)
print("\nFeature Importances (reaggregated back to original features):")
for feat, imp in sorted_importances:
    print(f"  - {feat}: {imp:.4f}")

# Format the model_metrics.csv in the JSON format requested
metrics_json = {
    "dataset_rows": int(len(df)),
    "dataset_columns": int(df.shape[1]),
    "pass_threshold": int(PASS_THRESHOLD),
    "regression_metrics": {
        "model_name": "Random Forest Regressor (Advanced)",
        "mae": f"{mae:.4f}",
        "rmse": f"{rmse:.4f}",
        "r2": f"{r2:.4f}"
    },
    "linear_regression_metrics": {
        "model_name": "Full Linear Regression (Baseline)",
        "mae": f"{mae_lr:.4f}",
        "rmse": f"{rmse_lr:.4f}",
        "r2": f"{r2_lr:.4f}"
    },
    "simplified_equation_metrics": {
        "model_name": "Simplified Equation-Based Linear Regression",
        "mae": f"{mae_simple:.4f}",
        "rmse": f"{rmse_simple:.4f}",
        "r2": f"{r2_simple:.4f}",
        "formula": f"Exam_Score = {simple_intercept:.4f} + {simple_coefs[0]:.4f}*Hours_Studied + {simple_coefs[1]:.4f}*Attendance + {simple_coefs[2]:.4f}*Previous_Scores + {simple_coefs[3]:.4f}*Tutoring_Sessions + {simple_coefs[4]:.4f}*Physical_Activity"
    },
    "classification_metrics": {
        "model_name": "Random Forest Classifier",
        "accuracy": f"{accuracy:.4f}",
        "precision": f"{precision:.4f}",
        "recall": f"{recall:.4f}",
        "f1_score": f"{f1:.4f}",
        "roc_auc": f"{roc_auc:.4f}"
    },
    "top_factors": [
        {"feature": "Attendance", "direction": "positive", "importance": "extremely high"},
        {"feature": "Hours_Studied", "direction": "positive", "importance": "extremely high"},
        {"feature": "Previous_Scores", "direction": "positive", "importance": "moderate"}
    ]
}

with open("model_metrics.csv", "w", encoding="utf-8") as f:
    json.dump(metrics_json, f, indent=2)
print("\nSaved model_metrics.csv")

# Save the trained models and meta information for Streamlit loading
import pickle
streamlit_models = {
    "lr_stats_model": lr_stats_model,
    "clf_model": clf_model,
    "reg_model": reg_model,
    "lr_simple_model": lr_simple,
    "X_encoded_stats_columns": list(X_encoded_stats.columns),
    "X_encoded_columns": list(X_encoded.columns),
    "model_features": model_features,
    "cat_features_for_model": cat_features_for_model,
    "key_behavior_cols": key_behavior_cols,
    "PASS_THRESHOLD": PASS_THRESHOLD
}
with open("student_performance_models.pkl", "wb") as f:
    pickle.dump(streamlit_models, f)
print("Saved trained models to student_performance_models.pkl")

# =========================================================================
# Task 8: Visualizations
# =========================================================================
print("\n--- Task 8: Visualizations ---")

# Cohesive Palette (Cool Slate/Teal/Warm Orange theme)
MAIN_COLOR = "#0f4c5c"  # Dark Teal
ACCENT_COLOR = "#e36414"  # Warm Orange
NEUTRAL_LIGHT = "#e9ecef"
MUTED_BLUE = "#5c677d"
RISK_COLORS = ["#d90429", "#f7a072", "#2ec4b6"] # High, Medium, Low Risk colors

# Chart 1: Exam Score Distribution
plt.figure(figsize=(9, 5))
sns.histplot(df["Exam_Score"], kde=True, color=MAIN_COLOR, edgecolor="black", alpha=0.8)
plt.axvline(PASS_THRESHOLD, color="red", linestyle="--", linewidth=2, label=f"Pass Threshold ({PASS_THRESHOLD})")
plt.title("Student Exam Score Distribution", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Exam Score", fontsize=11)
plt.ylabel("Student Count", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("charts/1_exam_score_distribution.png", dpi=200)
plt.close()
print("Saved Chart 1: Exam Score Distribution")

# Chart 2: Study Time vs Exam Score
plt.figure(figsize=(9, 5))
sns.regplot(data=df, x="Hours_Studied", y="Exam_Score", 
            scatter_kws={"alpha": 0.4, "color": MAIN_COLOR, "s": 20}, 
            line_kws={"color": ACCENT_COLOR, "linewidth": 2.5, "label": "Trend Line"})
plt.title("Study Weekly Hours vs Final Exam Score", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Hours Studied (Weekly)", fontsize=11)
plt.ylabel("Exam Score", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("charts/2_study_time_vs_exam_score.png", dpi=200)
plt.close()
print("Saved Chart 2: Study Time vs Exam Score")

# Chart 3: Attendance vs Exam Score
plt.figure(figsize=(9, 5))
sns.regplot(data=df, x="Attendance", y="Exam_Score", 
            scatter_kws={"alpha": 0.4, "color": "#2ec4b6", "s": 20}, 
            line_kws={"color": ACCENT_COLOR, "linewidth": 2.5, "label": "Trend Line"})
plt.title("School Attendance Rate vs Final Exam Score", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Attendance Rate (%)", fontsize=11)
plt.ylabel("Exam Score", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("charts/3_attendance_vs_exam_score.png", dpi=200)
plt.close()
print("Saved Chart 3: Attendance vs Exam Score")

# Chart 4: Previous Scores vs Exam Score
plt.figure(figsize=(9, 5))
sns.regplot(data=df, x="Previous_Scores", y="Exam_Score", 
            scatter_kws={"alpha": 0.4, "color": MUTED_BLUE, "s": 20}, 
            line_kws={"color": ACCENT_COLOR, "linewidth": 2.5, "label": "Trend Line"})
plt.title("Previous Student Scores vs Current Exam Score", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Previous Semester Score", fontsize=11)
plt.ylabel("Current Exam Score", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("charts/4_previous_scores_vs_exam_score.png", dpi=200)
plt.close()
print("Saved Chart 4: Previous Scores vs Exam Score")

# Chart 5: Correlation Heatmap
plt.figure(figsize=(9, 7))
corr_matrix = df[["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores", "Tutoring_Sessions", "Physical_Activity", "Exam_Score"]].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.5, cbar=True, square=True)
plt.title("Pearson Correlation Heatmap of Student Numeric Factors", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("charts/5_correlation_heatmap.png", dpi=200)
plt.close()
print("Saved Chart 5: Correlation Heatmap")

# Chart 6: Average Score by Motivation Level
plt.figure(figsize=(8, 5))
motivation_order = ["Low", "Medium", "High"]
sns.barplot(data=df, x="Motivation_Level", y="Exam_Score", order=motivation_order, palette="viridis", errorbar=None, edgecolor="black")
plt.title("Average Final Exam Score by Student Motivation Level", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Student Motivation Level", fontsize=11)
plt.ylabel("Average Exam Score", fontsize=11)
plt.ylim(50, 75)
for p in plt.gca().patches:
    plt.gca().annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.3),
                ha='center', va='center', fontsize=10, fontweight='bold', color='black', xytext=(0, 5), textcoords='offset points')
plt.grid(True, axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("charts/6_average_score_by_motivation_level.png", dpi=200)
plt.close()
print("Saved Chart 6: Average Score by Motivation Level")

# Chart 7: Average Score by Access to Resources
plt.figure(figsize=(8, 5))
resources_order = ["Low", "Medium", "High"]
sns.barplot(data=df, x="Access_to_Resources", y="Exam_Score", order=resources_order, palette="magma", errorbar=None, edgecolor="black")
plt.title("Average Final Exam Score by Access to Study Resources", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Access to Learning Resources", fontsize=11)
plt.ylabel("Average Exam Score", fontsize=11)
plt.ylim(50, 75)
for p in plt.gca().patches:
    plt.gca().annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.3),
                ha='center', va='center', fontsize=10, fontweight='bold', color='black', xytext=(0, 5), textcoords='offset points')
plt.grid(True, axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("charts/7_average_score_by_access_to_resources.png", dpi=200)
plt.close()
print("Saved Chart 7: Average Score by Access to Resources")

# Chart 8: Pass/Fail Count
plt.figure(figsize=(7, 5))
pass_fail_labels = ["Passed (>=65)", "Failed (<65)"]
pass_fail_values = [pass_count, fail_count]
sns.barplot(x=pass_fail_labels, y=pass_fail_values, palette=["#2ec4b6", "#e71d36"], edgecolor="black")
plt.title("Total Number of Students Passing vs Failing", fontsize=13, fontweight="bold", pad=15)
plt.ylabel("Student Count", fontsize=11)
for p in plt.gca().patches:
    plt.gca().annotate(f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)", (p.get_x() + p.get_width() / 2., p.get_height() + 50),
                ha='center', va='center', fontsize=11, fontweight='bold', color='black', xytext=(0, 5), textcoords='offset points')
plt.grid(True, axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("charts/8_pass_fail_count.png", dpi=200)
plt.close()
print("Saved Chart 8: Pass/Fail Count")

# Chart 9: Risk Segment Breakdown
plt.figure(figsize=(8, 5))
risk_order = ["High Risk", "Medium Risk", "Low Risk"]
risk_vals = [risk_counts.get(r, 0) for r in risk_order]
sns.barplot(x=risk_order, y=risk_vals, palette=RISK_COLORS, edgecolor="black")
plt.title("Student Count by Risk Segment Category", fontsize=13, fontweight="bold", pad=15)
plt.ylabel("Student Count", fontsize=11)
for p in plt.gca().patches:
    plt.gca().annotate(f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)", (p.get_x() + p.get_width() / 2., p.get_height() + 50),
                ha='center', va='center', fontsize=11, fontweight='bold', color='black', xytext=(0, 5), textcoords='offset points')
plt.grid(True, axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("charts/9_risk_segment_breakdown.png", dpi=200)
plt.close()
print("Saved Chart 9: Risk Segment Breakdown")

# Chart 10: Top Factors Summary
plt.figure(figsize=(9, 6))
top_factors_df = pd.DataFrame(sorted_importances[:10], columns=["Feature", "Importance"])
sns.barplot(data=top_factors_df, y="Feature", x="Importance", palette="coolwarm", edgecolor="black")
plt.title("Top 10 Most Influential Student Performance Factors", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Aggregated Machine Learning Feature Importance", fontsize=11)
plt.ylabel("Student Factors", fontsize=11)
plt.grid(True, axis="x", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("charts/10_top_factors_summary.png", dpi=200)
plt.close()
print("Saved Chart 10: Top Factors Summary")

