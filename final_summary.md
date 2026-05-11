# Student Performance Analytics System - Final Project Summary

## 1. Dataset Overview
* **Dataset File**: `StudentPerformanceFactors.csv`
* **Dataset Shape**: 6607 rows, 20 columns
* **Target Variable**: `Exam_Score`
* **Default Pass/Fail Threshold**: 65 marks
* **Pass Rate on Dataset**: 78.02% (Passed: 5155, Failed: 1452)
* **Average Exam Score**: 67.24 marks
* **Median Exam Score**: 67.00 marks
* **Exam Score Range**: [55, 101]

## 2. Data Cleaning & Validation Steps
1. **Column Validation**: Verified that all 20 expected columns are present. No missing columns or extra unexpected columns were found.
2. **Text Standardization**: Stripped leading/trailing whitespaces and mapped standard missing value strings to `NaN` for all categorical columns.
3. **Imputation**:
   * Numeric Columns: Replaced missing values with column medians (0 missing numeric values found).
   * Categorical Columns: Imputed missing values in `Teacher_Quality` (78 missing), `Parental_Education_Level` (90 missing), and `Distance_from_Home` (67 missing) using their respective statistical **modes**.
4. **Duplicate Detection**: Verified that there are **0** duplicate rows in the dataset.
5. **Data Quality Issue Flagging**: Identified **1** record where the `Exam_Score` was above 100 (Index 1525, score of **101**). As instructed, this was flagged as a data quality anomaly but *not* deleted, preserving the integrity of the raw record while ensuring transparency.

## 3. Feature Engineering Outcomes
The following five critical columns were engineered to enhance analytics:
1. `Pass_Flag`: Binary variable indicating if a student scored >= 65 (Pass = 1, Fail = 0).
2. `Risk_Level`: Categorical segment based on exam score: "High Risk" (< 65), "Medium Risk" (65-69), and "Low Risk" (>= 70).
3. `Study_Category`: Grouping of study time: "Low Study Hours" (< 15 hrs), "Medium Study Hours" (15-24 hrs), and "High Study Hours" (>= 25 hrs).
4. `Attendance_Category`: School attendance rate groupings: "Low Attendance" (< 70%), "Medium Attendance" (70-84%), and "High Attendance" (>= 85%).
5. `Previous_Performance_Category`: Prior performance categories: "Low" (< 65), "Medium" (65-84), and "High" (>= 85).

## 4. Key Exploratory Data Analysis & Correlation Findings
* **Strongest Relationships**: **Attendance** has an extremely high Pearson correlation of **0.5811** with `Exam_Score`. **Hours_Studied** is the second most crucial factor with a correlation of **0.4455**.
* **Moderate Relationships**: **Previous_Scores** (0.1751) and **Tutoring_Sessions** (0.1565) show solid, positive correlations, indicating that historical academic foundation and seeking supportive tutoring are beneficial.
* **Weak Linear Relationship**: **Sleep_Hours** has a correlation of **-0.0170** with `Exam_Score`. While sleep is critical for overall health, it has almost zero linear predictive value for the score in this range, indicating it should not be overemphasized as a linear predictor.
* **Environmental/Demographic Factors**: 
  * Students with high learning motivation average **67.70** marks, compared to **66.75** for low motivation.
  * High resource access averages **68.09** marks vs. **66.20** for low resource access.
  * Private school average scores (**67.29**) are slightly higher than public schools (**67.21**).
  * Female (**67.24**) and Male (**67.23**) scores are practically identical, demonstrating academic parity across genders in this dataset.

## 5. Model Performance Summary
We trained and compared multiple predictive models for student final scores:

### Regression Performance (Continuous score prediction)
We compared a baseline **Linear Regression** model with a non-linear **Random Forest Regressor**:

1. **Linear Regression (Baseline & Primary)**:
   * Mean Absolute Error (MAE): **0.4524** marks (predictions are on average within 0.45 marks of actual scores).
   * Root Mean Squared Error (RMSE): **1.8044**
   * R-squared (R2) Score: **0.7696** (meaning this model explains **77.0%** of the variance in student scores).
   * *Aesthetic & Statistical Note*: Linear Regression significantly outperforms Random Forest on this dataset. This suggests that student performance factors operate under a highly additive, direct, and linear logic in this data.

2. **Random Forest Regressor (Alternative)**:
   * Mean Absolute Error (MAE): **1.0841** marks
   * Root Mean Squared Error (RMSE): **2.1657**
   * R-squared (R2) Score: **0.6682** (66.8% variance explained)

### 5.1 Linear Regression Coefficient Significance Analysis
Using standard errors and Student's t-distributions, we computed the statistical significance (p-values) for each factor. A factor is considered highly significant if its **p-value is < 0.05** (proving the relationship has a less than 5% probability of being a random chance occurrence):

* **Highly Significant Behavioral Drivers (p-value = 0.0000)**:
  * **Hours Studied** (Coefficient = **+0.2932** per weekly hour): Every additional hour of study per week directly adds **0.29 marks** to the final grade.
  * **Attendance** (Coefficient = **+0.1989** per percentage point): Attendance has an exceptionally high impact. Every 10% increase in attendance adds **1.99 marks**.
  * **Previous Scores** (Coefficient = **+0.0490** per point): Reflects strong baseline knowledge.
  * **Tutoring Sessions** (Coefficient = **+0.5077** per monthly session): Direct positive intervention.
  * **Physical Activity** (Coefficient = **+0.1925** per day/week): Shows a small but highly significant positive coefficient.

* **Non-Significant Variables (p-value >= 0.05 - No Direct Linear Effect)**:
  * **Sleep Hours** (p-value = **0.5266**, Coef = **-0.0125**): Shows zero statistical significance. While vital for cognitive function, sleep hours within this range do not linearly influence final grades.
  * **School Type (Public)** (p-value = **0.7841**, Coef = **0.0172**): School type (Public vs Private) shows zero statistical significance, showing equal potential for success across school types.
  * **Gender (Male)** (p-value = **0.6589**, Coef = **-0.0257**): Gender shows zero statistical significance. Male and female students demonstrate equal performance.

### 5.2 Simplified Equation-Based Performance Model
To provide educators, students, and parents with a highly practical, easy-to-use forecasting tool, we built a simplified, action-oriented model using *only* the 5 key continuous behavioral factors. This model explains **64.2%** ($R^2 = 0.6422$) of the score variance:

$$\text{Estimated Exam Score} \approx 40.73 + 0.2891 \times H + 0.1988 \times A + 0.0483 \times P + 0.5102 \times T + 0.1507 \times Y$$

Where:
* $H$ = **Hours Studied** (weekly revision hours, e.g., 20)
* $A$ = **Attendance** (attendance percentage rate, e.g., 90)
* $P$ = **Previous Scores** (prior grade percentage, e.g., 75)
* $T$ = **Tutoring Sessions** (number of sessions per month, e.g., 2)
* $Y$ = **Physical Activity** (exercise frequency in days per week, e.g., 3)

**Example Calculation**: A student studying **18 hours/week** with **90% attendance**, **75% previous scores**, **2 tutoring sessions**, and **3 exercise days/week** is estimated to score:
$$\text{Estimated Score} = 40.73 + 0.2891(18) + 0.1988(90) + 0.0483(75) + 0.5102(2) + 0.1507(3) = 68.9 \text{ marks}.$$

### Classification Performance (Pass/Fail Prediction)
We trained a **Random Forest Classifier** to predict whether a student passes (Exam_Score >= 65):
* **Model Accuracy**: **90.77%**
* **Precision**: **0.9068**
* **Recall**: **0.9847**
* **F1-Score**: **0.9441**
* **ROC-AUC Score**: **0.9686** (showing exceptional diagnostic performance)

## 6. Student Risk Segmentation Analysis
* **Low Risk Segment** (Score >= 70): **1625** students (**24.60%**). These students show strong habits.
* **Medium Risk Segment** (Score 65 to 69): **3530** students (**53.43%**). These are borderline students who could fall into high risk without supportive intervention.
* **High Risk Segment** (Score < 65): **1452** students (**21.98%**). These students need urgent academic support and intervention.

## 7. SQL Insights Generated
The script created a comprehensive database schema and analytical script in `sql_queries.sql` including:
* Custom DDL table creation scripts.
* Performance breakdown queries by study categories, tutoring sessions, and previous academic scores.
* Dynamic pass rate calculations by attendance categories, and school type/gender.
* Intersectional analysis of motivation and resource access.
* Targeted academic support identification query to fetch vulnerable students.

## 8. GenAI Feedback Reports Usefulness
We generated **5 personalized student feedback reports** stored in `feedback_reports.txt`. Each report:
* Runs our machine learning models to provide an estimated exam score and exact pass probability.
* Dynamically scans student variables to select actual, controllable strengths (e.g. good attendance, active tutoring) and areas to improve.
* Suggests 3 specific, supportive, actionable steps to improve performance without blame or sensitivity biases.
* Word counts are optimized to stay between **120 and 180 words**, making them highly engaging and suitable for parent-teacher conferences or direct student handouts.

## 9. Limitations & Future Improvements
* **Socio-Cultural Context**: The models use home data like family income and parental education. While important for demographic understanding, these should *never* be used to make prejudiced predictive judgments about a student's potential. Models must focus on controllable behaviors (study hours, attendance, motivation, tutoring).
* **Linear vs Non-Linearity**: While tree-based models like Random Forest capture complex interactions, student performance in this dataset is remarkably linear and additive. Our baseline Linear Regression model achieved a superior R2 score of **0.7696** (vs. 0.6682 for Random Forest), highlighting the direct and transparent effect of controllable features like attendance and study hours.
* **Data Timeframe**: The dataset is a single-semester snapshot. Incorporating longitudinal student performance tracking (multiple semesters or weekly quiz metrics) could help build real-time early warning systems.
* **Action Plan**: Future updates could integrate automated email or SMS notifications of these feedback reports directly to students and parents.
