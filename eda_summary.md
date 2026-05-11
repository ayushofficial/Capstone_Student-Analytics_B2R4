# Student Performance Exploratory Data Analysis (EDA) Summary

This document summarizes the insights gained from the exploratory analysis of the `StudentPerformanceFactors.csv` dataset, which contains **6607** student profiles and **20** attributes.

## 1. Overall Performance Metrics
* **Total Student Records**: 6607
* **Exam Score Range**: 55 to 101 marks
* **Average Exam Score**: 67.24 marks
* **Median Exam Score**: 67.00 marks
* **Class Pass Rate (Threshold >= 65)**: 78.02% (5155 passed, 1452 failed)

## 2. Key Numeric Factor Analysis (Correlation with Exam_Score)
Correlation values range from -1 to +1, indicating the strength and direction of the linear relationship. 

| Feature | Pearson Correlation | Relationship Strength | Interpretation |
| :--- | :---: | :---: | :--- |
| Pass_Flag | 0.6385 | High | Strong Positive Influence |
| Attendance | 0.5811 | High | Strong Positive Influence |
| Hours_Studied | 0.4455 | Medium | Moderate Positive Influence |
| Previous_Scores | 0.1751 | Very Low | No strong linear relationship |
| Tutoring_Sessions | 0.1565 | Very Low | No strong linear relationship |
| Physical_Activity | 0.0278 | Very Low | No strong linear relationship |
| Sleep_Hours | -0.0170 | Very Low | No strong linear relationship |

### Important Insights:
1. **Attendance (0.5811) and Hours Studied (0.4455)** have the absolute strongest relationships with exam performance. Regular class attendance and dedicated study hours are the absolute foundations of academic success.
2. **Previous Scores (0.1751)** and **Tutoring Sessions (0.1565)** are also very useful, positive indicators, showing that past preparation and seek-help behaviors significantly aid final scores.
3. **Sleep Hours (-0.0170)** has a negligible linear correlation. While sleep is crucial for physical and cognitive well-being, its relationship to scores within this range is non-linear and should not be overemphasized as a direct predictor.
4. *Remember: Correlation does not prove causation.* While these factors move in tandem, other underlying student, home, or school variables can influence both factors simultaneously.

## 3. Socio-Cultural & Environment Factors Analysis (Averages)

### Average Score by Motivation Level
* **High Motivation**: 67.70 marks
* **Medium Motivation**: 67.33 marks
* **Low Motivation**: 66.75 marks

### Average Score by Access to Resources
* **High Access**: 68.09 marks
* **Medium Access**: 67.13 marks
* **Low Access**: 66.20 marks

### Average Score by Family Income
* **High Income**: 67.84 marks
* **Medium Income**: 67.33 marks
* **Low Income**: 66.85 marks

### Average Score by School Type
* **Private**: 67.29 marks
* **Public**: 67.21 marks

### Average Score by Gender
* **Female**: 67.24 marks
* **Male**: 67.23 marks
* Note: The average scores across genders are practically identical, showing no performance bias.

## 4. Student Risk Segmentation Breakdown
Students are segmented into three risk categories based on their exam scores:
* **Low Risk** (Score >= 70): **1625** students (24.60%)
* **Medium Risk** (Score 65 to 69): **3530** students (53.43%)
* **High Risk** (Score < 65): **1452** students (21.98%)
