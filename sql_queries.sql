-- =========================================================================
-- SQL queries for the Student Performance Analytics System
-- Table name assumed: student_performance
-- Clean, optimized, and copy-paste ready queries
-- =========================================================================
-- COPY student_performance 
--FROM 'D:\StudPerf\cleaned_student_performance.csv' 
--DELIMITER ',' 
--CSV HEADER;

-- 1. Create the student_performance table
CREATE TABLE student_performance (
    Hours_Studied INT,
    Attendance INT,
    Parental_Involvement VARCHAR(20),
    Access_to_Resources VARCHAR(20),
    Extracurricular_Activities VARCHAR(5),
    Sleep_Hours INT,
    Previous_Scores INT,
    Motivation_Level VARCHAR(20),
    Internet_Access VARCHAR(5),
    Tutoring_Sessions INT,
    Family_Income VARCHAR(20),
    Teacher_Quality VARCHAR(20),
    School_Type VARCHAR(20),
    Peer_Influence VARCHAR(20),
    Physical_Activity INT,
    Learning_Disabilities VARCHAR(5),
    Parental_Education_Level VARCHAR(30),
    Distance_from_Home VARCHAR(20),
    Gender VARCHAR(10),
    Exam_Score INT
);

-- 2. Average exam score by study category
-- Low Study Hours (< 15), Medium (15-24), High (>= 25)
SELECT 
    CASE 
        WHEN Hours_Studied < 15 THEN 'Low Study Hours'
        WHEN Hours_Studied BETWEEN 15 AND 24 THEN 'Medium Study Hours'
        ELSE 'High Study Hours'
    END AS Study_Category,
    COUNT(*) AS Student_Count,
    ROUND(AVG(Exam_Score), 2) AS Average_Exam_Score,
    MIN(Exam_Score) AS Min_Exam_Score,
    MAX(Exam_Score) AS Max_Exam_Score
FROM student_performance
GROUP BY 
    CASE 
        WHEN Hours_Studied < 15 THEN 'Low Study Hours'
        WHEN Hours_Studied BETWEEN 15 AND 24 THEN 'Medium Study Hours'
        ELSE 'High Study Hours'
    END
ORDER BY Average_Exam_Score DESC;

-- 3. Pass rate by attendance category
-- Low (< 70), Medium (70-84), High (>= 85). Pass threshold is 65.
SELECT 
    CASE 
        WHEN Attendance < 70 THEN 'Low Attendance'
        WHEN Attendance BETWEEN 70 AND 84 THEN 'Medium Attendance'
        ELSE 'High Attendance'
    END AS Attendance_Category,
    COUNT(*) AS Total_Students,
    SUM(CASE WHEN Exam_Score >= 65 THEN 1 ELSE 0 END) AS Passed_Students,
    SUM(CASE WHEN Exam_Score < 65 THEN 1 ELSE 0 END) AS Failed_Students,
    ROUND(100.0 * SUM(CASE WHEN Exam_Score >= 65 THEN 1 ELSE 0 END) / COUNT(*), 2) AS Pass_Rate_Percentage
FROM student_performance
GROUP BY 
    CASE 
        WHEN Attendance < 70 THEN 'Low Attendance'
        WHEN Attendance BETWEEN 70 AND 84 THEN 'Medium Attendance'
        ELSE 'High Attendance'
    END
ORDER BY Pass_Rate_Percentage DESC;

-- 4. Average score by motivation level and access to resources
SELECT 
    Motivation_Level,
    Access_to_Resources,
    COUNT(*) AS Student_Count,
    ROUND(AVG(Exam_Score), 2) AS Average_Exam_Score
FROM student_performance
GROUP BY Motivation_Level, Access_to_Resources
ORDER BY 
    CASE Motivation_Level 
        WHEN 'High' THEN 1 
        WHEN 'Medium' THEN 2 
        ELSE 'Low' 
    END, 
    Average_Exam_Score DESC;

-- 5. Students needing academic support
-- Defined as failed (Exam_Score < 65) AND low class attendance (< 70)
SELECT 
    Gender,
    Hours_Studied,
    Attendance,
    Previous_Scores,
    Motivation_Level,
    Tutoring_Sessions,
    Exam_Score
FROM student_performance
WHERE Exam_Score < 65 AND Attendance < 70
ORDER BY Exam_Score ASC, Attendance ASC;

-- 6. Pass rate by school type and gender
SELECT 
    School_Type,
    Gender,
    COUNT(*) AS Total_Students,
    SUM(CASE WHEN Exam_Score >= 65 THEN 1 ELSE 0 END) AS Passed_Students,
    ROUND(100.0 * SUM(CASE WHEN Exam_Score >= 65 THEN 1 ELSE 0 END) / COUNT(*), 2) AS Pass_Rate_Percentage
FROM student_performance
GROUP BY School_Type, Gender
ORDER BY School_Type, Pass_Rate_Percentage DESC;

-- 7. Average score by tutoring sessions
SELECT 
    Tutoring_Sessions,
    COUNT(*) AS Student_Count,
    ROUND(AVG(Exam_Score), 2) AS Average_Exam_Score,
    MIN(Exam_Score) AS Min_Score,
    MAX(Exam_Score) AS Max_Score
FROM student_performance
GROUP BY Tutoring_Sessions
ORDER BY Tutoring_Sessions ASC;

-- 8. Risk segment grouping (counts, percentages, and average metrics)
SELECT 
    CASE 
        WHEN Exam_Score < 65 THEN 'High Risk'
        WHEN Exam_Score BETWEEN 65 AND 69 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END AS Risk_Segment,
    COUNT(*) AS Student_Count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM student_performance), 2) AS Percentage_Of_Total,
    ROUND(AVG(Exam_Score), 2) AS Average_Exam_Score,
    ROUND(AVG(Attendance), 2) AS Average_Attendance,
    ROUND(AVG(Hours_Studied), 2) AS Average_Weekly_Study_Hours
FROM student_performance
GROUP BY 
    CASE 
        WHEN Exam_Score < 65 THEN 'High Risk'
        WHEN Exam_Score BETWEEN 65 AND 69 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END
ORDER BY Average_Exam_Score ASC;

-- 9. Average score by previous performance category
-- Low (< 65), Medium (65-84), High (>= 85)
SELECT 
    CASE 
        WHEN Previous_Scores < 65 THEN 'Low'
        WHEN Previous_Scores BETWEEN 65 AND 84 THEN 'Medium'
        ELSE 'High'
    END AS Previous_Performance_Category,
    COUNT(*) AS Student_Count,
    ROUND(AVG(Exam_Score), 2) AS Average_Exam_Score
FROM student_performance
GROUP BY 
    CASE 
        WHEN Previous_Scores < 65 THEN 'Low'
        WHEN Previous_Scores BETWEEN 65 AND 84 THEN 'Medium'
        ELSE 'High'
    END
ORDER BY Average_Exam_Score DESC;

-- 10. Top groups with highest average performance
-- Groups are constructed by combining school type, parental involvement, and motivation level
SELECT 
    School_Type,
    Parental_Involvement,
    Motivation_Level,
    COUNT(*) AS Student_Count,
    ROUND(AVG(Exam_Score), 2) AS Average_Exam_Score
FROM student_performance
GROUP BY School_Type, Parental_Involvement, Motivation_Level
HAVING COUNT(*) >= 15
ORDER BY Average_Exam_Score DESC
LIMIT 10;
