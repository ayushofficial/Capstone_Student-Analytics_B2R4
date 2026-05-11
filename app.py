import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load env file for environment variables like GEMINI_API_KEY (with override to support live updates)
load_dotenv("project.env", override=True)

# Set page configurations
st.set_page_config(
    page_title="Student Performance Diagnostics System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #fcfcfc;
    }
    h1 {
        color: #0f4c5c;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }
    h2, h3 {
        color: #1e6091;
        font-family: 'Outfit', sans-serif;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 6px solid #0f4c5c;
        margin-bottom: 15px;
    }
    .metric-card-risk {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 6px solid #e71d36;
        margin-bottom: 15px;
    }
    .metric-card-pass {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 6px solid #2ec4b6;
        margin-bottom: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        color: #5c677d;
    }
    .stTabs [aria-selected="true"] {
        color: #0f4c5c !important;
        border-bottom-color: #0f4c5c !important;
    }
    div[data-testid="stSidebar"] {
        background-color: #f1f5f9;
        border-right: 1px solid #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# Define constants
PASS_THRESHOLD = 65

# Helper: Load trained models and columns
@st.cache_resource
def load_models_and_meta():
    model_path = "student_performance_models.pkl"
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data

meta = load_models_and_meta()

if meta is None:
    st.error("⚠️ Model files not found! Please run the analysis script first: `py run_analysis.py` to train and save the predictive models.")
    st.stop()

# Extract cached entities
lr_stats_model = meta["lr_stats_model"]
clf_model = meta["clf_model"]
reg_model = meta["reg_model"]
lr_simple_model = meta["lr_simple_model"]
X_encoded_stats_columns = meta["X_encoded_stats_columns"]
X_encoded_columns = meta["X_encoded_columns"]
model_features = meta["model_features"]
cat_features_for_model = meta["cat_features_for_model"]
key_behavior_cols = meta["key_behavior_cols"]

# App Title & Subtitle with Hero Section
col1, col2 = st.columns([1, 10])
with col1:
    st.markdown("<h1 style='font-size: 4rem; margin:0; padding:0;'>🎓</h1>", unsafe_allow_html=True)
with col2:
    st.markdown("<h1 style='margin:0; padding:0;'>Student Performance Analytics & Diagnostic System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:1.15rem; color:#5c677d; margin-top:5px;'>An AI-assisted, statistically-validated framework for predicting exam results, assessing academic risks, and delivering supportive feedback.</p>", unsafe_allow_html=True)

st.markdown("---")

# =========================================================================
# SIDEBAR: USER INPUT CONTROLS
# =========================================================================
st.sidebar.markdown("<h2 style='color:#0f4c5c; text-align:center;'>💻 Student Profile</h2>", unsafe_allow_html=True)
st.sidebar.markdown("Configure the student's demographic details and academic behavioral habits below:")
# Readme refer
st.sidebar.markdown("### 📚 Academic Habits (Controllable)")
hours = st.sidebar.slider("Weekly Study Hours", min_value=1, max_value=40, value=18, help="Number of hours the student revises independently each week.")
attendance = st.sidebar.slider("School Attendance Rate (%)", min_value=50, max_value=100, value=88, help="Percentage of school classes attended in the semester.")
prev_scores = st.sidebar.slider("Previous Scores (%)", min_value=50, max_value=100, value=75, help="Academic performance level in the previous evaluation.")
tutoring = st.sidebar.slider("Tutoring Sessions / Month", min_value=0, max_value=8, value=2, help="Number of formal external tutoring sessions attended monthly.")
physical = st.sidebar.slider("Physical Activity (Days/Week)", min_value=0, max_value=7, value=3, help="Number of days per week the student exercises or plays sports.")
sleep = st.sidebar.slider("Daily Sleep Hours", min_value=4, max_value=12, value=7, help="Average duration of sleep the student gets each night.")

st.sidebar.markdown("### 🏠 Environmental & Background")
parental = st.sidebar.selectbox("Parental Involvement Level", options=["High", "Medium", "Low"], index=1)
motivation = st.sidebar.selectbox("Intrinsic Motivation Level", options=["High", "Medium", "Low"], index=1)
resources = st.sidebar.selectbox("Access to Study Resources", options=["High", "Medium", "Low"], index=1)
family_income = st.sidebar.selectbox("Family Income Category", options=["High", "Medium", "Low"], index=1)
teacher_quality = st.sidebar.selectbox("Perceived Teacher Quality", options=["High", "Medium", "Low"], index=1)
peer_influence = st.sidebar.selectbox("Peer Social Influence", options=["Positive", "Neutral", "Negative"], index=1)
internet = st.sidebar.selectbox("Home Internet Access?", options=["Yes", "No"], index=0)
extracurricular = st.sidebar.selectbox("Extracurricular Activities?", options=["Yes", "No"], index=1)
disability = st.sidebar.selectbox("Learning Disabilities?", options=["No", "Yes"], index=0)
parent_edu = st.sidebar.selectbox("Parental Education Level", options=["Postgraduate", "High School", "College"], index=2)
distance = st.sidebar.selectbox("Distance from Home to School", options=["Near", "Moderate", "Far"], index=0)
school_type = st.sidebar.selectbox("School Sector Type", options=["Private", "Public"], index=0)
gender = st.sidebar.selectbox("Student Gender", options=["Female", "Male"], index=0)

# Build a single DataFrame from user inputs matching the format of original dataset
input_dict = {
    "Hours_Studied": hours,
    "Attendance": attendance,
    "Parental_Involvement": parental,
    "Access_to_Resources": resources,
    "Extracurricular_Activities": extracurricular,
    "Sleep_Hours": sleep,
    "Previous_Scores": prev_scores,
    "Motivation_Level": motivation,
    "Internet_Access": internet,
    "Tutoring_Sessions": tutoring,
    "Family_Income": family_income,
    "Teacher_Quality": teacher_quality,
    "School_Type": school_type,
    "Peer_Influence": peer_influence,
    "Physical_Activity": physical,
    "Learning_Disabilities": disability,
    "Parental_Education_Level": parent_edu,
    "Distance_from_Home": distance,
    "Gender": gender
}
df_input = pd.DataFrame([input_dict])

# Preprocess and dummy encode for the Linear Regression Model (drop_first=True)
df_encoded_stats = pd.get_dummies(df_input, columns=cat_features_for_model, drop_first=True)
df_encoded_stats = df_encoded_stats.reindex(columns=X_encoded_stats_columns, fill_value=0)

# Preprocess and dummy encode for the RF models (drop_first=False)
df_encoded_clf = pd.get_dummies(df_input, columns=cat_features_for_model, drop_first=False)
df_encoded_clf = df_encoded_clf.reindex(columns=X_encoded_columns, fill_value=0)

# =========================================================================
# WEB APPLICATION TABS
# =========================================================================
tab1, tab2, tab3, tab4 = st.tabs(["🎯 AI Predictive Diagnostic", "📊 Model Comparative Analytics", "📈 Student Data Explorer", "💬 Aura: Academic Success Coach"])

# -------------------------------------------------------------------------
# TAB 1: AI PREDICTIVE DIAGNOSTIC & FEEDBACK
# -------------------------------------------------------------------------
with tab1:
    st.markdown("### Real-Time Student Predictive Insights")
    st.markdown("This tab displays immediate predictions of academic scores and pass outcomes based on the student's profile, driven by our best-performing models.")

    # 1. RUN PREDICTIONS
    # Best Regression Model: lr_stats_model (Full Linear Regression, R2 = 77.0%)
    pred_score = lr_stats_model.predict(df_encoded_stats)[0]
    
    # Best Classifier: clf_model (Random Forest Classifier, AUC = 0.969)
    pass_prob = clf_model.predict_proba(df_encoded_clf)[0][1] * 100
    
    # Determine risk level
    if pred_score >= 70:
        risk_label = "Low Risk"
        risk_class = "metric-card-pass"
        risk_color = "#2ec4b6"
    elif pred_score >= 65:
        risk_label = "Medium Risk"
        risk_class = "metric-card"
        risk_color = "#f7a072"
    else:
        risk_label = "High Risk"
        risk_class = "metric-card-risk"
        risk_color = "#e71d36"

    # Display Metrics in dynamic premium columns
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.markdown(f"""
            <div class="{risk_class}">
                <p style='color:#5c677d; font-size:0.9rem; text-transform:uppercase; font-weight:600; margin:0;'>Estimated Exam Score</p>
                <h2 style='font-size:2.8rem; margin: 10px 0; color:#0f4c5c;'>{pred_score:.1f} <span style='font-size:1.2rem; color:#5c677d;'>marks</span></h2>
                <p style='margin:0; font-size:0.95rem; color:#5c677d;'>Model accuracy: <b>MAE ± 0.45 marks</b></p>
            </div>
        """, unsafe_allow_html=True)
        
    with m_col2:
        st.markdown(f"""
            <div class="{risk_class}">
                <p style='color:#5c677d; font-size:0.9rem; text-transform:uppercase; font-weight:600; margin:0;'>Pass Probability</p>
                <h2 style='font-size:2.8rem; margin: 10px 0; color:#0f4c5c;'>{pass_prob:.1f}%</h2>
                <p style='margin:0; font-size:0.95rem; color:#5c677d;'>Pass threshold is: <b>{PASS_THRESHOLD} marks</b></p>
            </div>
        """, unsafe_allow_html=True)
        
    with m_col3:
        st.markdown(f"""
            <div class="{risk_class}">
                <p style='color:#5c677d; font-size:0.9rem; text-transform:uppercase; font-weight:600; margin:0;'>Academic Risk Segment</p>
                <h2 style='font-size:2.8rem; margin: 10px 0; color:{risk_color};'>{risk_label}</h2>
                <p style='margin:0; font-size:0.95rem; color:#5c677d;'>Diagnostic status: <b>Active</b></p>
            </div>
        """, unsafe_allow_html=True)

    # 2. PERSONALIZED SUPPORTIVE FEEDBACK REPORT
    st.markdown("<h3 style='margin-top:25px;'>📋 Personalized Student Diagnostic Report</h3>", unsafe_allow_html=True)
    
    # Generate Strengths, Areas to Improve, and Actionable Steps based on Controllable parameters
    strengths = []
    areas_to_improve = []
    actions = []
    
    # Evaluate Strengths
    if attendance >= 85:
        strengths.append(f"Excellent class attendance ({attendance}%), demonstrating high academic engagement.")
    if hours >= 20:
        strengths.append(f"Superb weekly study discipline, spending {hours} hours on independent revision.")
    if tutoring >= 2:
        strengths.append(f"Proactive help-seeking habits, attending {tutoring} tutoring sessions monthly.")
    if motivation == "High":
        strengths.append("High levels of intrinsic study motivation and proactive academic drive.")
    if prev_scores >= 80:
        strengths.append(f"Strong prior semester knowledge, with solid performance foundation ({prev_scores}%).")
    if physical >= 3:
        strengths.append(f"Healthy physical activity level ({physical} days/week), supporting cognitive stamina.")
        
    # Ensure at least 2 strengths
    if len(strengths) < 2:
        strengths.append("Possesses home internet access, offering high connectivity for self-learning resource retrieval.")
        if internet == "Yes":
            strengths.append("Active home study framework with supportive internet educational access.")
        else:
            strengths.append("Enjoys consistent daily study opportunities and is fully capable of expanding revision routines.")

    # Evaluate Areas to Improve & Actions
    if attendance < 85:
        areas_to_improve.append(f"School attendance is currently low ({attendance}%), leading to gaps in understanding classroom curricula.")
        actions.append("Commit to full attendance of classes daily on-time, avoiding absences unless strictly necessary.")
    if hours < 18:
        areas_to_improve.append(f"Weekly independent study time is currently limited ({hours} hours), which impacts practical exercises.")
        actions.append(f"Develop a weekly revision calendar, aiming to study independently for 1.5 to 2 hours every day.")
    if tutoring < 2:
        areas_to_improve.append(f"Coaching or tutoring resources are currently limited ({tutoring} sessions/month), which could assist with doubts.")
        actions.append("Join peer-study groups or attend teacher-led academic counseling hours weekly to clear lingering doubts.")
    if motivation == "Low":
        areas_to_improve.append("Academic focus shows signs of low motivation, hindering persistent studying.")
        actions.append("Break up long revision subjects into short 25-minute study segments (Pomodoro Technique) to boost motivation.")
    if prev_scores < 65:
        areas_to_improve.append(f"Foundational skills show vulnerability due to low past semester results ({prev_scores}%).")
        actions.append("Begin reviews from primary baseline topics, solidifying fundamental concepts before tackling advanced work.")

    # Enforce minimums for areas and actions
    if len(areas_to_improve) < 2:
        areas_to_improve.append("Sleep routine represents a vital biological baseline that must be guarded alongside study routines.")
        actions.append("Maintain an organized biological calendar, securing 7 to 8 hours of sleep each night to maximize mental acuity.")
    if len(actions) < 3:
        actions.append("Work on practice tests under timed conditions to improve exam confidence and performance.")

    # Render Report
    report_col, chart_col = st.columns([3, 2])
    
    with report_col:
        st.markdown(f"""
        <div style='background-color:#ffffff; border: 1px solid #e2e8f0; border-radius:12px; padding:25px; box-shadow: 0 4px 6px rgba(0,0,0,0.02);'>
            <h4 style='color:#0f4c5c; margin-top:0;'>🛡️ Strengths & Pillars</h4>
            <ul style='padding-left: 20px; font-size:1.02rem; line-height:1.6;'>
                <li>{"</li><li>".join(strengths[:2])}</li>
            </ul>
            
            <h4 style='color:#e36414; margin-top:20px;'>⚠️ Areas for Growth</h4>
            <ul style='padding-left: 20px; font-size:1.02rem; line-height:1.6;'>
                <li>{"</li><li>".join(areas_to_improve[:2])}</li>
            </ul>
            
            <h4 style='color:#1e6091; margin-top:20px;'>🚀 Recommended Action Plan</h4>
            <ol style='padding-left: 20px; font-size:1.02rem; line-height:1.6;'>
                <li>{"</li><li>".join(actions[:3])}</li>
            </ol>
            
            <p style='font-size:0.88rem; color:#718096; font-style:italic; margin-top:25px;'>
                Note: This AI diagnostic evaluation report utilizes our statistically-validated machine learning models to analyze behavioral patterns. It is designed to act as supportive coaching guidance for teachers, parents, and students.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with chart_col:
        # Display a beautiful dial chart or donut chart of pass probability
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#ffffff')
        
        # Donut plot for pass probability
        sizes = [pass_prob, 100 - pass_prob]
        colors = ['#2ec4b6', '#e71d36'] if pass_prob >= 50 else ['#e36414', '#e71d36']
        wedges, texts, autotexts = ax.pie(
            sizes, 
            colors=colors, 
            startangle=90, 
            wedgeprops=dict(width=0.3, edgecolor='white', linewidth=2), 
            autopct='%1.1f%%', 
            pctdistance=0.85
        )
        
        # Customize center text
        ax.text(0, 0, f"Pass\nProbability", ha='center', va='center', fontsize=14, fontweight='bold', color='#0f4c5c')
        plt.setp(texts, size=10, color='#5c677d')
        plt.setp(autotexts, size=12, weight="bold", color="white")
        
        # Disable wedge percentages if probability is extremely low or high to avoid overlaying text
        if pass_prob < 10 or pass_prob > 90:
            autotexts[1].set_text("")
            
        ax.axis('equal')  
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# TAB 2: MODEL COMPARATIVE ANALYTICS & EQUATIONS
# -------------------------------------------------------------------------
with tab2:
    st.markdown("### Model Comparative Leaderboard")
    st.markdown("See how different predictive models perform on this dataset. In tabular student data, simpler additive models can perform significantly better than tree-based ensembles due to highly linear underlying dimensions.")
    
    # 1. Models Leaderboard Table
    leaderboard_data = {
        "Model Name": [
            "Full Linear Regression (Baseline & Primary)",
            "Random Forest Regressor (Advanced Ensembles)",
            "Simplified Equation-Based Regression",
            "Random Forest Classifier (Pass/Fail Model)"
        ],
        "Task Type": [
            "Continuous Regression",
            "Continuous Regression",
            "Continuous Regression",
            "Binary Classification"
        ],
        "R² / Accuracy": [
            "77.0% (R² = 0.7696)",
            "66.8% (R² = 0.6682)",
            "64.2% (R² = 0.6422)",
            "90.8% (Accuracy)"
        ],
        "Mean Absolute Error (MAE)": [
            "0.4524 marks",
            "1.0841 marks",
            "1.2642 marks",
            "Not Applicable (Classification)"
        ],
        "Status": [
            "🏆 Best Score Model",
            "Alternative",
            "Best Simple Formula",
            "🏆 Best Classifier"
        ]
    }
    df_lead = pd.DataFrame(leaderboard_data)
    st.table(df_lead)
    
    st.markdown("### 🧮 Practical Student Score Estimator Formula")
    st.markdown("""
        We isolated the **5 key statistically significant, controllable behavioral dimensions** to construct a simplified forecasting equation. 
        Teachers and parents can compute this mental-math equation directly to estimate student exam marks!
    """)
    
    # Display formula box
    st.markdown(f"""
        <div style='background-color: #0f4c5c; color: white; border-radius:12px; padding:25px; text-align:center; box-shadow: 0 4px 10px rgba(0,0,0,0.1);'>
            <h3 style='color: white; margin-top:0;'>Estimated Exam Score Formula</h3>
            <p style='font-size:1.6rem; font-family:"Courier New", monospace; font-weight:bold; margin: 15px 0;'>
                Exam_Score = 40.73 + 0.2891 × H + 0.1988 × Attendance + 0.0483 × Previous_Score + 0.5102 × Tutoring + 0.1507 × Exercise
            </p>
            <p style='font-size:0.95rem; opacity:0.9; margin:0;'>
                H: Weekly Study Hours | Attendance: Percentage Rate | Previous_Score: Past Semester Mark | Tutoring: Sessions/Month | Exercise: Active Days/Week
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Run the simple math prediction
    simple_score = 40.73 + (0.2891 * hours) + (0.1988 * attendance) + (0.0483 * prev_scores) + (0.5102 * tutoring) + (0.1507 * physical)
    
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(f"""
            <div class="metric-card" style='margin-top:20px; border-left-color:#e36414;'>
                <p style='color:#5c677d; font-size:0.9rem; text-transform:uppercase; font-weight:600; margin:0;'>Simplified Equation Estimate</p>
                <h2 style='font-size:2.5rem; margin:10px 0; color:#e36414;'>{simple_score:.2f} <span style='font-size:1.1rem; color:#5c677d;'>marks</span></h2>
                <p style='margin:0; font-size:0.95rem; color:#5c677d;'>Explains <b>64.2%</b> of overall score variance</p>
            </div>
        """, unsafe_allow_html=True)
    with sc2:
        st.markdown(f"""
            <div class="metric-card" style='margin-top:20px; border-left-color:#1e6091;'>
                <p style='color:#5c677d; font-size:0.9rem; text-transform:uppercase; font-weight:600; margin:0;'>Full Linear Regression Estimate</p>
                <h2 style='font-size:2.5rem; margin:10px 0; color:#1e6091;'>{pred_score:.2f} <span style='font-size:1.1rem; color:#5c677d;'>marks</span></h2>
                <p style='margin:0; font-size:0.95rem; color:#5c677d;'>Explains <b>77.0%</b> of overall score variance</p>
            </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# TAB 3: STUDENT DATA EXPLORER
# -------------------------------------------------------------------------
with tab3:
    st.markdown("### Interactive Exploratory Data Playground")
    st.markdown("Analyze how study behaviors correlate with student final grades. Load our cleaned student performance database to perform diagnostic group analyses.")
    
    # Load dataset
    @st.cache_data
    def load_clean_data():
        if os.path.exists("cleaned_student_performance.csv"):
            return pd.read_csv("cleaned_student_performance.csv")
        return None
        
    df_clean = load_clean_data()
    
    if df_clean is not None:
        st.markdown(f"📊 Cleaned Dataset Loaded: **{len(df_clean)} records** and **{df_clean.shape[1]} features**.")
        
        # Display dataset preview
        with st.expander("🔍 View Cleaned Database Preview"):
            st.dataframe(df_clean.head(10))
            
        # Group-by Selector
        st.markdown("### Category Distribution Explorer")
        g_col1, g_col2 = st.columns([1, 2])
        
        with g_col1:
            group_col = st.selectbox(
                "Choose Student Category Variable:",
                options=[
                    "Motivation_Level", "Access_to_Resources", "Study_Category", 
                    "Attendance_Category", "Previous_Performance_Category", 
                    "Parental_Involvement", "Family_Income", "Teacher_Quality", "Gender"
                ]
            )
            
            # Show descriptive statistics for groups
            grouped_df = df_clean.groupby(group_col)["Exam_Score"].agg(["count", "mean", "min", "max"]).reset_index()
            grouped_df.columns = [group_col, "Student Count", "Average Grade", "Minimum", "Maximum"]
            st.write(grouped_df)
            
        with g_col2:
            # Generate custom seaborn plot
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#ffffff')
            
            sns.barplot(
                data=df_clean, 
                x=group_col, 
                y="Exam_Score", 
                ax=ax, 
                palette="viridis", 
                edgecolor="black", 
                errorbar=None
            )
            ax.set_title(f"Average Exam Score by {group_col.replace('_', ' ')}", fontsize=12, fontweight='bold', color='#0f4c5c')
            ax.set_ylabel("Average Exam Score", fontsize=10, color='#0f4c5c')
            ax.set_xlabel(group_col.replace('_', ' '), fontsize=10, color='#0f4c5c')
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(fig)
            
    else:
        st.warning("Cleaned student dataset (`cleaned_student_performance.csv`) was not found in the workspace directory. Run `py run_analysis.py` to compile it.")

# =========================================================================
# TAB 4: AURA: PERSONALIZED ACADEMIC SUCCESS COACH (CHATBOT)
# =========================================================================
with tab4:
    # 1. Define Questionnaire Questions
    CHATBOT_QUESTIONS = [
        {
            "key": "Student_Name",
            "question": "Hello! I'm Aura, your GenAI Academic Coach. Let's create a customized success plan for you. First, **what is your name**?",
            "type": "str"
        },
        {
            "key": "Hours_Studied",
            "question": "Hi! **How many hours do you study independently each week**? (e.g., 15)",
            "type": "int",
            "min": 1,
            "max": 40
        },
        {
            "key": "Attendance",
            "question": "Great! Next, **what is your school attendance rate as a percentage**? (e.g., 90)",
            "type": "int",
            "min": 50,
            "max": 100
        },
        {
            "key": "Previous_Scores",
            "question": "Got it. **What was your average score (%) in the previous evaluations or exams**? (e.g., 75)",
            "type": "int",
            "min": 50,
            "max": 100
        },
        {
            "key": "Tutoring_Sessions",
            "question": "Thanks. **How many formal external tutoring or coaching sessions do you attend per month**? (e.g., 2)",
            "type": "int",
            "min": 0,
            "max": 8
        },
        {
            "key": "Sleep_Hours",
            "question": "Biological baselines are crucial! **On average, how many hours of sleep do you get each night**? (e.g., 7)",
            "type": "int",
            "min": 4,
            "max": 12
        },
        {
            "key": "Physical_Activity",
            "question": "How about physical fitness? **How many days per week do you exercise or play sports**? (e.g., 3)",
            "type": "int",
            "min": 0,
            "max": 7
        },
        {
            "key": "Distraction_Level",
            "question": "How would you describe your **daily screen, gaming, or general distraction level**? (Please reply with **High**, **Medium**, or **Low**)",
            "type": "choice",
            "choices": ["High", "Medium", "Low"]
        },
        {
            "key": "Motivation_Level",
            "question": "How would you describe your **intrinsic learning motivation**? (Please reply with **High**, **Medium**, or **Low**)",
            "type": "choice",
            "choices": ["High", "Medium", "Low"]
        },
        {
            "key": "Parental_Involvement",
            "question": "How would you describe your **parent's or guardian's involvement** in your academic activities? (Please reply with **High**, **Medium**, or **Low**)",
            "type": "choice",
            "choices": ["High", "Medium", "Low"]
        }
    ]

    # Helper function to validate & parse inputs
    def validate_and_parse_input(user_input, current_q):
        val_type = current_q["type"]
        if val_type == "str":
            stripped = user_input.strip()
            if len(stripped) > 0:
                return stripped, None
            return None, "Please enter a valid text name."
        elif val_type == "int":
            try:
                # Extract numerical digits or period
                clean_input = "".join(c for c in user_input if c.isdigit() or c == ".")
                if not clean_input:
                    return None, "Please enter a valid number."
                val = int(float(clean_input))
                min_val = current_q["min"]
                max_val = current_q["max"]
                if min_val <= val <= max_val:
                    return val, None
                else:
                    return None, f"Please enter a number between {min_val} and {max_val}."
            except Exception:
                return None, "I didn't quite catch that. Please enter a valid number."
        elif val_type == "choice":
            choices = current_q["choices"]
            clean_input = user_input.strip().lower()
            for choice in choices:
                if clean_input == choice.lower() or choice.lower() in clean_input:
                    return choice, None
            return None, f"Please reply with one of these options: {', '.join(choices)}."
        return user_input, None

    # Helper function to query Gemini with streaming, system instruction and robust fallbacks
    def stream_gemini_response(api_key, prompt_text, system_instruction=None):
        client = genai.Client(api_key=api_key)
        model = "gemma-4-26b-a4b-it"
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt_text),
                ],
            ),
        ]
        
        tools = [
            types.Tool(googleSearch=types.GoogleSearch()),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            thinking_config=types.ThinkingConfig(
                thinking_level="HIGH",
            ),
            tools=tools,
        )
        
        try:
            response_stream = client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            # Fallback to gemini-2.5-flash if the gemma model fails or isn't available
            st.warning(f"Note: Model '{model}' is not available or encountered an issue. Falling back to 'gemini-2.5-flash'...")
            try:
                fallback_config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=tools
                )
                response_stream = client.models.generate_content_stream(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=fallback_config,
                )
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text
            except Exception as fallback_e:
                yield f"\n\n❌ **Error generating suggestions:** {str(fallback_e)}. Please verify your API key and connection."

    # Header with premium coaching banner
    st.markdown("""
        <div style='background-color:#0f4c5c; color:white; border-radius:12px; padding:25px; margin-bottom:25px; position:relative; overflow:hidden;'>
            <div style='position:absolute; right:-30px; bottom:-30px; font-size:10rem; opacity:0.1; transform:rotate(15deg);'>💬</div>
            <h2 style='color:white; margin:0; font-family:"Outfit", sans-serif; font-size:1.8rem;'>💬 Aura: GenAI Academic Success Coach</h2>
            <p style='margin:10px 0 0 0; opacity:0.9; font-size:1.05rem;'>
                Aura is an elite GenAI Academic Coach for secondary school students. 
                She will guide you through an interactive questionnaire about your academic habits and environmental factors, 
                and then formulate an optimized, crisp, and candid performance report.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Force loading of environment variables freshest from project.env on every rerun
    load_dotenv("project.env", override=True)
    
    # Check project.env directly first to secure freshest live updates
    gemini_api_key = ""
    if os.path.exists("project.env"):
        with open("project.env", "r") as f:
            for line in f:
                clean_line = line.strip()
                if "=" in clean_line:
                    key_part, val_part = clean_line.split("=", 1)
                    if key_part.strip() == "GEMINI_API_KEY":
                        val = val_part.strip()
                        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                            val = val[1:-1]
                        gemini_api_key = val.strip()
                        break

    # Fallback to system environment variable
    if not gemini_api_key:
        gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    # If key is missing, placeholder, or invalid, display system offline warning stop page
    if not gemini_api_key or "DUMMY" in gemini_api_key.upper() or len(gemini_api_key) < 10:
        st.markdown("""
        <div style='background-color:#ffe3e3; border-left: 6px solid #e71d36; padding:20px; border-radius:8px; margin-bottom:20px;'>
            <h4 style='color:#c01528; margin:0;'>⚠️ AI Success Coach is Offline</h4>
            <p style='color:#c01528; margin:10px 0 0 0; font-size:1rem;'>
                The <b>Aura Success Coach</b> chatbot requires a valid Gemini API key configured in the <b><code>project.env</code></b> file to operate.
                Please define your key inside your workspace's environment file:
                <br><br>
                <code>GEMINI_API_KEY=AIzaSy...</code>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Initialize state variables
        if "chatbot_messages" not in st.session_state:
            st.session_state.chatbot_messages = [
                {"role": "assistant", "content": CHATBOT_QUESTIONS[0]["question"]}
            ]
        if "chatbot_stage" not in st.session_state:
            st.session_state.chatbot_stage = 0
        if "chatbot_answers" not in st.session_state:
            st.session_state.chatbot_answers = {}

        stage = st.session_state.chatbot_stage
        total_q = len(CHATBOT_QUESTIONS)

        # Header controls (Reset and Prefill)
        col_ctrl1, col_ctrl2 = st.columns([2, 1])
        with col_ctrl2:
            if st.button("🔄 Reset Coach", use_container_width=True, help="Wipe chat history and restart questionnaire"):
                st.session_state.chatbot_messages = [
                    {"role": "assistant", "content": CHATBOT_QUESTIONS[0]["question"]}
                ]
                st.session_state.chatbot_stage = 0
                st.session_state.chatbot_answers = {}
                st.rerun()
                
        with col_ctrl1:
            if stage < total_q:
                if st.button("⚡ Quick Prefill from Sidebar Profile", use_container_width=True, help="Skip typing and auto-fill the questionnaire using current sidebar settings"):
                    sidebar_mapping = {
                        "Student_Name": "Alex",
                        "Hours_Studied": hours,
                        "Attendance": attendance,
                        "Previous_Scores": prev_scores,
                        "Tutoring_Sessions": tutoring,
                        "Sleep_Hours": sleep,
                        "Physical_Activity": physical,
                        "Distraction_Level": "Medium",
                        "Motivation_Level": motivation,
                        "Parental_Involvement": parental
                    }
                    st.session_state.chatbot_answers = sidebar_mapping.copy()
                    
                    prefill_messages = [
                        {"role": "assistant", "content": "Hello! I'm Aura, your GenAI Academic Coach. Let's create a customized success plan for you. First, **what is your name**?"},
                        {"role": "user", "content": f"Please auto-fill my answers using my active sidebar settings:\n\n" + \
                                                   f"• **Student Name**: Alex\n" + \
                                                   f"• **Weekly Study**: {hours} hours\n" + \
                                                   f"• **School Attendance**: {attendance}%\n" + \
                                                   f"• **Previous score**: {prev_scores}%\n" + \
                                                   f"• **Tutoring sessions**: {tutoring}/month\n" + \
                                                   f"• **Sleep hours**: {sleep} hrs/night\n" + \
                                                   f"• **Physical Activity**: {physical} days/week\n" + \
                                                   f"• **Distraction level**: Medium\n" + \
                                                   f"• **Motivation level**: {motivation}\n" + \
                                                   f"• **Parental involvement**: {parental}"},
                        {"role": "assistant", "content": "Perfect! Profile loaded successfully from your sidebar configurations. I have compiled all factors. Let me analyze these dimensions..."}
                    ]
                    st.session_state.chatbot_messages = prefill_messages
                    st.session_state.chatbot_stage = total_q
                    st.rerun()

        # Display progress indicator
        if stage < total_q:
            st.markdown(f"**Questionnaire Progress ({stage}/{total_q})**")
            st.progress(stage / total_q)
        else:
            st.markdown("<span style='color:#2ec4b6; font-weight:bold;'>✓ All factors compiled! Aura Coaching Assistant is Active.</span>", unsafe_allow_html=True)
            st.markdown("---")

        # Display Chat History Container
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chatbot_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Handle Chat Input & Processing
        if stage < total_q:
            # Questionnaire stage
            user_input = st.chat_input("Type your answer here...")
            if user_input:
                current_q = CHATBOT_QUESTIONS[stage]
                parsed_val, error_msg = validate_and_parse_input(user_input, current_q)
                
                if error_msg:
                    # Append invalid input message
                    st.session_state.chatbot_messages.append({"role": "user", "content": user_input})
                    st.session_state.chatbot_messages.append({
                        "role": "assistant", 
                        "content": f"⚠️ **Invalid Input:** {error_msg}\n\n{current_q['question']}"
                    })
                    st.rerun()
                else:
                    # Save answer
                    st.session_state.chatbot_answers[current_q["key"]] = parsed_val
                    st.session_state.chatbot_messages.append({"role": "user", "content": user_input})
                    
                    next_stage = stage + 1
                    st.session_state.chatbot_stage = next_stage
                    
                    if next_stage < total_q:
                        st.session_state.chatbot_messages.append({
                            "role": "assistant", 
                            "content": CHATBOT_QUESTIONS[next_stage]["question"]
                        })
                    else:
                        st.session_state.chatbot_messages.append({
                            "role": "assistant", 
                            "content": "Superb! Questionnaire completed! Aura will now compile your statistics and generate custom suggestions..."
                        })
                    st.rerun()
        
        elif stage == total_q:
            # Compile and analyze answers, generate suggestions using Gemini!
            with st.chat_message("assistant"):
                st.markdown("🧠 **Aura is synthesizing your custom student performance roadmap...**")
                
                answers = st.session_state.chatbot_answers
                student_name = answers.get("Student_Name", "Student")
                
                # Predict scores based on answers to supply model results as context
                chat_input_dict = {
                    "Hours_Studied": answers.get("Hours_Studied", 15),
                    "Attendance": answers.get("Attendance", 85),
                    "Parental_Involvement": answers.get("Parental_Involvement", "Medium"),
                    "Access_to_Resources": resources,
                    "Extracurricular_Activities": extracurricular,
                    "Sleep_Hours": answers.get("Sleep_Hours", 7),
                    "Previous_Scores": answers.get("Previous_Scores", 75),
                    "Motivation_Level": answers.get("Motivation_Level", "Medium"),
                    "Internet_Access": internet,
                    "Tutoring_Sessions": answers.get("Tutoring_Sessions", 2),
                    "Family_Income": family_income,
                    "Teacher_Quality": teacher_quality,
                    "School_Type": school_type,
                    "Peer_Influence": peer_influence,
                    "Physical_Activity": answers.get("Physical_Activity", 3),
                    "Learning_Disabilities": disability,
                    "Parental_Education_Level": parent_edu,
                    "Distance_from_Home": distance,
                    "Gender": gender
                }
                df_chat = pd.DataFrame([chat_input_dict])
                
                # Preprocess
                df_encoded_stats_chat = pd.get_dummies(df_chat, columns=cat_features_for_model, drop_first=True)
                df_encoded_stats_chat = df_encoded_stats_chat.reindex(columns=X_encoded_stats_columns, fill_value=0)
                
                # Predict
                pred_score_chat = lr_stats_model.predict(df_encoded_stats_chat)[0]
                
                # Package other factors
                other_factors_list = []
                if resources:
                    other_factors_list.append(f"Access to resources: {resources}")
                if teacher_quality:
                    other_factors_list.append(f"Teacher quality: {teacher_quality}")
                if peer_influence:
                    other_factors_list.append(f"Peer social circle influence: {peer_influence}")
                if family_income:
                    other_factors_list.append(f"Family income: {family_income}")
                if school_type:
                    other_factors_list.append(f"School sector: {school_type}")
                if extracurricular:
                    other_factors_list.append(f"Extracurricular activities: {extracurricular}")
                if distance:
                    other_factors_list.append(f"Distance to school: {distance}")
                if gender:
                    other_factors_list.append(f"Student gender: {gender}")
                other_factors = ", ".join(other_factors_list)

                # Custom Prompt requested by user
                system_prompt_template = """Act as a GenAI Academic Coach for secondary school students.

Your task is to generate a crisp, concise, motivating, and candid student feedback report based on the student’s performance data.

Target user:
Secondary school student.

Tone rules:
- Be motivating, but not overly soft.
- Be candid and practical.
- Use simple English.
- No emojis.
- No long paragraphs.
- Do not shame the student.
- Focus on improvement, effort, and clear next steps.
- Keep the response short, structured, and easy to follow.

Input data:
Student Name: {student_name}
Exam Score: {exam_score}
Attendance: {attendance}
Hours Studied: {hours_studied}
Previous Scores: {previous_scores}
Sleep Hours: {sleep_hours}
Tutoring Sessions: {tutoring_sessions}
Physical Activity: {physical_activity}
Screen/Gaming/Distraction Level: {distraction_level}
Parental Involvement: {parental_involvement}
Motivation Level: {motivation_level}
Learning Disabilities: {learning_disabilities}
Other Relevant Factors: {other_factors}

Main objective:
Analyze the student’s current performance and generate a personalized improvement report.

Output format:

1. Performance Snapshot
Write 2 to 3 lines explaining the student’s current academic position.

2. Key Strengths
List 2 strengths based on the data.

3. Main Areas to Improve
List 2 to 3 improvement areas.
Be direct but supportive.

4. Week-by-Week Improvement Plan
Create a 4-week plan.

Week 1:
- Focus:
- Action:

Week 2:
- Focus:
- Action:

Week 3:
- Focus:
- Action:

Week 4:
- Focus:
- Action:

5. Atomic Habits to Adopt
Give 3 small daily habits the student can realistically follow.
Each habit should be simple, specific, and measurable.

Example:
- Study for 25 minutes before using the phone.
- Revise 5 questions from yesterday’s topic.
- Sleep before a fixed time on school nights.

6. Motivation Framework 1: SMART Goal
Create one SMART academic goal for the student.
It should be specific, measurable, achievable, relevant, and time-bound.

7. Motivation Framework 2: WOOP Plan
Create a WOOP plan:
- Wish:
- Outcome:
- Obstacle:
- Plan:

8. Time Management Advice
Only include this section if the student’s data shows poor study consistency, low study hours, weak attendance, high distractions, or poor task focus.

If needed, recommend:
- Pomodoro technique for focus issues.
- Eisenhower Prioritization Matrix for students struggling with too many tasks or poor planning.

Do not recommend time management methods if the student does not need them.

9. Final Coach Message
End with 2 to 3 motivating but realistic lines.
The message should make the student feel capable of improving through consistent action.

Important rules:
- Do not provide generic advice.
- Every suggestion must connect to the student’s data.
- Keep the total response concise.
- Avoid technical terms.
- Do not mention machine learning, regression, probability, or analytics to the student.
- Do not use emojis."""

                system_instruction = system_prompt_template.format(
                    student_name=student_name,
                    exam_score=f"{pred_score_chat:.1f}",
                    attendance=f"{answers.get('Attendance')}%",
                    hours_studied=f"{answers.get('Hours_Studied')}",
                    previous_scores=f"{answers.get('Previous_Scores')}%",
                    sleep_hours=f"{answers.get('Sleep_Hours')}",
                    tutoring_sessions=f"{answers.get('Tutoring_Sessions')}",
                    physical_activity=f"{answers.get('Physical_Activity')}",
                    distraction_level=answers.get("Distraction_Level", "Medium"),
                    parental_involvement=answers.get("Parental_Involvement", "Medium"),
                    motivation_level=answers.get("Motivation_Level", "Medium"),
                    learning_disabilities=disability,
                    other_factors=other_factors
                )

                st.session_state.chatbot_system_instruction = system_instruction

                placeholder = st.empty()
                full_response = ""
                trigger_prompt = "Generate my personalized student feedback report and academic improvement report."
                
                for text_chunk in stream_gemini_response(gemini_api_key, trigger_prompt, system_instruction=system_instruction):
                    full_response += text_chunk
                    placeholder.markdown(full_response)
                    
                st.session_state.chatbot_messages.append({"role": "assistant", "content": full_response})
                st.session_state.chatbot_stage = total_q + 1
                st.rerun()
                
        else:
            # Free discussion follow-up chat
            user_msg = st.chat_input("Ask Aura any follow-up questions about your study plan...")
            if user_msg:
                st.session_state.chatbot_messages.append({"role": "user", "content": user_msg})
                
                with st.chat_message("user"):
                    st.markdown(user_msg)
                    
                with st.chat_message("assistant"):
                    st.markdown("💬 **Aura is formulating a coaching response...**")
                    
                    # Construct context
                    convo_history = []
                    for m in st.session_state.chatbot_messages[-10:]:
                        convo_history.append(f"{m['role'].upper()}: {m['content']}")
                    convo_str = "\n".join(convo_history)
                    
                    chat_prompt = f"""
You are Aura, the elite AI Academic Success Coach.
You are in an active, supportive, encouraging coaching conversation with a student.
Below is the conversation history including their profile context, initial custom report, and latest follow-up question.

Conversation History:
{convo_str}

Please reply to the student's latest message in an empathetic, supportive, and pedagogical manner. Provide specific, tactical, and concrete learning and study habits, keeping your recommendations perfectly aligned with their student profile. Keep your answers focused on their academic improvement.
"""
                    placeholder = st.empty()
                    full_response = ""
                    system_instruction = st.session_state.get("chatbot_system_instruction")
                    
                    for text_chunk in stream_gemini_response(gemini_api_key, chat_prompt, system_instruction=system_instruction):
                        full_response += text_chunk
                        placeholder.markdown(full_response)
                        
                    st.session_state.chatbot_messages.append({"role": "assistant", "content": full_response})
                    st.rerun()
