import streamlit as st
import pandas as pd
import sqlite3
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Robust check for IMBLearn / SMOTE package presence
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

from catboost import CatBoostClassifier

# ==========================================
# PAGE CONFIGURATION (RESPONSIVE ENGINE)
# ==========================================
st.set_page_config(
    page_title="Pro-Mind | Student Depression Analytics", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "username" not in st.session_state: st.session_state.username = ""
if "banner_notification" not in st.session_state: st.session_state.banner_notification = None
if "app_step" not in st.session_state: st.session_state.app_step = "upload"
if "trained_model" not in st.session_state: st.session_state.trained_model = None
if "le_gender" not in st.session_state: st.session_state.le_gender = None
if "le_department" not in st.session_state: st.session_state.le_department = None
if "summary_stats" not in st.session_state: st.session_state.summary_stats = {}
if "prediction_result" not in st.session_state: st.session_state.prediction_result = None

# ==========================================
# DEPRESSION RISK GRADIENT MAPPING
# ==========================================
bg_img_url = "https://images.unsplash.com/photo-1456513080510-7bf3a84b82f8?q=80&w=1920&auto=format&fit=crop"
accent_color = "#00f2fe" 
hover_color = "#4facfe"
glow_shadow = "rgba(0, 242, 254, 0.25)"

if st.session_state.app_step == "input" and st.session_state.prediction_result is not None:
    if st.session_state.prediction_result.get("depression_val") == 1:
        bg_img_url = "https://images.unsplash.com/photo-1516589178581-6cd7833ae3b2?q=80&w=1920&auto=format&fit=crop"
        accent_color = "#ff0055" 
        hover_color = "#ff4d88"
        glow_shadow = "rgba(255, 0, 85, 0.35)"
    else:
        bg_img_url = "https://images.unsplash.com/photo-1523240795612-9a054b0db644?q=80&w=1920&auto=format&fit=crop"
        accent_color = "#00ffaa" 
        hover_color = "#33ffbb"
        glow_shadow = "rgba(0, 255, 170, 0.35)"

# Responsive CSS Engine Injector
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght=300;400;500;600;700;800&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background-image: linear-gradient(rgba(11, 11, 28, 0.55), rgba(11, 11, 28, 0.55)), url("{bg_img_url}") !important;
    background-size: cover !important;
    background-position: center center !important;
    background-repeat: no-repeat !important;
    background-attachment: fixed !important;
    transition: background-image 0.5s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}

html, body, [data-testid="stAppViewContainer"] {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #f8fafc;
}}

[data-testid="stMainBlockContainer"] {{
    background-color: rgba(9, 9, 22, 0.84) !important;
    backdrop-filter: blur(20px) !important;
    padding: 2.5rem !important;
    border-radius: 24px !important;
    margin-top: 2rem !important;
    box-shadow: 0 25px 60px rgba(0,0,0,0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    max-width: 1350px !important;
    width: 92% !important;
}}

@media screen and (max-width: 768px) {{
    [data-testid="stMainBlockContainer"] {{
        width: 100% !important;
        padding: 1.25rem !important;
        border-radius: 0px !important;
    }}
    .kpi-container {{ flex-direction: column !important; }}
}}

[data-testid="stVerticalBlock"], [data-testid="stForm"] {{
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
}}

[data-testid="stSidebar"] {{
    background-color: rgba(5, 5, 12, 0.88) !important;
    backdrop-filter: blur(25px) !important;
}}

.main-title {{
    font-size: 3.4rem;
    font-weight: 800;
    color: {accent_color};
    text-align: center;
    text-shadow: 0px 4px 25px {glow_shadow};
}}

.sub-title {{
    color: #94a3b8;
    text-align: center;
    font-size: 1.15rem;
    margin-bottom: 2.5rem;
}}

.glass-card {{
    background: rgba(25, 25, 50, 0.35);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
}}

.kpi-container {{
    display: flex;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 25px;
}}
.kpi-box {{
    flex: 1;
    background: rgba(7, 7, 16, 0.9);
    border-left: 4px solid {accent_color};
    padding: 22px;
    border-radius: 14px;
}}
.kpi-val {{ font-size: 2.3rem; font-weight: 800; color: #ffffff; }}
.kpi-lbl {{ font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; }}

.stButton > button {{
    width: 100%;
    border-radius: 14px !important;
    height: 50px;
    font-weight: 700 !important;
    background-color: {accent_color} !important;
    color: #04060d !important;
    box-shadow: 0 4px 20px {glow_shadow} !important;
}}
.stButton > button:hover {{
    background-color: {hover_color} !important;
}}
</style>
""", unsafe_allow_html=True)

# ==========================================
# DATABASE SETTING & SYSTEM FUNCTIONS
# ==========================================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

conn = sqlite3.connect("student_database.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)")
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions(
    id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, age INTEGER, gender TEXT, department TEXT,
    cgpa REAL, sleep_duration REAL, study_hours REAL, social_media_hours REAL, physical_activity INTEGER,
    stress_level INTEGER, prediction TEXT
)
""")
conn.commit()

# =====================================
# AUTHENTICATION SCREEN
# =====================================
if not st.session_state.logged_in:
    st.markdown('<div class="main-title">🎓 ProMind Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Predictive Intelligence Infrastructure for Student Health Management</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        tabs = st.tabs(["🔐 Security Access", "📝 Registration System"])
        
        with tabs[0]:
            user_in = st.text_input("Username Identifier", key="lin_user")
            pass_in = st.text_input("Access Security Key", type="password", key="lin_pass")
            if st.button("Authenticate Identity", key="btn_login"):
                res = cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (user_in, hash_password(pass_in))).fetchone()
                if res:
                    st.session_state.logged_in = True
                    st.session_state.username = user_in
                    st.session_state.banner_notification = None
                    st.rerun()
                else:
                    st.error("Access Denied: Invalid configuration parameters.")
                    
        with tabs[1]:
            new_user = st.text_input("Configure New Username", key="reg_user")
            new_pass = st.text_input("Configure Security Key", type="password", key="reg_pass")
            if st.button("Provision Account", key="btn_reg"):
                if not new_user.strip() or not new_pass.strip():
                    st.error("Structural validation fault: Empty strings.")
                else:
                    try:
                        cursor.execute("INSERT INTO users (username, password) VALUES (?,?)", (new_user, hash_password(new_pass)))
                        conn.commit()
                        st.success("Account provisioned successfully. Move to access node.")
                    except sqlite3.IntegrityError:
                        st.error("Identity collision: Unique username allocated.")
        st.markdown('</div>', unsafe_allow_html=True)

# =====================================
# PRIMARY ANALYTICS NODE ACTIVE
# =====================================
else:
    st.markdown(f'<div class="main-title">🎓 ProMind Analytics Hub</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">Operational Node Active &mdash; Welcome, <b>{st.session_state.username}</b></div>', unsafe_allow_html=True)
    
    if st.sidebar.button("🚪 Terminate Session"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.app_step = "upload"
        st.session_state.trained_model = None
        st.session_state.prediction_result = None
        st.rerun()

    if st.session_state.app_step == "upload":
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📁 Primary Matrix Compilation (CSV Ingestion)")
        up_file = st.file_uploader("Drop target analytics record sheet below (student_lifestyle_100k.csv)", type=["csv"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        if up_file:
            df = pd.read_csv(up_file)
            df.columns = df.columns.str.strip()
            column_mapping = {col.lower(): col for col in df.columns}
            required_lower = ["age", "gender", "department", "cgpa", "sleep_duration", "study_hours", "social_media_hours", "physical_activity", "stress_level", "depression"]
            missing_cols = [c.capitalize() for c in required_lower if c not in column_mapping]
            
            if missing_cols:
                st.error(f"Ingestion validation breakdown: Missing columns {missing_cols}")
            else:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("📊 Ingested Dataset Properties (Console Matrix Output)")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**FIRST 5 ROWS**")
                    st.dataframe(df.head())
                with c2:
                    st.markdown("**MISSING VALUES LOG**")
                    st.dataframe(df.isnull().sum().to_frame(name="Missing Elements Count"))
                st.markdown('</div>', unsafe_allow_html=True)

                if st.session_state.trained_model is None:
                    with st.spinner("Processing deep network layers via CatBoost cycles..."):
                        df = df.rename(columns={column_mapping[c]: c.replace('_',' ').title().replace(' ','_') for c in required_lower})
                        if "Cgpa" in df.columns: df = df.rename(columns={"Cgpa": "CGPA"})
                        if "Student_ID" in df.columns: df.drop("Student_ID", axis=1, inplace=True)
                        
                        le_gender, le_department = LabelEncoder(), LabelEncoder()
                        df["Gender"] = le_gender.fit_transform(df["Gender"])
                        df["Department"] = le_department.fit_transform(df["Department"])
                        df["Depression"] = df["Depression"].astype(int)
                        
                        X = df.drop("Depression", axis=1)
                        y = df["Depression"]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                        
                        # Apply class balance handling based on SMOTE availability fallback
                        if HAS_SMOTE:
                            X_train_smote, y_train_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)
                            model = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, verbose=0).fit(X_train_smote, y_train_smote)
                        else:
                            model = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, auto_class_weights='Balanced', verbose=0).fit(X_train, y_train)
                        
                        predictions = model.predict(X_test)
                        accuracy = accuracy_score(y_test, predictions)
                        rep_text = classification_report(y_test, predictions)
                        cm_matrix = confusion_matrix(y_test, predictions)
                        
                        # --- COMPLETE DIRECT STANDALONE CONSOLE MIRRORING ---
                        print("\n" + "="*50)
                        print("FIRST 5 ROWS")
                        print(df.head())
                        print("\nDATASET SHAPE")
                        print(df.shape)
                        print("\nMISSING VALUES")
                        print(df.isnull().sum())
                        print("\nMODEL TRAINING COMPLETED")
                        print("\nACCURACY")
                        print(f"{round(accuracy * 100, 2)} %")
                        print("\nCLASSIFICATION REPORT")
                        print(rep_text)
                        print("\nCONFUSION MATRIX")
                        print(cm_matrix)
                        print("="*50 + "\n")
                        
                        st.session_state.trained_model = model
                        st.session_state.le_gender = le_gender
                        st.session_state.le_department = le_department
                        st.session_state.summary_stats = {
                            "rows": df.shape[0], 
                            "features": df.shape[1], 
                            "score": round(accuracy * 100, 2),
                            "report": rep_text,
                            "cm": cm_matrix
                        }

                s = st.session_state.summary_stats
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-box"><div class="kpi-val">{s['rows']}</div><div class="kpi-lbl">DATASET SHAPE (ROWS)</div></div>
                    <div class="kpi-box"><div class="kpi-val">{s['features']}</div><div class="kpi-lbl">DATASET SHAPE (COLUMNS)</div></div>
                    <div class="kpi-box"><div class="kpi-val">{s['score']}%</div><div class="kpi-lbl">ACCURACY INDEX</div></div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("📊 Engine Classification Diagnostics")
                
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.markdown("**CLASSIFICATION REPORT**")
                    st.text(s["report"])
                with col_m2:
                    st.markdown("**CONFUSION MATRIX VISUALIZATION**")
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    sns.heatmap(s["cm"], annot=True, fmt="d", cmap="Blues", ax=ax)
                    fig.patch.set_alpha(0.0)
                    ax.set_facecolor("none")
                    ax.tick_params(colors="white")
                    ax.xaxis.label.set_color("white")
                    ax.yaxis.label.set_color("white")
                    st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
                
            if st.button("⚡ Initialize Vector Prediction Stage"):
                st.session_state.app_step = "input"
                st.rerun()

    elif st.session_state.app_step == "input":
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🧠 Custom Vector Variable Parameters Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Target Group Age", 15, 40, 20)
            gender = st.selectbox("Gender Demographic", list(st.session_state.le_gender.classes_))
            department = st.selectbox("Department Focus Area", list(st.session_state.le_department.classes_))
        with col2:
            cgpa = st.slider("Cumulative CGPA Profile", 0.0, 4.0, 3.0)
            sleep_duration = st.slider("Circadian Rest (Hours/Day)", 0.0, 12.0, 7.0)
            study_hours = st.slider("Study Allocation (Hours/Day)", 0.0, 15.0, 4.0)
        with col3:
            social_media_hours = st.slider("Social Networks (Hours/Day)", 0.0, 15.0, 3.0)
            physical_activity = st.slider("Kinetic Workouts (Min/Week)", 0, 1000, 150)
            stress_level = st.slider("Internal Stress Gradient", 0, 10, 5)
        st.markdown('</div>', unsafe_allow_html=True)
            
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("🔍 Run Prediction Engine"):
                gender_encoded = st.session_state.le_gender.transform([gender])[0]
                department_encoded = st.session_state.le_department.transform([department])[0]
                
                input_data = pd.DataFrame({
                    "Age": [age], "Gender": [gender_encoded], "Department": [department_encoded], "CGPA": [cgpa],
                    "Sleep_Duration": [sleep_duration], "Study_Hours": [study_hours], "Social_Media_Hours": [social_media_hours],
                    "Physical_Activity": [physical_activity], "Stress_Level": [stress_level]
                })
                
                prediction = st.session_state.trained_model.predict(input_data)[0]
                probability = st.session_state.trained_model.predict_proba(input_data)[0]
                confidence = round(max(probability) * 100, 2)
                    
                result_text = "⚠️ High Depression Risk Detected" if prediction == 1 else "✅ Healthy Student Status Verified"
                
                # Live query inference tracing console output logging
                print("\n" + "~"*50)
                print("===== DEPRESSION PREDICTION RESULT =====")
                print(f"User Subject Context Profile: Age={age}, Gender={gender}, Department={department}, CGPA={cgpa}")
                print(f"Prediction Output Evaluation: {result_text}")
                print(f"Confidence Level: {confidence}%")
                print("~"*50 + "\n")

                st.session_state.prediction_result = {
                    "text": result_text, "confidence": confidence, "depression_val": int(prediction),
                    "raw_probabilities": { "Healthy Status (Class 0)": f"{round(probability[0]*100, 2)}%", "Depression Risk (Class 1)": f"{round(probability[1]*100, 2)}%" }
                }
                
                cursor.execute("""
                    INSERT INTO predictions (username, age, gender, department, cgpa, sleep_duration, 
                    study_hours, social_media_hours, physical_activity, stress_level, prediction) 
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """, (st.session_state.username, age, gender, department, cgpa, sleep_duration, study_hours, social_media_hours, physical_activity, stress_level, result_text))
                conn.commit()
                st.rerun()
                
        with col_b2:
            if st.button("⬅️ Clear Workspace Frame"):
                st.session_state.app_step = "upload"
                st.session_state.prediction_result = None
                st.rerun()
                
        if st.session_state.prediction_result:
            res = st.session_state.prediction_result
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f"### Diagnostic Breakdown Summary Output")
            if res['depression_val'] == 1:
                st.error(f"### {res['text']}")
            else:
                st.success(f"### {res['text']}")
                
            st.metric(label="Model Predictive Evaluation Certainty Index", value=f"{res['confidence']}%")
            st.write("#### Engine Probability Arrays Configuration Matrix:")
            st.json(res['raw_probabilities'])
            st.markdown('</div>', unsafe_allow_html=True)