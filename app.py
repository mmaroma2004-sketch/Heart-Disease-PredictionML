import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import requests
from streamlit_lottie import st_lottie
import time

# ---------- Helpers ----------
st.set_page_config(page_title="❤️ Heart Disease Prediction", layout="wide", initial_sidebar_state="auto")

def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None

def show_result_card(title, message, probability, high_risk):
    # Card-like layout using columns and markdown
    col1, col2 = st.columns([1, 2])
    with col1:
        if high_risk:
            st.markdown(f"<div style='background:#ffccd2;padding:18px;border-radius:12px;text-align:center'>"
                        f"<h3 style='color:#8b0000;margin:0'>🚨 {title}</h3>"
                        f"<p style='color:#8b0000;margin:4px 0 0 0'>{message}</p>"
                        f"</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#d8f7e3;padding:18px;border-radius:12px;text-align:center'>"
                        f"<h3 style='color:#0b7a3b;margin:0'>✅ {title}</h3>"
                        f"<p style='color:#0b7a3b;margin:4px 0 0 0'>{message}</p>"
                        f"</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='padding:10px;border-radius:12px;background:linear-gradient(90deg,#ffffff,#f3f6ff);'>", unsafe_allow_html=True)
        st.markdown(f"**Probability (model):** {probability:.2f}")
        # progress bar animation
        progress_placeholder = st.empty()
        for i in range(1, int(probability*100)+1):
            progress_placeholder.progress(i)
            time.sleep(0.005)
        progress_placeholder.empty()
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- Load pipeline ----------

current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "lr_heart_disease_model.pkl")

try:
    lr_pipeline = joblib.load(model_path)
except Exception as e:
    st.error(f"Could not load model pipeline `{model_path}`. Error: {e}")
    st.stop()

# ---------- Lottie assets ----------
lottie_header = load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_tutvdkg0.json")  # heart animation
lottie_processing = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_j2ka6q9h.json")  # processing
lottie_success = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_jbrw3hcz.json")

# ---------- Sidebar ----------
with st.sidebar:
    st_lottie(lottie_header, height=180, key="side_heart")
    st.markdown("### About")
    st.markdown("""
    **Heart Disease Prediction App**  
    Cinematic UI: Lottie animations, cards, tabs and progress.
    """)
    st.markdown("---")
    st.markdown("Made with ❤️ — Ensure `lr_heart_disease_model.pkl` is the saved pipeline containing preprocessing + model.")

# ---------- Main - Tabs ----------
tab_home, tab_predict, tab_about = st.tabs(["🏠 Home", "🔮 Predict", "ℹ️ About"])

with tab_home:
    # Hero section
    st.markdown("<div style='display:flex;align-items:center;gap:20px'>", unsafe_allow_html=True)
    st.markdown("<div style='flex:1'>", unsafe_allow_html=True)
    st.markdown("<h1 style='margin:0'>❤️ Heart Disease Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:gray;margin-top:6px'>Enter your data or use the Predict tab — stylish UI with animations and result cards.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if lottie_header:
        st_lottie(lottie_header, height=220)
    st.markdown("</div>", unsafe_allow_html=True)

with tab_predict:
    st.subheader("🔎 Enter your health information")
    left, right = st.columns([2, 1])

    with left:
        # Inputs (grouped)
        with st.form(key="input_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                age = st.number_input("🧍 Age", min_value=1, max_value=120, value=30)
                gender = st.selectbox("🚻 Gender", options=['Male', 'Female'])
                weight = st.number_input("⚖️ Weight (kg)", min_value=1, max_value=300, value=70)
                height = st.number_input("📏 Height (cm)", min_value=50, max_value=250, value=170)
                bmi = st.number_input("📊 BMI (you can edit)", min_value=10.0, max_value=60.0, value=round(weight/((height/100)**2),2))
                smoking = st.selectbox("🚬 Smoking Status", options=['Never', 'Current', 'Former'])
                alcohol_intake = st.selectbox("🍺 Alcohol Intake", options=['unknown', 'Low', 'Moderate', 'High'])
                physical_activity = st.selectbox("🏃 Physical Activity", options=['Sedentary', 'Active', 'Moderate'])
                diet = st.selectbox("🥗 Diet Quality", options=['Healthy', 'Average', 'Unhealthy'])
                stress_level = st.selectbox("😣 Stress Level", options=['Low', 'Medium', 'High'])

            with col_b:
                hypertension = st.number_input("🩺 Hypertension (0/1)", min_value=0, max_value=1, value=0)
                diabetes = st.number_input("🩸 Diabetes (0/1)", min_value=0, max_value=1, value=0)
                hyperlipidemia = st.number_input("🧬 Hyperlipidemia (0/1)", min_value=0, max_value=1, value=0)
                family_history = st.number_input("👨‍👩‍👦 Family History (0/1)", min_value=0, max_value=1, value=0)
                previous_heart_attack = st.number_input("❤️ Previous Heart Attack (0/1)", min_value=0, max_value=1, value=0)
                systolic_bp = st.number_input("🔴 Systolic BP", min_value=50, max_value=250, value=120)
                diastolic_bp = st.number_input("🔵 Diastolic BP", min_value=30, max_value=200, value=80)
                heart_rate = st.number_input("💓 Heart Rate", min_value=30, max_value=220, value=70)
                blood_sugar_fasting = st.number_input("🧪 Blood Sugar Fasting", min_value=40, max_value=400, value=100)
                cholesterol_total = st.number_input("🧫 Total Cholesterol", min_value=80, max_value=400, value=180)

            center = st.columns(3)
            with center[1]:
                submit = st.form_submit_button("🔍 Predict Heart Disease Risk")


    with right:
        st.markdown("### Preview")
        # live preview card
        preview_df = pd.DataFrame({
            'age':[age], 'gender':[gender], 'weight':[weight], 'height':[height], 'bmi':[bmi],
            'smoking':[smoking], 'alcohol_intake':[alcohol_intake], 'physical_activity':[physical_activity],
            'diet':[diet], 'stress_level':[stress_level], 'hypertension':[hypertension], 'diabetes':[diabetes],
            'hyperlipidemia':[hyperlipidemia], 'family_history':[family_history], 'previous_heart_attack':[previous_heart_attack],
            'systolic_bp':[systolic_bp], 'diastolic_bp':[diastolic_bp], 'heart_rate':[heart_rate],
            'blood_sugar_fasting':[blood_sugar_fasting], 'cholesterol_total':[cholesterol_total]
        })
        st.table(preview_df.T.rename(columns={0:"value"}))

        if lottie_processing:
            st_lottie(lottie_processing, height=200)

    # When the user submits: run prediction and show cinematic results
    if submit:
        # build input dataframe (match training column names)
        input_data = pd.DataFrame({
            'age':[age], 'gender':[gender], 'weight':[weight], 'height':[height], 'bmi':[bmi],
            'smoking':[smoking], 'alcohol_intake':[alcohol_intake], 'physical_activity':[physical_activity],
            'diet':[diet], 'stress_level':[stress_level], 'hypertension':[hypertension], 'diabetes':[diabetes],
            'hyperlipidemia':[hyperlipidemia], 'family_history':[family_history], 'previous_heart_attack':[previous_heart_attack],
            'systolic_bp':[systolic_bp], 'diastolic_bp':[diastolic_bp], 'heart_rate':[heart_rate],
            'blood_sugar_fasting':[blood_sugar_fasting], 'cholesterol_total':[cholesterol_total]
        })

        # run model (inside spinner)
        with st.spinner("⏳ Running model and preparing cinematic result..."):
            try:
                proba = lr_pipeline.predict_proba(input_data)[0][1]
                pred = int(lr_pipeline.predict(input_data)[0])
            except Exception as e:
                st.error(f"Model prediction error: {e}")
                st.stop()
            time.sleep(0.6)

        # show big result section
        if pred == 1:
            show_result_card("High Risk", "Model indicates elevated risk for heart disease. Consult a clinician.", proba, high_risk=True)
            if lottie_success:
                st_lottie(lottie_success, height=160)
            st.warning("This is a predictive model — not a diagnosis. See a professional for medical advice.")
            st.balloons()
        else:
            show_result_card("Low Risk", "Model indicates low risk. Maintain healthy habits!", proba, high_risk=False)
            if lottie_success:
                st_lottie(lottie_success, height=160)
            st.success("Keep up the healthy lifestyle! ❤️")

with tab_about:
    st.markdown("## About this app")
    st.markdown("""
    - This app uses a saved sklearn pipeline (`lr_heart_disease_model.pkl`) that must include preprocessing (encoders/scalers) + model.
    - The UI contains Lottie animations (internet required) — if Lottie fails to load the app still works.
    - The prediction is probabilistic — always consult a medical professional for diagnosis.
    """)
    st.markdown("### Notes")
    st.markdown("- If model raises errors about missing columns, ensure the pipeline was trained with these exact column names.")
    st.markdown("- To save a pipeline in training: `joblib.dump(pipeline, 'lr_heart_disease_model.pkl')`")



