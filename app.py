"""
app.py — Diabetes Prediction Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered",
)

# ── Load model ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with gzip.open("model.pkl", "rb") as f:
        return pickle.load(f)

try:
    bundle = load_model()
except FileNotFoundError:
    st.error("⚠️ Model file `model.pkl` not found. Run `python train_model.py` first.")
    st.stop()

model        = bundle["model"]
scaler       = bundle["scaler"]
le_gender    = bundle["le_gender"]
le_smoking   = bundle["le_smoking"]
best_thresh  = bundle["best_thresh"]
feature_cols = bundle["feature_cols"]

# ── Header ────────────────────────────────────────────────────────
st.title("🩺 Diabetes Risk Prediction")
st.markdown(
    "Enter your health information below to check your diabetes risk. "
    "This tool uses a **Random Forest** model trained on 96,000+ patient records."
)
st.divider()

# ── Input Form ───────────────────────────────────────────────────
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    age    = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
    bmi    = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1,
                             help="Body Mass Index (weight kg / height m²)")
    hypertension = st.selectbox("Hypertension", [0, 1],
                                format_func=lambda x: "Yes" if x else "No")

with col2:
    heart_disease = st.selectbox("Heart Disease", [0, 1],
                                 format_func=lambda x: "Yes" if x else "No")
    smoking_history = st.selectbox(
        "Smoking History",
        ["never", "No Info", "current", "former", "ever", "not current"]
    )
    hba1c = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0,
                             value=5.5, step=0.1,
                             help="Glycated haemoglobin — normal range: 4–5.6%")
    glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=50, max_value=400,
                               value=100, step=1,
                               help="Fasting blood glucose — normal: < 100 mg/dL")

st.divider()

# ── Prediction ───────────────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True, type="primary"):
    # Encode inputs
    gender_enc  = le_gender.transform([gender])[0]
    smoking_enc = le_smoking.transform([smoking_history])[0]

    input_data = pd.DataFrame([{
        "gender"              : gender_enc,
        "age"                 : age,
        "hypertension"        : hypertension,
        "heart_disease"       : heart_disease,
        "smoking_history"     : smoking_enc,
        "bmi"                 : bmi,
        "HbA1c_level"         : hba1c,
        "blood_glucose_level" : glucose,
    }])[feature_cols]

    input_scaled = scaler.transform(input_data)
    prob         = model.predict_proba(input_scaled)[0][1]
    
    # Hitung persentase probabilitas
    prob_percentage = prob * 100

    st.subheader("Result")

    # Pembagian menjadi 5 kelompok risiko
    if prob_percentage <= 10.0:
        st.success(
            f"### 🟢 Very Low Diabetes Risk\n"
            f"Predicted probability: **{prob_percentage:.1f}%**\n\n"
            "Excellent! Your clinical indicators are optimal. Keep maintaining your healthy lifestyle!"
        )
    elif prob_percentage <= 30.0:
        st.info(
            f"### 🔵 Low Diabetes Risk\n"
            f"Predicted probability: **{prob_percentage:.1f}%**\n\n"
            "Your risk is low. Continue keeping a balanced diet and regular physical activity."
        )
    elif prob_percentage <= 60.0:
        st.warning(
            f"### 🟡 Medium Diabetes Risk\n"
            f"Predicted probability: **{prob_percentage:.1f}%**\n\n"
            "Moderate risk detected. Consider monitoring your sugar intake and lifestyle habits."
        )
    elif prob_percentage <= 85.0:
        st.error(
            f"### 🟠 High Diabetes Risk\n"
            f"Predicted probability: **{prob_percentage:.1f}%**\n\n"
            "High risk detected. It is highly recommended to consult a doctor for a medical check-up."
        )
    else:
        st.error(
            f"### 🔴 Very High Diabetes Risk\n"
            f"Predicted probability: **{prob_percentage:.1f}%**\n\n"
            "Very high risk detected. Please seek medical evaluation from a healthcare professional immediately."
        )


    # Probability bar
    st.metric("Risk Probability", f"{prob*100:.1f}%")
    st.progress(float(prob))

# ── Footer ───────────────────────────────────────────────────────
st.divider()
st.caption(
    "Diabetes Prediction App · Built with Streamlit · "
    "Dataset: [Kaggle Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)"
)
