import pandas as pd
import pickle as pk
import numpy as np
import streamlit as st

# 🎨 Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# 🔍 Load model and data
model = pk.load(open(r'C:\Users\User\Documents\ML\HEART DX PREDICTION\heart_disease_prdx.pkl', 'rb'))
data = pd.read_csv(r'C:\Users\User\Documents\ML\HEART DX PREDICTION\heart_disease.csv')

# 🌈 Background & Custom CSS
st.markdown("""
<style>
body {
    background: linear-gradient(to bottom right, #f5f7fa, #c3cfe2);
}
.stApp {
    background: transparent;
}
div[data-testid="stForm"] {
    background-color: #ffffffdd;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0px 0px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# 🧠 App Title
st.title("🫀 AI-Powered Heart Disease Predictor")

# 📋 Collect User Inputs
with st.form("Heart Disease Form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.radio("Patient's Gender", options=["Male", "Female"])
        gen = 1 if gender == "Male" else 0
        age = st.slider("Age", 20, 100, 40)
        currentSmoker = st.radio("Are you a current smoker?", [0, 1])
        cigsPerDay = st.slider("Cigarettes per day", 0, 60, 0)

    with col2:
        BPMeds = st.radio("On BP meds?", [0, 1])
        prevalentStroke = st.radio("History of stroke?", [0, 1])
        prevalentHyp = st.radio("History of hypertension?", [0, 1])
        diabetes = st.radio("Diabetic?", [0, 1])

    with col3:
        totChol = st.number_input("Total Cholesterol (mg/dL)", min_value=100.0, max_value=500.0, value=200.0)
        sysBP = st.number_input("Systolic BP", min_value=80.0, max_value=250.0, value=120.0)
        diaBP = st.number_input("Diastolic BP", min_value=40.0, max_value=150.0, value=80.0)
        BMI = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
        heartRate = st.number_input("Heart Rate", min_value=40.0, max_value=180.0, value=72.0)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=100.0)

    # ✅ Submit Button
    submitted = st.form_submit_button("🔍 Predict")

# 🔮 Prediction Logic
if submitted:
    input_data = np.array([[gen, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke,
                            prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]])
    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success("✅ Patient is **Healthy** – No signs of heart disease 💚")
    else:
        st.error("⚠️ Risk Alert: Patient **may have heart disease**. Please consult a cardiologist. ❤️‍🩹")
