import pandas as pd
import pickle as pk
import numpy as np
import streamlit as st
import plotly.express as px

# Load model and dataset
model = pk.load(open(r'C:\Users\User\Documents\ML\HEART DX PREDICTION\heart_disease_prdx.pkl', 'rb'))
data = pd.read_csv(r'C:\Users\User\Documents\ML\HEART DX PREDICTION\heart_disease.csv')

st.set_page_config(page_title="Heart Disease App", layout="wide")

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["🔍 Predictor", "📊 Dashboard"])

# 💻 Predictor Page
if page == "🔍 Predictor":
    st.title("🫀 Heart Disease Predictor")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.radio("Gender", ["Male", "Female"])
            gen = 1 if gender == "Male" else 0
            age = st.slider("Age", 20, 100, 45)
            currentSmoker = st.radio("Smoker?", [0, 1])
            cigsPerDay = st.slider("Cigarettes/day", 0, 50, 0)
        with col2:
            BPMeds = st.radio("On BP meds?", [0, 1])
            prevalentStroke = st.radio("Had stroke?", [0, 1])
            prevalentHyp = st.radio("Hypertension?", [0, 1])
            diabetes = st.radio("Diabetic?", [0, 1])
        with col3:
            totChol = st.number_input("Total Cholesterol", 100.0, 600.0, 240.0)
            sysBP = st.number_input("Systolic BP", 90.0, 250.0, 120.0)
            diaBP = st.number_input("Diastolic BP", 60.0, 150.0, 80.0)
            BMI = st.number_input("BMI", 15.0, 50.0, 25.0)
            heartRate = st.number_input("Heart Rate", 40.0, 180.0, 72.0)
            glucose = st.number_input("Glucose Level", 50.0, 300.0, 100.0)

        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        input_data = np.array([[gen, age, currentSmoker, cigsPerDay, BPMeds,
                                prevalentStroke, prevalentHyp, diabetes,
                                totChol, sysBP, diaBP, BMI, heartRate, glucose]])
        prediction = model.predict(input_data)

        if prediction[0] == 0:
            st.success("✅ Patient is Healthy")
        else:
            st.error("⚠️ Patient may have Heart Disease")


# 📊 Dashboard Page
elif page == "📊 Dashboard":
    st.title("📊 Heart Disease Dashboard")

    st.subheader("🧠 Patient Summary")
    col1, col2, col3 = st.columns(3)

    # Convert gender if necessary
    if data['Gender'].dtype != 'object':
        data['Gender'] = data['Gender'].map({1: 'Male', 0: 'Female'})

    col1.metric("Total Patients", len(data))
    col2.metric("Smokers", data['currentSmoker'].sum())
    col3.metric("Diabetics", data['diabetes'].sum())

    # Pie chart: Gender
    st.subheader("👤 Gender Distribution")
    gender_counts = data['Gender'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'count']  # Rename for clarity
    fig1 = px.pie(gender_counts, names='Gender', values='count', title='Gender Split')
    st.plotly_chart(fig1, use_container_width=True)


    # Histogram: Age
    st.subheader("📈 Age Distribution")
    fig2 = px.histogram(data, x='age', nbins=20, title='Patient Age Distribution')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🔎 Explore Any Two Variables")
    # Select numeric columns
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    # Let user select X and Y axes
    col_x = st.selectbox("Select X-axis variable", numeric_cols, index=numeric_cols.index('age') if 'age' in numeric_cols else 0)
    col_y = st.selectbox("Select Y-axis variable", numeric_cols, index=numeric_cols.index('totChol') if 'totChol' in numeric_cols else 1)
    # Create dynamic scatter plot
    fig3 = px.scatter(data, x=col_x, y=col_y, color='Gender',
                  title=f"{col_y} vs {col_x}", trendline='ols')
    st.plotly_chart(fig3, use_container_width=True)



st.markdown("---")
st.markdown("""
### ⚠️ Disclaimer
This application is built **solely for educational purposes** and **should not be used as a medical diagnostic tool**.  
It is powered by a **machine learning algorithm** trained on a historical dataset and achieves an **R² score of approximately 0.65**, meaning its predictions are accurate to a degree of **65%**.  
Always consult a licensed medical professional for any health concerns or decisions.

© 2025 **Pelumi Fasulu**. All rights reserved.
""")
