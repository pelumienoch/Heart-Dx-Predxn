import pandas as pd
import pickle as pk
import numpy as np
import streamlit as st

# I import the model
model = pk.load(open('C:\Users\User\Documents\ML\HEART DX PREDICTION\heart_disease_prdx.pkl','rb'))
data = pd.read_csv('C:\Users\User\Documents\ML\HEART DX PREDICTION\heart_disease.csv')

st.header('Heart Disease Predictor')

gender = selectionbox("Please choose patient's gender", data['Gender'].unique)
if gender == 'Male':
    gen = 1
else:
    gen = 0

age = st.number_input("Enter patient's Age")
currentSmoker = st.number_input("Are you a current smoker, Enter 1 for yes, 0 for no")
cigsPerDay = st.number_input("How many cigarette do you smoke per day")
BPMeds = st.number_input("Are you on BP meds, Enter 1 for yes, 0 for no")
prevalentStroke = st.number_input("Has patient had stroke, Enter 1 for yes, 0 for no")
prevalentHyp = st.number_input("Has patient had hypertension, Enter 1 for yes, 0 for no")
diabetes = st.number_input("Is patient diabetic, Enter 1 for yes, 0 for no")
totChol = st.number_input("Enter total Cholesterol")
sysBP = st.number_input("Enter systolic BP")
diaBP = st.number_input("Enter diastolic BP")
BMI = st.number_input("Enter BMI")
heartRate = st.number_input("Enter Heart rate")
glucose = st.number_input("Enter glucose level")

# I want to add the input to take in parameters
if st.button('Predict'):
    input = np.array([[gen,age,currentSmoker,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,
                   diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose]])
    output = model.predict(input)
    if output[0] == 0:
        stn = 'Patient is Healthy, No heart Disease'
    else:
        stn ='Patient may have some heart Disease'
    st.markdown(stn)

  