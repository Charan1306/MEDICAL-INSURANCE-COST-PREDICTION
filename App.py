import numpy as np
import pandas as pd
import pickle as pkl 
import streamlit as st

# Load the model
model = pkl.load(open('MIPML.pkl', 'rb'))

# Streamlit header
st.header('Medical Insurance Premium Predictor')

# User inputs
Age = st.slider('Enter Age', 5 , 80)
Diabetes = st.selectbox('If any Diabetes', ['Yes', 'No'])
BloodPressureProblems = st.selectbox('BP Problems', ['Yes', 'No'])
AnyTransplants = st.selectbox('Any Transplants', ['Yes', 'No'])
AnyChronicDiseases = st.selectbox('Any Chronic Diseases', ['Yes', 'No'])
Height = st.slider('Enter Height (cm)', 0 , 200)
Weight = st.slider('Enter Weight (kg)', 0, 150)
KnownAllergies = st.selectbox('Known Allergies', ['Yes', 'No'])
HistoryOfCancerInFamily = st.selectbox('History Of Cancer In Family', ['Yes', 'No'])
NumberOfMajorSurgeries = st.slider('Number of Major Surgeries', 0, 3)

# Prediction button
if st.button('Predict'):
    # Convert categorical inputs to binary
    Diabetes = 0 if Diabetes == 'Yes' else 1
    BloodPressureProblems = 0 if BloodPressureProblems == 'Yes' else 1
    AnyTransplants = 0 if AnyTransplants == 'Yes' else 1
    AnyChronicDiseases = 0 if AnyChronicDiseases == 'Yes' else 1
    KnownAllergies = 0 if KnownAllergies == 'Yes' else 1
    HistoryOfCancerInFamily = 0 if HistoryOfCancerInFamily == 'Yes' else 1

    # Prepare input data
    input_data = (Age, Diabetes, BloodPressureProblems, AnyTransplants,
                  AnyChronicDiseases, Height, Weight, KnownAllergies,
                  HistoryOfCancerInFamily, NumberOfMajorSurgeries)
    
    input_data_array = np.asarray(input_data).reshape(1, -1)

    # Make prediction
    predicted_prem = model.predict(input_data_array)

    # Display result
    display_string = 'Insurance Premium will be ' + str(round(predicted_prem[0], 2)) + ' Indian Rupees'
    st.markdown(display_string)
