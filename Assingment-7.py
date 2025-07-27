import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("glass_model.pkl")

st.title("Glass Type Predictor")

# Inputs
RI = st.number_input("Refractive Index (RI)", 1.4, 1.6, step=0.01, value=1.52)
Na = st.number_input("Sodium (Na)", 0.0, 20.0, step=0.1, value=13.0)
Mg = st.number_input("Magnesium (Mg)", 0.0, 5.0, step=0.1, value=2.0)
Al = st.number_input("Aluminum (Al)", 0.0, 5.0, step=0.1, value=1.5)
Si = st.number_input("Silicon (Si)", 60.0, 80.0, step=0.1, value=72.0)
K = st.number_input("Potassium (K)", 0.0, 6.0, step=0.1, value=0.5)
Ca = st.number_input("Calcium (Ca)", 0.0, 20.0, step=0.1, value=9.0)
Ba = st.number_input("Barium (Ba)", 0.0, 3.0, step=0.1, value=0.0)
Fe = st.number_input("Iron (Fe)", 0.0, 1.0, step=0.01, value=0.1)

features = np.array([[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]])

if st.button("Predict"):
    prediction = model.predict(features)[0]
    glass_types = {
        1: "Building Windows Float Processed",
        2: "Building Windows Non-Float Processed",
        3: "Vehicle Windows Float Processed",
        4: "Vehicle Windows Non-Float Processed",
        5: "Containers",
        6: "Tableware",
        7: "Headlamps"
    }
    result = glass_types.get(prediction, "Unknown Type")
    st.success(f"Predicted Glass Type: {result}")
