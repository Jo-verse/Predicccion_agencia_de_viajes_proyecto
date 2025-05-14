### plantilla básica en Streamlit para mostrar tu modelo de predicción en una interfaz web sencilla y visual. Podrás ejecutarla fácilmente y enseñarla como demo de tu proyecto.

import streamlit as st
import pandas as pd
import pickle

# Carga del modelo
model = pickle.load(open('models/spam_detector.sav', 'rb'))  # Cambia el nombre si tu modelo tiene otro

st.title("🧳 Travel Booking Prediction App")
st.markdown("Predict whether a customer will complete a booking based on their data.")

# Entradas del usuario
age = st.slider("Age", 18, 100, 30)
channel = st.selectbox("Booking Channel", ["Web", "Agency", "Phone"])
companions = st.slider("Number of Companions", 0, 10, 1)
duration = st.slider("Trip Duration (days)", 1, 30, 7)

# Codificación de variables (ejemplo simplificado)
channel_map = {"Web": 0, "Agency": 1, "Phone": 2}
channel_encoded = channel_map[channel]

# Crear DataFrame con los datos
input_data = pd.DataFrame({
    "age": [age],
    "channel": [channel_encoded],
    "companions": [companions],
    "duration": [duration]
})

# Predicción
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Likely to complete booking (probability: {prob:.2f})")
    else:
        st.error(f"❌ Unlikely to complete booking (probability: {prob:.2f})")
